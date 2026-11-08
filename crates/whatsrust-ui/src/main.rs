//! WhatsRust GPUI Desktop Interface
//!
//! Connects to the `whatsrust` daemon via REST + SSE (`/api/events`), displaying:
//! - Incoming messages & chats filtered by WhatsApp Tag/Label (`LabelEditAction` / `LabelAssociationAction`)
//! - AI Rolling Thread Summaries ("Resume")
//! - Project classification (`triage.toml` projects)
//! - CLI Subprocess Agent execution (`Run CLI Agent`) and 1-click `Approve & Send` to WhatsApp.

use std::io::{BufRead, BufReader};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, Application, Bounds, ClipboardItem, Context, Keystroke, MouseButton, MouseUpEvent,
    SharedString, Timer, Window, WindowBounds, WindowOptions, div, prelude::*, px, rgb, size,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data models (matching whatsrust daemon JSON schemas)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LabelRow {
    pub label_id: String,
    pub name: String,
    pub color: i32,
    pub source: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTriageRow {
    pub chat_jid: String,
    pub chat_name: String,
    pub last_message_id: String,
    pub unread_count: i64,
    pub rolling_summary: String,
    pub project_id: Option<String>,
    pub project_confidence: f64,
    pub matched_agent_id: Option<String>,
    pub agent_prompt: Option<String>,
    pub agent_status: String,
    pub agent_draft: Option<String>,
    pub agent_error: Option<String>,
    pub updated_at: i64,
    #[serde(default)]
    pub labels: Vec<LabelRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundRow {
    pub id: i64,
    pub chat_jid: String,
    pub sender_jid: String,
    pub message_id: String,
    pub content_kind: String,
    pub body_text: Option<String>,
    pub timestamp: i64,
    pub media_path: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TriageFileConfig {
    #[serde(default)]
    pub triage: TriageGeneralConfig,
    #[serde(default)]
    pub projects: Vec<ProjectConfig>,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TriageGeneralConfig {
    pub enabled: bool,
    pub debounce_ms: u64,
    pub auto_transcribe_voice: bool,
    pub filter_tags: Vec<String>,
    pub include_all_dms: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub id: String,
    pub name: String,
    pub workdir: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub jids: Vec<String>,
    pub default_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub auto_run: bool,
    #[serde(default)]
    pub timeout_secs: u64,
}

// ---------------------------------------------------------------------------
// Background Daemon Client (SSE + REST worker thread)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum DaemonCommand {
    RefreshAll,
    FetchChatHistory(String),
    RetriageChat(String),
    RunCliAgent {
        jid: String,
        agent_id: Option<String>,
        prompt: Option<String>,
    },
    ApproveDraft {
        jid: String,
        text: String,
    },
    DismissChat(String),
    AssignTag {
        jid: String,
        tag: String,
        labeled: bool,
    },
    ToggleIncludeAllDms(bool),
    ReloadConfig,
    SimulateInbound {
        jid: String,
        sender_name: String,
        text: String,
        tag: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct SharedDaemonState {
    pub generation: u64,
    pub daemon_connected: bool,
    pub wa_state: String,
    pub status_banner: String,
    pub labels: Vec<LabelRow>,
    pub triages: Vec<ChatTriageRow>,
    pub config: TriageFileConfig,
    pub selected_chat_messages: Vec<InboundRow>,
    pub selected_chat_jid_for_history: Option<String>,
}

fn spawn_daemon_worker(
    base_url: String,
    api_token: Option<String>,
    shared: Arc<Mutex<SharedDaemonState>>,
) -> Sender<DaemonCommand> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<DaemonCommand>();

    // Thread 1: Command & REST polling worker
    let base_cmd = base_url.clone();
    let token_cmd = api_token.clone();
    let shared_cmd = shared.clone();
    std::thread::spawn(move || {
        // Initial fetch
        refresh_all(&base_cmd, token_cmd.as_deref(), &shared_cmd);

        loop {
            match cmd_rx.recv_timeout(Duration::from_secs(5)) {
                Ok(cmd) => {
                    handle_daemon_command(&base_cmd, token_cmd.as_deref(), &shared_cmd, cmd);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    refresh_all(&base_cmd, token_cmd.as_deref(), &shared_cmd);
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });

    // Thread 2: Live SSE `/api/events` stream listener
    let base_sse = base_url;
    let token_sse = api_token;
    let shared_sse = shared;
    let cmd_tx_sse = cmd_tx.clone();
    std::thread::spawn(move || loop {
        let events_url = format!("{}/api/events", base_sse.trim_end_matches('/'));
        let mut req = ureq::get(&events_url).header("Accept", "text/event-stream");
        if let Some(ref tok) = token_sse {
            req = req.header("Authorization", &format!("Bearer {tok}"));
        }

        match req.call() {
            Ok(mut resp) => {
                {
                    let mut st = shared_sse.lock();
                    st.daemon_connected = true;
                    st.generation += 1;
                }
                let reader = BufReader::new(resp.body_mut().as_reader());
                let mut current_event = String::new();
                let mut current_data = String::new();

                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    if line.is_empty() {
                        if !current_event.is_empty() {
                            process_sse_event(
                                &current_event,
                                &current_data,
                                &shared_sse,
                                &cmd_tx_sse,
                            );
                        }
                        current_event.clear();
                        current_data.clear();
                    } else if let Some(ev) = line.strip_prefix("event: ") {
                        current_event = ev.trim().to_string();
                    } else if let Some(dat) = line.strip_prefix("data: ") {
                        if !current_data.is_empty() {
                            current_data.push('\n');
                        }
                        current_data.push_str(dat);
                    }
                }
            }
            Err(_) => {
                // Status polling thread (Thread 1) handles daemon_connected / wa_state
            }
        }
        std::thread::sleep(Duration::from_secs(3));
    });

    cmd_tx
}

fn process_sse_event(
    event: &str,
    data: &str,
    shared: &Arc<Mutex<SharedDaemonState>>,
    cmd_tx: &Sender<DaemonCommand>,
) {
    match event {
        "triage_update" => {
            if let Ok(row) = serde_json::from_str::<ChatTriageRow>(data) {
                let mut st = shared.lock();
                if let Some(existing) = st.triages.iter_mut().find(|t| t.chat_jid == row.chat_jid) {
                    *existing = row.clone();
                } else {
                    st.triages.insert(0, row.clone());
                }
                st.triages.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                st.status_banner = format!(
                    "Updated triage for {} ({})",
                    row.chat_name, row.agent_status
                );
                st.generation += 1;
            }
        }
        "label_update" => {
            let _ = cmd_tx.send(DaemonCommand::RefreshAll);
        }
        "inbound" => {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(jid) = val["jid"].as_str() {
                    let selected_jid = shared.lock().selected_chat_jid_for_history.clone();
                    if selected_jid.as_deref() == Some(jid) {
                        let _ = cmd_tx.send(DaemonCommand::FetchChatHistory(jid.to_string()));
                    }
                }
            }
        }
        _ => {}
    }
}

fn handle_daemon_command(
    base_url: &str,
    token: Option<&str>,
    shared: &Arc<Mutex<SharedDaemonState>>,
    cmd: DaemonCommand,
) {
    match cmd {
        DaemonCommand::RefreshAll => refresh_all(base_url, token, shared),
        DaemonCommand::FetchChatHistory(jid) => {
            let url = format!("{}/api/history?jid={}&limit=20", base_url, jid);
            if let Ok(val) = http_get_json(&url, token) {
                if let Ok(mut msgs) =
                    serde_json::from_value::<Vec<InboundRow>>(val["messages"].clone())
                {
                    msgs.reverse();
                    let mut st = shared.lock();
                    st.selected_chat_jid_for_history = Some(jid);
                    st.selected_chat_messages = msgs;
                    st.generation += 1;
                }
            }
        }
        DaemonCommand::RetriageChat(jid) => {
            set_banner(shared, &format!("Re-summarizing {jid} with AI..."));
            let url = format!("{}/api/triage/retriage", base_url);
            match http_post_json(&url, token, &serde_json::json!({ "jid": jid })) {
                Ok(val) => {
                    if let Ok(row) = serde_json::from_value::<ChatTriageRow>(val["triage"].clone())
                    {
                        upsert_triage_in_shared(shared, row, "AI Summary updated.");
                    }
                }
                Err(e) => set_banner(shared, &format!("Retriage failed: {e}")),
            }
        }
        DaemonCommand::RunCliAgent {
            jid,
            agent_id,
            prompt,
        } => {
            set_banner(
                shared,
                &format!("Spawning CLI agent for {jid} in project directory..."),
            );
            let url = format!("{}/api/triage/run-agent", base_url);
            let payload = serde_json::json!({
                "jid": jid,
                "agent_id": agent_id,
                "prompt": prompt,
            });
            match http_post_json(&url, token, &payload) {
                Ok(val) => {
                    if let Ok(row) = serde_json::from_value::<ChatTriageRow>(val["triage"].clone())
                    {
                        upsert_triage_in_shared(shared, row, "CLI Agent draft ready!");
                    }
                }
                Err(e) => set_banner(shared, &format!("CLI Agent error: {e}")),
            }
        }
        DaemonCommand::ApproveDraft { jid, text } => {
            set_banner(shared, &format!("Sending approved reply to {jid}..."));
            let url = format!("{}/api/triage/approve", base_url);
            let payload = serde_json::json!({ "jid": jid, "text": text });
            match http_post_json(&url, token, &payload) {
                Ok(val) => {
                    if let Ok(row) = serde_json::from_value::<ChatTriageRow>(val["triage"].clone())
                    {
                        upsert_triage_in_shared(shared, row, "Reply sent to WhatsApp!");
                    }
                }
                Err(e) => set_banner(shared, &format!("Send failed: {e}")),
            }
        }
        DaemonCommand::DismissChat(jid) => {
            let url = format!("{}/api/triage/dismiss", base_url);
            if let Ok(val) = http_post_json(&url, token, &serde_json::json!({ "jid": jid })) {
                if let Ok(row) = serde_json::from_value::<ChatTriageRow>(val["triage"].clone()) {
                    upsert_triage_in_shared(shared, row, "Marked chat as handled.");
                }
            }
        }
        DaemonCommand::AssignTag { jid, tag, labeled } => {
            let url = format!("{}/api/labels/assign", base_url);
            let payload = serde_json::json!({
                "jid": jid,
                "tag": tag,
                "labeled": labeled,
            });
            let _ = http_post_json(&url, token, &payload);
            refresh_all(base_url, token, shared);
        }
        DaemonCommand::ToggleIncludeAllDms(include) => {
            let url = format!("{}/api/triage/config", base_url);
            let payload = serde_json::json!({ "include_all_dms": include });
            if let Ok(val) = http_post_json(&url, token, &payload) {
                if let Ok(cfg) = serde_json::from_value::<TriageFileConfig>(val["config"].clone()) {
                    let mut st = shared.lock();
                    st.config = cfg;
                    st.generation += 1;
                }
            }
        }
        DaemonCommand::ReloadConfig => {
            let url = format!("{}/api/triage/config", base_url);
            if let Ok(val) = http_post_json(&url, token, &serde_json::json!({ "reload": true })) {
                if let Ok(cfg) = serde_json::from_value::<TriageFileConfig>(val["config"].clone()) {
                    let mut st = shared.lock();
                    st.config = cfg;
                    st.status_banner = "Reloaded triage.toml configuration.".into();
                    st.generation += 1;
                }
            }
        }
        DaemonCommand::SimulateInbound {
            jid,
            sender_name,
            text,
            tag,
        } => {
            set_banner(shared, "Simulating incoming tagged WhatsApp message...");
            let url = format!("{}/api/triage/simulate", base_url);
            let payload = serde_json::json!({
                "jid": jid,
                "sender_name": sender_name,
                "text": text,
                "tag": tag,
            });
            match http_post_json(&url, token, &payload) {
                Ok(_) => {
                    refresh_all(base_url, token, shared);
                    set_banner(shared, "Simulated message triaged!");
                }
                Err(e) => set_banner(shared, &format!("Simulation failed: {e}")),
            }
        }
    }
}

fn upsert_triage_in_shared(
    shared: &Arc<Mutex<SharedDaemonState>>,
    row: ChatTriageRow,
    banner: &str,
) {
    let mut st = shared.lock();
    if let Some(existing) = st.triages.iter_mut().find(|t| t.chat_jid == row.chat_jid) {
        *existing = row;
    } else {
        st.triages.insert(0, row);
    }
    st.status_banner = banner.to_string();
    st.generation += 1;
}

fn set_banner(shared: &Arc<Mutex<SharedDaemonState>>, msg: &str) {
    let mut st = shared.lock();
    st.status_banner = msg.to_string();
    st.generation += 1;
}

fn refresh_all(base_url: &str, token: Option<&str>, shared: &Arc<Mutex<SharedDaemonState>>) {
    let status_url = format!("{}/api/status", base_url);
    let wa_state = match http_get_json(&status_url, token) {
        Ok(v) => v["state"].as_str().unwrap_or("Unknown").to_string(),
        Err(_) => {
            let mut st = shared.lock();
            st.daemon_connected = false;
            st.wa_state = "Daemon Offline".into();
            st.generation += 1;
            return;
        }
    };

    let labels = http_get_json(&format!("{}/api/labels", base_url), token)
        .ok()
        .and_then(|v| serde_json::from_value::<Vec<LabelRow>>(v["labels"].clone()).ok())
        .unwrap_or_default();

    let triages = http_get_json(&format!("{}/api/triage", base_url), token)
        .ok()
        .and_then(|v| serde_json::from_value::<Vec<ChatTriageRow>>(v["items"].clone()).ok())
        .unwrap_or_default();

    let config = http_get_json(&format!("{}/api/triage/config", base_url), token)
        .ok()
        .and_then(|v| serde_json::from_value::<TriageFileConfig>(v["config"].clone()).ok())
        .unwrap_or_default();

    let selected_jid = shared.lock().selected_chat_jid_for_history.clone();
    let messages = if let Some(ref jid) = selected_jid {
        let h_url = format!("{}/api/history?jid={}&limit=20", base_url, jid);
        http_get_json(&h_url, token)
            .ok()
            .and_then(|v| serde_json::from_value::<Vec<InboundRow>>(v["messages"].clone()).ok())
            .map(|mut m| {
                m.reverse();
                m
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut st = shared.lock();
    st.daemon_connected = true;
    st.wa_state = wa_state;
    st.labels = labels;
    st.triages = triages;
    st.config = config;
    if selected_jid.is_some() {
        st.selected_chat_messages = messages;
    }
    st.generation += 1;
}

fn http_get_json(url: &str, token: Option<&str>) -> anyhow::Result<serde_json::Value> {
    let mut req = ureq::get(url);
    if let Some(tok) = token {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }
    let mut resp = req.call()?;
    let text = resp.body_mut().read_to_string()?;
    let val: serde_json::Value = serde_json::from_str(&text)?;
    Ok(val)
}

fn http_post_json(
    url: &str,
    token: Option<&str>,
    body: &serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let mut req = ureq::post(url).header("Content-Type", "application/json");
    if let Some(tok) = token {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }
    let payload = serde_json::to_vec(body)?;
    let mut resp = req.send(&payload[..])?;
    let text = resp.body_mut().read_to_string()?;
    let val: serde_json::Value = serde_json::from_str(&text)?;
    Ok(val)
}

// ---------------------------------------------------------------------------
// GPUI View & Application State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveEditor {
    None,
    Prompt,
    Draft,
}

struct WhatsRustTriageApp {
    shared: Arc<Mutex<SharedDaemonState>>,
    cmd_tx: Sender<DaemonCommand>,
    last_seen_generation: u64,

    // Synced state
    daemon_connected: bool,
    wa_state: String,
    status_banner: String,
    labels: Vec<LabelRow>,
    triages: Vec<ChatTriageRow>,
    config: TriageFileConfig,
    selected_chat_messages: Vec<InboundRow>,

    // UI Filters & Selection
    selected_tag_filter: Option<String>,
    selected_project_filter: Option<String>,
    selected_chat_jid: Option<String>,
    selected_agent_override: Option<String>,

    // Inline editors for Agent Prompt & Draft Reply
    active_editor: ActiveEditor,
    prompt_buffer: String,
    draft_buffer: String,
    sim_counter: usize,
}

impl WhatsRustTriageApp {
    fn new(shared: Arc<Mutex<SharedDaemonState>>, cmd_tx: Sender<DaemonCommand>) -> Self {
        Self {
            shared,
            cmd_tx,
            last_seen_generation: 0,
            daemon_connected: false,
            wa_state: "Connecting...".into(),
            status_banner: "Ready — listening for tagged WhatsApp messages.".into(),
            labels: Vec::new(),
            triages: Vec::new(),
            config: TriageFileConfig::default(),
            selected_chat_messages: Vec::new(),
            selected_tag_filter: None,
            selected_project_filter: None,
            selected_chat_jid: None,
            selected_agent_override: None,
            active_editor: ActiveEditor::None,
            prompt_buffer: String::new(),
            draft_buffer: String::new(),
            sim_counter: 0,
        }
    }

    fn sync_from_shared(&mut self) -> bool {
        let guard = self.shared.lock();
        if guard.generation == self.last_seen_generation {
            return false;
        }
        self.last_seen_generation = guard.generation;
        self.daemon_connected = guard.daemon_connected;
        self.wa_state = guard.wa_state.clone();
        if !guard.status_banner.is_empty() {
            self.status_banner = guard.status_banner.clone();
        }
        self.labels = guard.labels.clone();
        self.triages = guard.triages.clone();
        self.config = guard.config.clone();
        self.selected_chat_messages = guard.selected_chat_messages.clone();
        drop(guard);

        // Auto-select first filtered chat if none selected
        if self.selected_chat_jid.is_none() {
            let first_jid = self.filtered_triages().first().map(|f| f.chat_jid.clone());
            if let Some(jid) = first_jid {
                self.select_chat(&jid);
            }
        } else if let Some(ref sel_jid) = self.selected_chat_jid.clone() {
            if let Some(row) = self.triages.iter().find(|t| &t.chat_jid == sel_jid) {
                if self.active_editor != ActiveEditor::Prompt {
                    if let Some(ref p) = row.agent_prompt {
                        self.prompt_buffer = p.clone();
                    }
                }
                if self.active_editor != ActiveEditor::Draft {
                    if let Some(ref d) = row.agent_draft {
                        self.draft_buffer = d.clone();
                    }
                }
            }
        }
        true
    }

    fn select_chat(&mut self, jid: &str) {
        self.selected_chat_jid = Some(jid.to_string());
        self.active_editor = ActiveEditor::None;
        if let Some(row) = self.triages.iter().find(|t| t.chat_jid == jid) {
            self.prompt_buffer = row.agent_prompt.clone().unwrap_or_default();
            self.draft_buffer = row.agent_draft.clone().unwrap_or_default();
            self.selected_agent_override = row.matched_agent_id.clone();
        }
        let _ = self
            .cmd_tx
            .send(DaemonCommand::FetchChatHistory(jid.to_string()));
    }

    fn handle_keystroke(&mut self, ks: &Keystroke) -> bool {
        if self.active_editor == ActiveEditor::None {
            return false;
        }
        let target = match self.active_editor {
            ActiveEditor::Prompt => &mut self.prompt_buffer,
            ActiveEditor::Draft => &mut self.draft_buffer,
            ActiveEditor::None => return false,
        };

        match ks.key.as_str() {
            "backspace" => {
                target.pop();
                true
            }
            "enter" => {
                target.push('\n');
                true
            }
            "escape" => {
                self.active_editor = ActiveEditor::None;
                true
            }
            "space" => {
                target.push(' ');
                true
            }
            _ => {
                if !ks.modifiers.control && !ks.modifiers.platform && !ks.modifiers.alt {
                    if let Some(ref ch) = ks.key_char {
                        target.push_str(ch);
                        return true;
                    }
                }
                false
            }
        }
    }

    fn filtered_triages(&self) -> Vec<ChatTriageRow> {
        let include_all = self.config.triage.include_all_dms;
        self.triages
            .iter()
            .filter(|t| {
                if !include_all && t.labels.is_empty() {
                    let is_explicit_jid = self
                        .config
                        .projects
                        .iter()
                        .any(|p| p.jids.iter().any(|j| t.chat_jid.contains(j.trim_end_matches("@s.whatsapp.net"))));
                    if !is_explicit_jid {
                        return false;
                    }
                }
                let tag_ok = match &self.selected_tag_filter {
                    Some(tag) => t.labels.iter().any(|l| {
                        l.name.eq_ignore_ascii_case(tag) || l.label_id.eq_ignore_ascii_case(tag)
                    }),
                    None => true,
                };
                let proj_ok = match &self.selected_project_filter {
                    Some(pid) => t.project_id.as_deref() == Some(pid.as_str()),
                    None => true,
                };
                tag_ok && proj_ok
            })
            .cloned()
            .collect()
    }
}

fn tag_color_hex(color_idx: i32) -> u32 {
    match color_idx.rem_euclid(8) {
        0 => 0x3b82f6, // blue
        1 => 0x10b981, // emerald
        2 => 0xf59e0b, // amber
        3 => 0xef4444, // red
        4 => 0x8b5cf6, // violet
        5 => 0xec4899, // pink
        6 => 0x06b6d4, // cyan
        _ => 0x84cc16, // lime
    }
}

fn status_badge_color(status: &str) -> u32 {
    match status {
        "draft_ready" => 0x10b981,   // green
        "running_agent" => 0xf59e0b, // amber
        "triaging" => 0x3b82f6,      // blue
        "sent" => 0x6366f1,          // indigo
        "failed" => 0xef4444,        // red
        _ => 0x64748b,               // slate
    }
}

impl Render for WhatsRustTriageApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let filtered = self.filtered_triages();
        let selected_row = self
            .selected_chat_jid
            .as_ref()
            .and_then(|jid| self.triages.iter().find(|t| &t.chat_jid == jid))
            .cloned();

        let daemon_online = self.daemon_connected;
        let wa_state = self.wa_state.clone();
        let banner = self.status_banner.clone();

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x0f172a))
            .text_color(rgb(0xf8fafc))
            .text_sm()
            // 1. Top Header Bar
            .child(self.render_header(daemon_online, &wa_state, &banner, cx))
            // 2. Main 3-Pane Body
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_1()
                    .w_full()
                    .overflow_hidden()
                    .child(self.render_left_sidebar(cx))
                    .child(self.render_center_feed(&filtered, selected_row.as_ref(), cx))
                    .child(self.render_right_inspector(selected_row.as_ref(), cx)),
            )
    }
}

impl WhatsRustTriageApp {
    fn render_header(
        &self,
        daemon_online: bool,
        wa_state: &str,
        banner: &str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let status_color = if daemon_online { 0x10b981 } else { 0xef4444 };

        div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .px_4()
            .py_2()
            .bg(rgb(0x1e293b))
            .border_b_1()
            .border_color(rgb(0x334155))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_base()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0xffffff))
                            .child("WhatsRust • AI Triage & Agent Router"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap_1()
                            .px_2()
                            .py_0p5()
                            .rounded_md()
                            .bg(rgb(0x0f172a))
                            .border_1()
                            .border_color(rgb(status_color))
                            .child(div().size_2().rounded_full().bg(rgb(status_color)))
                            .child(format!("WA: {wa_state}")),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x94a3b8))
                            .child(banner.to_string()),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .id("btn-simulate")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(rgb(0x4f46e5))
                            .hover(|s| s.bg(rgb(0x6366f1)).cursor_pointer())
                            .text_xs()
                            .child("+ Simulate Tagged Msg")
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                    let samples = [
                                        (
                                            "5511988881111@s.whatsapp.net",
                                            "Lucas (Tech Lead)",
                                            "Work",
                                            "Hey! In whatsrust, does the new GPUI interface support reading WhatsApp Business chat labels and spawning our Claude CLI agent?",
                                        ),
                                        (
                                            "5511977772222@s.whatsapp.net",
                                            "Marina (Product)",
                                            "Urgent",
                                            "We just received a voice note from the client about the bridge reconnection metrics. Can the agent check how reconnect_count works?",
                                        ),
                                        (
                                            "5511966663333@s.whatsapp.net",
                                            "Carlos (DevOps)",
                                            "Work",
                                            "Quick question on whatsrust SQLite WAL mode: where is inbound_messages stored and how do we run a backup?",
                                        ),
                                    ];
                                    let (jid, sender, tag, text) =
                                        samples[this.sim_counter % samples.len()];
                                    this.sim_counter += 1;
                                    this.selected_chat_jid = Some(jid.to_string());
                                    let _ = this.cmd_tx.send(DaemonCommand::SimulateInbound {
                                        jid: jid.into(),
                                        sender_name: sender.into(),
                                        text: text.into(),
                                        tag: tag.into(),
                                    });
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(
                        div()
                            .id("btn-reload-cfg")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(rgb(0x334155))
                            .hover(|s| s.bg(rgb(0x475569)).cursor_pointer())
                            .text_xs()
                            .child("Reload triage.toml")
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, _| {
                                    let _ = this.cmd_tx.send(DaemonCommand::ReloadConfig);
                                }),
                            ),
                    )
                    .child(
                        div()
                            .id("btn-refresh")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(rgb(0x334155))
                            .hover(|s| s.bg(rgb(0x475569)).cursor_pointer())
                            .text_xs()
                            .child("Refresh")
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, _| {
                                    let _ = this.cmd_tx.send(DaemonCommand::RefreshAll);
                                }),
                            ),
                    ),
            )
    }

    fn render_left_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let include_all_dms = self.config.triage.include_all_dms;
        let selected_tag = self.selected_tag_filter.clone();
        let selected_proj = self.selected_project_filter.clone();

        div()
            .id("left-sidebar")
            .w(px(250.0))
            .h_full()
            .overflow_y_scroll()
            .flex()
            .flex_col()
            .gap_4()
            .p_3()
            .bg(rgb(0x111827))
            .border_r_1()
            .border_color(rgb(0x1f2937))
            // Filter Mode Section
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child("TRIAGE SCOPE"),
                    )
                    .child(
                        div()
                            .id("toggle-all-dms")
                            .flex()
                            .flex_row()
                            .items_center()
                            .justify_between()
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .bg(if include_all_dms {
                                rgb(0x1e3a8a)
                            } else {
                                rgb(0x1f2937)
                            })
                            .hover(|s| s.opacity(0.9).cursor_pointer())
                            .child("Include All DMs")
                            .child(if include_all_dms { "ON" } else { "OFF" })
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                    let _ = this
                                        .cmd_tx
                                        .send(DaemonCommand::ToggleIncludeAllDms(!include_all_dms));
                                }),
                            ),
                    ),
            )
            // WhatsApp Tags / Lists Section
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child("WHATSAPP TAGS / LISTS"),
                    )
                    .child(
                        div()
                            .id("tag-filter-all")
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .bg(if selected_tag.is_none() {
                                rgb(0x334155)
                            } else {
                                rgb(0x111827)
                            })
                            .hover(|s| s.bg(rgb(0x1f2937)).cursor_pointer())
                            .child(format!("All Tags ({})", self.triages.len()))
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                    this.selected_tag_filter = None;
                                    cx.notify();
                                }),
                            ),
                    )
                    .children(self.labels.iter().enumerate().map(|(idx, lbl)| {
                        let tag_name = lbl.name.clone();
                        let is_selected = selected_tag
                            .as_ref()
                            .is_some_and(|s| s.eq_ignore_ascii_case(&tag_name));
                        let dot_color = tag_color_hex(lbl.color);
                        let count = self
                            .triages
                            .iter()
                            .filter(|t| {
                                t.labels
                                    .iter()
                                    .any(|l| l.name.eq_ignore_ascii_case(&tag_name))
                            })
                            .count();

                        div()
                            .id(SharedString::from(format!("tag-pill-{idx}")))
                            .flex()
                            .flex_row()
                            .items_center()
                            .justify_between()
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .bg(if is_selected {
                                rgb(0x334155)
                            } else {
                                rgb(0x111827)
                            })
                            .hover(|s| s.bg(rgb(0x1f2937)).cursor_pointer())
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap_2()
                                    .child(div().size_2().rounded_full().bg(rgb(dot_color)))
                                    .child(tag_name.clone()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x94a3b8))
                                    .child(count.to_string()),
                            )
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(move |this, _: &MouseUpEvent, _, cx| {
                                    this.selected_tag_filter = Some(tag_name.clone());
                                    cx.notify();
                                }),
                            )
                    })),
            )
            // Projects Section
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child("PROJECTS (triage.toml)"),
                    )
                    .child(
                        div()
                            .id("proj-filter-all")
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .bg(if selected_proj.is_none() {
                                rgb(0x334155)
                            } else {
                                rgb(0x111827)
                            })
                            .hover(|s| s.bg(rgb(0x1f2937)).cursor_pointer())
                            .child("All Projects")
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                    this.selected_project_filter = None;
                                    cx.notify();
                                }),
                            ),
                    )
                    .children(self.config.projects.iter().enumerate().map(|(idx, proj)| {
                        let pid = proj.id.clone();
                        let pname = proj.name.clone();
                        let is_selected = selected_proj.as_deref() == Some(pid.as_str());
                        let count = self
                            .triages
                            .iter()
                            .filter(|t| t.project_id.as_deref() == Some(pid.as_str()))
                            .count();

                        div()
                            .id(SharedString::from(format!("proj-pill-{idx}")))
                            .flex()
                            .flex_row()
                            .items_center()
                            .justify_between()
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .bg(if is_selected {
                                rgb(0x334155)
                            } else {
                                rgb(0x111827)
                            })
                            .hover(|s| s.bg(rgb(0x1f2937)).cursor_pointer())
                            .child(pname)
                            .child(
                                div()
                                    .px_1p5()
                                    .rounded_sm()
                                    .bg(rgb(0x1e293b))
                                    .text_xs()
                                    .text_color(rgb(0x38bdf8))
                                    .child(count.to_string()),
                            )
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(move |this, _: &MouseUpEvent, _, cx| {
                                    this.selected_project_filter = Some(pid.clone());
                                    cx.notify();
                                }),
                            )
                    })),
            )
    }

    fn render_center_feed(
        &self,
        filtered: &[ChatTriageRow],
        selected_row: Option<&ChatTriageRow>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .id("center-feed")
            .flex_1()
            .h_full()
            .flex()
            .flex_col()
            .overflow_hidden()
            .bg(rgb(0x0f172a))
            // Top half: Triaged Chat Cards with AI Rolling Summary
            .child(
                div()
                    .id("triage-cards-scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .p_4()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child(format!(
                                "INCOMING TRIAGED CHATS & AI RESUME ({})",
                                filtered.len()
                            )),
                    )
                    .when(filtered.is_empty(), |this| {
                        this.child(
                            div()
                                .p_6()
                                .rounded_lg()
                                .bg(rgb(0x1e293b))
                                .border_1()
                                .border_color(rgb(0x334155))
                                .text_color(rgb(0x94a3b8))
                                .child("No triaged messages match the current filter. Click '+ Simulate Tagged Msg' in the top bar or send a WhatsApp message to a tagged chat!"),
                        )
                    })
                    .children(filtered.iter().enumerate().map(|(idx, item)| {
                        let jid = item.chat_jid.clone();
                        let is_selected = self.selected_chat_jid.as_deref() == Some(jid.as_str());
                        let status_hex = status_badge_color(&item.agent_status);
                        let conf_pct = (item.project_confidence * 100.0).round() as i32;

                        div()
                            .id(SharedString::from(format!("triage-card-{idx}")))
                            .flex()
                            .flex_col()
                            .gap_2()
                            .p_3()
                            .rounded_lg()
                            .bg(if is_selected {
                                rgb(0x1e293b)
                            } else {
                                rgb(0x111827)
                            })
                            .border_1()
                            .border_color(if is_selected {
                                rgb(0x6366f1)
                            } else {
                                rgb(0x1f2937)
                            })
                            .hover(|s| s.border_color(rgb(0x475569)).cursor_pointer())
                            // Row 1: Chat Name + Status Badge
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .items_center()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .font_weight(gpui::FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child(item.chat_name.clone()),
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(rgb(0x64748b))
                                                    .child(item.chat_jid.clone()),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(rgb(status_hex))
                                            .text_xs()
                                            .text_color(rgb(0xffffff))
                                            .child(item.agent_status.clone()),
                                    ),
                            )
                            // Row 2: WhatsApp Tags + Matched Project + Matched Agent badges
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .flex_wrap()
                                    .gap_2()
                                    .children(item.labels.iter().map(|lbl| {
                                        let c = tag_color_hex(lbl.color);
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(rgb(0x0f172a))
                                            .border_1()
                                            .border_color(rgb(c))
                                            .text_xs()
                                            .text_color(rgb(0xe2e8f0))
                                            .child(format!("#{}", lbl.name))
                                    }))
                                    .when_some(item.project_id.clone(), |this, pid| {
                                        this.child(
                                            div()
                                                .px_2()
                                                .py_0p5()
                                                .rounded_sm()
                                                .bg(rgb(0x1e3a8a))
                                                .text_xs()
                                                .text_color(rgb(0x93c5fd))
                                                .child(format!("Project: {pid} ({conf_pct}%)")),
                                        )
                                    })
                                    .when_some(item.matched_agent_id.clone(), |this, aid| {
                                        this.child(
                                            div()
                                                .px_2()
                                                .py_0p5()
                                                .rounded_sm()
                                                .bg(rgb(0x312e81))
                                                .text_xs()
                                                .text_color(rgb(0xc7d2fe))
                                                .child(format!("Agent: {aid}")),
                                        )
                                    }),
                            )
                            // Row 3: AI Rolling Summary ("Resume")
                            .child(
                                div()
                                    .p_2()
                                    .rounded_md()
                                    .bg(rgb(0x0f172a))
                                    .border_l_2()
                                    .border_color(rgb(0x38bdf8))
                                    .text_color(rgb(0xe2e8f0))
                                    .child(format!("AI Resume: {}", item.rolling_summary)),
                            )
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(move |this, _: &MouseUpEvent, _, cx| {
                                    this.select_chat(&jid);
                                    cx.notify();
                                }),
                            )
                    })),
            )
            // Bottom panel: Recent Messages in Selected Chat
            .child(
                div()
                    .id("chat-history-panel")
                    .h(px(210.0))
                    .w_full()
                    .overflow_y_scroll()
                    .p_3()
                    .bg(rgb(0x111827))
                    .border_t_1()
                    .border_color(rgb(0x1f2937))
                    .flex()
                    .flex_col()
                    .gap_1p5()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child(match selected_row {
                                Some(r) => {
                                    format!("RECENT MESSAGES IN {} ({})", r.chat_name, r.chat_jid)
                                }
                                None => "RECENT MESSAGES (Select a chat above)".to_string(),
                            }),
                    )
                    .children(self.selected_chat_messages.iter().map(|m| {
                        let body = m.body_text.clone().unwrap_or_else(|| "[media]".into());
                        div()
                            .p_1p5()
                            .rounded_sm()
                            .bg(rgb(0x1e293b))
                            .text_xs()
                            .child(format!("[{}] {}: {}", m.content_kind, m.sender_jid, body))
                    })),
            )
    }

    fn render_right_inspector(
        &self,
        selected_row: Option<&ChatTriageRow>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let Some(row) = selected_row else {
            return div()
                .id("right-inspector-empty")
                .w(px(390.0))
                .h_full()
                .p_4()
                .bg(rgb(0x111827))
                .border_l_1()
                .border_color(rgb(0x1f2937))
                .child("Select a triaged chat to inspect project details and run CLI agents.");
        };

        let jid = row.chat_jid.clone();
        let matched_project = row
            .project_id
            .as_ref()
            .and_then(|pid| self.config.projects.iter().find(|p| &p.id == pid));

        let active_agent_id = self
            .selected_agent_override
            .clone()
            .or_else(|| row.matched_agent_id.clone())
            .or_else(|| self.config.agents.first().map(|a| a.id.clone()));

        let matched_agent = active_agent_id
            .as_ref()
            .and_then(|aid| self.config.agents.iter().find(|a| &a.id == aid));

        let is_prompt_focused = self.active_editor == ActiveEditor::Prompt;
        let is_draft_focused = self.active_editor == ActiveEditor::Draft;

        let quick_tags = ["Work", "Clients", "Urgent", "Dev"];

        div()
            .id("right-inspector")
            .w(px(390.0))
            .h_full()
            .overflow_y_scroll()
            .p_4()
            .bg(rgb(0x111827))
            .border_l_1()
            .border_color(rgb(0x1f2937))
            .flex()
            .flex_col()
            .gap_3()
            // Section 1: Chat Tags Manager
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child("ASSIGN / TOGGLE WHATSAPP TAG"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .flex_wrap()
                            .gap_1p5()
                            .children(quick_tags.into_iter().enumerate().map(|(idx, qt)| {
                                let has_tag = row
                                    .labels
                                    .iter()
                                    .any(|l| l.name.eq_ignore_ascii_case(qt));
                                let jid_clone = jid.clone();
                                let tag_str = qt.to_string();

                                div()
                                    .id(SharedString::from(format!("assign-tag-{idx}")))
                                    .px_2()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(if has_tag {
                                        rgb(0x10b981)
                                    } else {
                                        rgb(0x1f2937)
                                    })
                                    .hover(|s| s.opacity(0.85).cursor_pointer())
                                    .text_xs()
                                    .child(if has_tag {
                                        format!("✓ #{qt}")
                                    } else {
                                        format!("+ #{qt}")
                                    })
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                            let _ =
                                                this.cmd_tx.send(DaemonCommand::AssignTag {
                                                    jid: jid_clone.clone(),
                                                    tag: tag_str.clone(),
                                                    labeled: !has_tag,
                                                });
                                        }),
                                    )
                            })),
                    ),
            )
            // Section 2: Matched Project Info
            .child(
                div()
                    .p_2p5()
                    .rounded_md()
                    .bg(rgb(0x1e293b))
                    .border_1()
                    .border_color(rgb(0x334155))
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x38bdf8))
                            .child(match matched_project {
                                Some(p) => format!("PROJECT: {} ({})", p.name, p.id),
                                None => "PROJECT: Unassigned".to_string(),
                            }),
                    )
                    .when_some(matched_project, |this, p| {
                        this.child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x94a3b8))
                                .child(format!("Workdir: {}", p.workdir)),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(rgb(0xcbdaef))
                                .child(p.description.clone()),
                        )
                    }),
            )
            // Section 3: CLI Agent Selector
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::BOLD)
                            .text_color(rgb(0x94a3b8))
                            .child("CLI SUBPROCESS AGENT"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .flex_wrap()
                            .gap_1p5()
                            .children(self.config.agents.iter().enumerate().map(|(idx, ag)| {
                                let aid = ag.id.clone();
                                let is_sel = active_agent_id.as_deref() == Some(aid.as_str());
                                div()
                                    .id(SharedString::from(format!("select-agent-{idx}")))
                                    .px_2()
                                    .py_1()
                                    .rounded_sm()
                                    .bg(if is_sel {
                                        rgb(0x4f46e5)
                                    } else {
                                        rgb(0x1f2937)
                                    })
                                    .hover(|s| s.opacity(0.9).cursor_pointer())
                                    .text_xs()
                                    .child(ag.name.clone())
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, cx| {
                                            this.selected_agent_override = Some(aid.clone());
                                            cx.notify();
                                        }),
                                    )
                            })),
                    )
                    .when_some(matched_agent, |this, ag| {
                        this.child(
                            div()
                                .text_xs()
                                .text_color(rgb(0x64748b))
                                .child(format!("Cmd: {} {}", ag.command, ag.args.join(" "))),
                        )
                    }),
            )
            // Section 4: Editable Agent Prompt + Run CLI Agent Button
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(rgb(0x94a3b8))
                                    .child(if is_prompt_focused {
                                        "AGENT PROMPT (Typing active — Esc to done)"
                                    } else {
                                        "AGENT PROMPT (Click box to edit)"
                                    }),
                            )
                            .child(
                                div()
                                    .id("btn-paste-prompt")
                                    .px_2()
                                    .py_0p5()
                                    .rounded_sm()
                                    .bg(rgb(0x1f2937))
                                    .hover(|s| s.bg(rgb(0x334155)).cursor_pointer())
                                    .text_xs()
                                    .child("Paste")
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                            if let Some(item) = cx.read_from_clipboard() {
                                                if let Some(txt) = item.text() {
                                                    this.prompt_buffer = txt;
                                                    cx.notify();
                                                }
                                            }
                                        }),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .id("prompt-editor-box")
                            .p_2()
                            .min_h(px(64.0))
                            .rounded_md()
                            .bg(rgb(0x0f172a))
                            .border_1()
                            .border_color(if is_prompt_focused {
                                rgb(0x38bdf8)
                            } else {
                                rgb(0x334155)
                            })
                            .hover(|s| s.cursor_pointer())
                            .text_xs()
                            .child(if self.prompt_buffer.is_empty() {
                                "(Click to enter custom prompt for CLI agent...)".to_string()
                            } else {
                                self.prompt_buffer.clone()
                            })
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                    this.active_editor = ActiveEditor::Prompt;
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .gap_2()
                            .child({
                                let jid_run = jid.clone();
                                let agent_run = active_agent_id.clone();
                                let prompt_run = if self.prompt_buffer.trim().is_empty() {
                                    None
                                } else {
                                    Some(self.prompt_buffer.clone())
                                };
                                div()
                                    .id("btn-run-cli-agent")
                                    .flex_1()
                                    .py_1p5()
                                    .px_3()
                                    .rounded_md()
                                    .bg(rgb(0x4f46e5))
                                    .hover(|s| s.bg(rgb(0x6366f1)).cursor_pointer())
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(rgb(0xffffff))
                                    .child(if row.agent_status == "running_agent" {
                                        "Running CLI Agent..."
                                    } else {
                                        "▶ Run CLI Agent in Project"
                                    })
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                            let _ =
                                                this.cmd_tx.send(DaemonCommand::RunCliAgent {
                                                    jid: jid_run.clone(),
                                                    agent_id: agent_run.clone(),
                                                    prompt: prompt_run.clone(),
                                                });
                                        }),
                                    )
                            })
                            .child({
                                let jid_re = jid.clone();
                                div()
                                    .id("btn-retriage")
                                    .py_1p5()
                                    .px_2p5()
                                    .rounded_md()
                                    .bg(rgb(0x334155))
                                    .hover(|s| s.bg(rgb(0x475569)).cursor_pointer())
                                    .child("↻ Re-Triage")
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                            let _ = this
                                                .cmd_tx
                                                .send(DaemonCommand::RetriageChat(jid_re.clone()));
                                        }),
                                    )
                            }),
                    ),
            )
            // Error banner if CLI agent failed
            .when_some(row.agent_error.clone(), |this, err_text| {
                this.child(
                    div()
                        .p_2()
                        .rounded_md()
                        .bg(rgb(0x450a0a))
                        .border_1()
                        .border_color(rgb(0xef4444))
                        .text_xs()
                        .text_color(rgb(0xfca5a5))
                        .child(format!("Agent Error: {err_text}")),
                )
            })
            // Section 5: Editable Draft Reply & 1-Click Approve & Send
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(rgb(0x10b981))
                                    .child(if is_draft_focused {
                                        "WHATSAPP DRAFT REPLY (Typing active — Esc to done)"
                                    } else {
                                        "WHATSAPP DRAFT REPLY (Click box to edit)"
                                    }),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .gap_1()
                                    .child(
                                        div()
                                            .id("btn-copy-draft")
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(rgb(0x1f2937))
                                            .hover(|s| s.bg(rgb(0x334155)).cursor_pointer())
                                            .text_xs()
                                            .child("Copy")
                                            .on_mouse_up(
                                                MouseButton::Left,
                                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                                    cx.write_to_clipboard(
                                                        ClipboardItem::new_string(
                                                            this.draft_buffer.clone(),
                                                        ),
                                                    );
                                                    this.status_banner =
                                                        "Copied draft to clipboard.".into();
                                                    cx.notify();
                                                }),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .id("btn-paste-draft")
                                            .px_2()
                                            .py_0p5()
                                            .rounded_sm()
                                            .bg(rgb(0x1f2937))
                                            .hover(|s| s.bg(rgb(0x334155)).cursor_pointer())
                                            .text_xs()
                                            .child("Paste")
                                            .on_mouse_up(
                                                MouseButton::Left,
                                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                                    if let Some(item) = cx.read_from_clipboard() {
                                                        if let Some(txt) = item.text() {
                                                            this.draft_buffer = txt;
                                                            cx.notify();
                                                        }
                                                    }
                                                }),
                                            ),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .id("draft-editor-box")
                            .p_2()
                            .min_h(px(90.0))
                            .rounded_md()
                            .bg(rgb(0x0f172a))
                            .border_1()
                            .border_color(if is_draft_focused {
                                rgb(0x10b981)
                            } else {
                                rgb(0x334155)
                            })
                            .hover(|s| s.cursor_pointer())
                            .text_xs()
                            .child(if self.draft_buffer.is_empty() {
                                "(No draft yet — run CLI agent or click here to type a reply...)"
                                    .to_string()
                            } else {
                                self.draft_buffer.clone()
                            })
                            .on_mouse_up(
                                MouseButton::Left,
                                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                                    this.active_editor = ActiveEditor::Draft;
                                    cx.notify();
                                }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .gap_2()
                            .mt_1()
                            .child({
                                let jid_send = jid.clone();
                                let draft_send = self.draft_buffer.clone();
                                div()
                                    .id("btn-approve-send")
                                    .flex_1()
                                    .py_2()
                                    .px_3()
                                    .rounded_md()
                                    .bg(rgb(0x059669))
                                    .hover(|s| s.bg(rgb(0x10b981)).cursor_pointer())
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .text_color(rgb(0xffffff))
                                    .child("✓ Approve & Send via WhatsApp")
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                            let _ =
                                                this.cmd_tx.send(DaemonCommand::ApproveDraft {
                                                    jid: jid_send.clone(),
                                                    text: draft_send.clone(),
                                                });
                                        }),
                                    )
                            })
                            .child({
                                let jid_dis = jid;
                                div()
                                    .id("btn-dismiss")
                                    .py_2()
                                    .px_3()
                                    .rounded_md()
                                    .bg(rgb(0x334155))
                                    .hover(|s| s.bg(rgb(0x475569)).cursor_pointer())
                                    .child("Dismiss")
                                    .on_mouse_up(
                                        MouseButton::Left,
                                        cx.listener(move |this, _: &MouseUpEvent, _, _| {
                                            let _ = this
                                                .cmd_tx
                                                .send(DaemonCommand::DismissChat(jid_dis.clone()));
                                        }),
                                    )
                            }),
                    ),
            )
    }
}

fn main() {
    let port = std::env::var("WHATSRUST_PORT")
        .or_else(|_| std::env::var("HEALTH_PORT"))
        .unwrap_or_else(|_| "7270".to_string());
    let base_url = std::env::var("WHATSRUST_API_URL")
        .unwrap_or_else(|_| format!("http://127.0.0.1:{port}"));
    let api_token = std::env::var("WHATSRUST_API_TOKEN")
        .ok()
        .filter(|s| !s.is_empty());

    let shared = Arc::new(Mutex::new(SharedDaemonState::default()));
    let cmd_tx = spawn_daemon_worker(base_url, api_token, shared.clone());

    Application::new().run(move |cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1240.0), px(780.0)), cx);

        let shared_win = shared.clone();
        let cmd_tx_win = cmd_tx.clone();

        let window = cx
            .open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                move |window, cx| {
                    let view =
                        cx.new(|_| WhatsRustTriageApp::new(shared_win.clone(), cmd_tx_win.clone()));

                    // Periodic 200ms sync from background SSE/REST worker thread
                    let view_poll = view.clone();
                    window
                        .spawn(cx, async move |cx| loop {
                            Timer::after(Duration::from_millis(200)).await;
                            let res = cx.update(|_, cx| {
                                view_poll.update(cx, |app, cx| {
                                    if app.sync_from_shared() {
                                        cx.notify();
                                    }
                                });
                            });
                            if res.is_err() {
                                break;
                            }
                        })
                        .detach();

                    view
                },
            )
            .expect("failed to open GPUI window");

        let view_keys = window.update(cx, |_, _, cx| cx.entity()).unwrap();
        cx.observe_keystrokes(move |ev, _, cx| {
            view_keys.update(cx, |app, cx| {
                if app.handle_keystroke(&ev.keystroke) {
                    cx.notify();
                }
            });
        })
        .detach();

        cx.activate(true);
    });
}
