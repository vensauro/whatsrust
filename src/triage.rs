//! AI Message Triage & CLI Subprocess Agent Engine.
//!
//! Responsibilities:
//! - Load and hot-reload `triage.toml` (Projects, CLI Subprocess Agents, Tag Filter Rules, LLM Config).
//! - Subscribe to `BridgeEvent::Inbound`, auto-transcribe audio messages via `TranscriptionQueue`,
//!   and coalesce rapid message bursts per chat using a debounce timer.
//! - Filter chats by WhatsApp Tag/Label (`filter_tags`), explicit project JIDs, or `include_all_dms`.
//! - Call an OpenAI-compatible `/chat/completions` endpoint (or deterministic fallback if no API key
//!   is configured) to produce a rolling thread summary ("resume"), assign a project, and match an
//!   AI agent that can answer it.
//! - Spawn CLI subprocess agents inside the matched project's `workdir` (either automatically or on
//!   explicit request from GPUI/API) and store the resulting draft reply for 1-click approval.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::bridge::{InboundContent, WhatsAppInbound};
use crate::bridge_events::{BridgeEvent, LabelUpdateEvent};
use crate::storage::{ChatTriageRow, InboundRow, LabelRow, Store};
use crate::transcribe::{TranscriptionOptions, TranscriptionQueue};

/// Root structure of `triage.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageFileConfig {
    #[serde(default)]
    pub triage: TriageGeneralConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub projects: Vec<ProjectConfig>,
    #[serde(default)]
    pub agents: Vec<AgentConfig>,
}

impl Default for TriageFileConfig {
    fn default() -> Self {
        Self {
            triage: TriageGeneralConfig::default(),
            llm: LlmConfig::default(),
            projects: vec![ProjectConfig {
                id: "whatsrust".into(),
                name: "WhatsRust Bridge & GPUI".into(),
                workdir: std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .to_string_lossy()
                    .to_string(),
                description: "Pure Rust WhatsApp Web bridge, MCP server, and GPUI triage desktop interface.".into(),
                tags: vec!["Work".into(), "Dev".into()],
                jids: vec![
                    "557791229001@s.whatsapp.net".into(),
                    "5577991229001@s.whatsapp.net".into(),
                    "557791229001".into(),
                ],
                keywords: vec!["whatsrust".into(), "gpui".into(), "whatsapp".into(), "bridge".into(), "rust".into()],
                default_agent: Some("claude-cli".into()),
            }],
            agents: vec![
                AgentConfig {
                    id: "claude-cli".into(),
                    name: "Claude Code CLI".into(),
                    description: "Spawns Claude Code in headless print mode inside the project directory to answer technical questions or inspect code.".into(),
                    command: "claude".into(),
                    args: vec!["-p".into(), "{{prompt}}".into()],
                    auto_run: false,
                    timeout_secs: 120,
                },
                AgentConfig {
                    id: "quick-echo".into(),
                    name: "Quick Draft Helper (Shell)".into(),
                    description: "Fast local shell responder for testing subprocess execution in the project directory.".into(),
                    command: if cfg!(windows) { "powershell".into() } else { "sh".into() },
                    args: if cfg!(windows) {
                        vec!["-NoProfile".into(), "-Command".into(), "Write-Output \"[Draft for {{project_id}}] Regarding: {{summary}}\"".into()]
                    } else {
                        vec!["-c".into(), "echo \"[Draft for {{project_id}}] Regarding: {{summary}}\"".into()]
                    },
                    auto_run: false,
                    timeout_secs: 30,
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageGeneralConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_debounce_ms")]
    pub debounce_ms: u64,
    #[serde(default = "default_true")]
    pub auto_transcribe_voice: bool,
    /// WhatsApp label names or IDs that trigger automatic AI triage.
    /// Empty list means all labeled chats (or all chats if include_all_dms is true).
    #[serde(default = "default_filter_tags")]
    pub filter_tags: Vec<String>,
    #[serde(default)]
    pub include_all_dms: bool,
}

impl Default for TriageGeneralConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debounce_ms: default_debounce_ms(),
            auto_transcribe_voice: true,
            filter_tags: default_filter_tags(),
            include_all_dms: false,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_debounce_ms() -> u64 {
    2500
}
fn default_filter_tags() -> Vec<String> {
    vec!["Work".into(), "Clients".into(), "Urgent".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,
    #[serde(default = "default_llm_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_llm_model")]
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: default_llm_base_url(),
            api_key_env: default_llm_api_key_env(),
            model: default_llm_model(),
        }
    }
}

fn default_llm_base_url() -> String {
    std::env::var("TR_LLM_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta/openai".into())
}
fn default_llm_api_key_env() -> String {
    "OPENAI_API_KEY".into()
}
fn default_llm_model() -> String {
    std::env::var("TR_LLM_MODEL").unwrap_or_else(|_| "gemini-3-flash-preview".into())
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
    #[serde(default)]
    pub keywords: Vec<String>,
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
    #[serde(default = "default_agent_timeout")]
    pub timeout_secs: u64,
}

fn default_agent_timeout() -> u64 {
    120
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmTriageOutput {
    pub rolling_summary: String,
    pub project_id: Option<String>,
    #[serde(default)]
    pub project_confidence: f64,
    pub matched_agent_id: Option<String>,
    pub suggested_agent_prompt: Option<String>,
    pub quick_reply_draft: Option<String>,
}

/// Coordinates debounced per-chat AI triage and CLI subprocess agent execution.
pub struct TriageEngine {
    store: Store,
    event_tx: broadcast::Sender<Arc<BridgeEvent>>,
    transcription_queue: Arc<TranscriptionQueue>,
    config: Arc<RwLock<TriageFileConfig>>,
    config_path: PathBuf,
    pending_debounces: Arc<DashMap<String, tokio::task::AbortHandle>>,
    chat_names: Arc<DashMap<String, String>>,
}

impl TriageEngine {
    /// Create and start the background triage engine.
    pub fn start(
        store: Store,
        event_tx: broadcast::Sender<Arc<BridgeEvent>>,
        transcription_queue: Arc<TranscriptionQueue>,
    ) -> Arc<Self> {
        let config_path = std::env::var("WHATSRUST_TRIAGE_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("triage.toml"));

        let cfg = load_or_create_config(&config_path);
        let engine = Arc::new(Self {
            store,
            event_tx,
            transcription_queue,
            config: Arc::new(RwLock::new(cfg)),
            config_path,
            pending_debounces: Arc::new(DashMap::new()),
            chat_names: Arc::new(DashMap::new()),
        });

        // Seed config tags & JID associations into SQLite
        let eng_seed = engine.clone();
        tokio::spawn(async move {
            eng_seed.sync_config_labels().await;
        });

        // Subscribe to inbound messages on the broadcast bus
        let eng_loop = engine.clone();
        let mut rx = engine.event_tx.subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(evt) => {
                        if let BridgeEvent::Inbound(ref msg) = *evt {
                            eng_loop.on_inbound_message(msg.clone()).await;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "triage engine lagged on event bus");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        engine
    }

    /// Return a snapshot of the current `TriageFileConfig`.
    pub fn config(&self) -> TriageFileConfig {
        self.config.read().clone()
    }

    /// Reload `triage.toml` from disk (or update runtime settings).
    pub async fn reload_config(self: &Arc<Self>) -> Result<TriageFileConfig> {
        let content = std::fs::read_to_string(&self.config_path)
            .with_context(|| format!("failed to read {}", self.config_path.display()))?;
        let parsed: TriageFileConfig = toml::from_str(&content)
            .with_context(|| format!("failed to parse {}", self.config_path.display()))?;
        *self.config.write() = parsed.clone();
        self.sync_config_labels().await;
        Ok(parsed)
    }

    /// Update `include_all_dms` or `filter_tags` at runtime and save to `triage.toml`.
    pub async fn update_filter_settings(
        &self,
        include_all_dms: Option<bool>,
        filter_tags: Option<Vec<String>>,
    ) -> Result<TriageFileConfig> {
        let updated = {
            let mut guard = self.config.write();
            if let Some(all_dms) = include_all_dms {
                guard.triage.include_all_dms = all_dms;
            }
            if let Some(tags) = filter_tags {
                guard.triage.filter_tags = tags;
            }
            guard.clone()
        };
        if let Ok(toml_str) = toml::to_string_pretty(&updated) {
            let _ = std::fs::write(&self.config_path, toml_str);
        }
        Ok(updated)
    }

    /// Seed any tags and JID associations declared in `triage.toml` into SQLite,
    /// and triage any existing chat history matching configured project JIDs.
    async fn sync_config_labels(self: &Arc<Self>) {
        let cfg = self.config();
        let mut color_idx = 1;
        for tag in &cfg.triage.filter_tags {
            let label_id = format!("tag:{}", tag.to_lowercase());
            let _ = self
                .store
                .upsert_label(&label_id, tag, color_idx % 10, "config")
                .await;
            color_idx += 1;
        }
        for proj in &cfg.projects {
            for tag in &proj.tags {
                let label_id = format!("tag:{}", tag.to_lowercase());
                let _ = self
                    .store
                    .upsert_label(&label_id, tag, color_idx % 10, "config")
                    .await;
                color_idx += 1;
                for jid in &proj.jids {
                    let norm_jid = if jid.contains('@') {
                        jid.clone()
                    } else {
                        let digits: String = jid.chars().filter(|c| c.is_ascii_digit()).collect();
                        format!("{digits}@s.whatsapp.net")
                    };
                    let _ = self.store.set_chat_label(&norm_jid, &label_id, true).await;
                }
            }
        }

        // If existing inbound messages match a configured project JID/phone and have no triage row yet,
        // automatically triage them so they appear in GPUI immediately.
        if let Ok(recent_msgs) = self.store.search_inbound(None, None, 200, None).await {
            let mut matched_chats = std::collections::HashSet::new();
            for m in recent_msgs {
                let is_configured_jid = cfg.projects.iter().any(|p| {
                    p.jids.iter().any(|j| {
                        jid_or_phone_matches(j, &m.chat_jid)
                            || jid_or_phone_matches(j, &m.sender_jid)
                    })
                });
                if is_configured_jid && matched_chats.insert(m.chat_jid.clone()) {
                    if let Ok(None) = self.store.get_chat_triage(&m.chat_jid).await {
                        // Also attach project tags to the actual chat_jid (e.g. if it's an @lid or 9-digit JID)
                        for proj in &cfg.projects {
                            if proj.jids.iter().any(|j| {
                                jid_or_phone_matches(j, &m.chat_jid)
                                    || jid_or_phone_matches(j, &m.sender_jid)
                            }) {
                                for tag in &proj.tags {
                                    let label_id = format!("tag:{}", tag.to_lowercase());
                                    let _ = self
                                        .store
                                        .set_chat_label(&m.chat_jid, &label_id, true)
                                        .await;
                                }
                            }
                        }
                        self.schedule_debounced_triage(m.chat_jid, 500);
                    }
                }
            }
        }
    }

    /// Handle an incoming WhatsApp message.
    async fn on_inbound_message(self: &Arc<Self>, msg: Arc<WhatsAppInbound>) {
        // Ignore our own outbound messages and non-message receipts/revokes
        if msg.is_from_me {
            return;
        }
        if matches!(
            msg.content,
            InboundContent::DeliveryReceipt { .. }
                | InboundContent::ReactionRemoved { .. }
                | InboundContent::Revoke { .. }
        ) {
            return;
        }

        if !msg.push_name.is_empty() {
            self.chat_names
                .insert(msg.jid.clone(), msg.push_name.clone());
        }

        let cfg = self.config();
        if !cfg.triage.enabled {
            return;
        }

        // Check if this chat passes the tag/DM/project filter
        if !self
            .should_triage_chat(&msg.jid, &msg.sender, msg.is_dm(), &cfg)
            .await
        {
            debug!(jid = %msg.jid, sender = %msg.sender, "skipping AI triage for chat (does not match tag/DM/JID filter)");
            return;
        }

        // If this is a voice/audio message and auto_transcribe_voice is enabled,
        // wait for Whisper transcription before scheduling the debounced summary.
        if cfg.triage.auto_transcribe_voice && matches!(msg.content, InboundContent::Audio { .. }) {
            let eng = self.clone();
            let msg_clone = msg.clone();
            let debounce_ms = cfg.triage.debounce_ms;
            tokio::spawn(async move {
                eng.transcribe_inbound_audio(&msg_clone).await;
                eng.schedule_debounced_triage(msg_clone.jid.clone(), debounce_ms);
            });
        } else {
            self.schedule_debounced_triage(msg.jid.clone(), cfg.triage.debounce_ms);
        }
    }

    async fn transcribe_inbound_audio(&self, msg: &WhatsAppInbound) {
        // Find the saved media_path from inbound_messages
        let rows = self
            .store
            .search_inbound(Some(&msg.jid), None, 10, None)
            .await
            .unwrap_or_default();
        let media_path = rows
            .into_iter()
            .find(|r| r.message_id == msg.id)
            .and_then(|r| r.media_path);

        if let Some(path_str) = media_path {
            let path = PathBuf::from(&path_str);
            if path.exists() {
                info!(msg_id = %msg.id, path = %path_str, "auto-transcribing voice note before AI triage");
                match self
                    .transcription_queue
                    .enqueue_and_wait(path, TranscriptionOptions::default(), Duration::from_secs(90))
                    .await
                {
                    Ok(res) => {
                        let new_body = format!("[voice transcript] {}", res.text.trim());
                        let _ = self.store.update_inbound_body(&msg.id, &new_body).await;
                    }
                    Err(e) => {
                        warn!(msg_id = %msg.id, error = %e, "voice note auto-transcription failed");
                    }
                }
            }
        }
    }

    async fn should_triage_chat(
        &self,
        chat_jid: &str,
        sender_jid: &str,
        is_dm: bool,
        cfg: &TriageFileConfig,
    ) -> bool {
        // 1. Explicit JID or phone number match in any project (matches chat_jid or resolved sender)
        if cfg.projects.iter().any(|p| {
            p.jids.iter().any(|j| {
                jid_or_phone_matches(j, chat_jid) || jid_or_phone_matches(j, sender_jid)
            })
        }) {
            return true;
        }
        // 2. Include all DMs toggle
        if cfg.triage.include_all_dms && is_dm {
            return true;
        }
        // 3. Empty filter_tags means triage all chats
        if cfg.triage.filter_tags.is_empty() {
            return true;
        }
        // 4. Check chat's assigned WhatsApp / local labels
        let labels = self.store.get_chat_labels(chat_jid).await.unwrap_or_default();
        labels.iter().any(|l| {
            cfg.triage.filter_tags.iter().any(|ft| {
                ft.eq_ignore_ascii_case(&l.name) || ft.eq_ignore_ascii_case(&l.label_id)
            })
        })
    }

    /// Schedule (or reset) a debounced triage run for `chat_jid`.
    pub fn schedule_debounced_triage(self: &Arc<Self>, chat_jid: String, debounce_ms: u64) {
        if let Some((_, old_handle)) = self.pending_debounces.remove(&chat_jid) {
            old_handle.abort();
        }
        let eng = self.clone();
        let jid_key = chat_jid.clone();
        let join = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(debounce_ms)).await;
            eng.pending_debounces.remove(&jid_key);
            if let Err(e) = eng.triage_chat_now(&jid_key).await {
                warn!(chat_jid = %jid_key, error = %e, "AI triage failed for chat");
            }
        });
        self.pending_debounces
            .insert(chat_jid, join.abort_handle());
    }

    /// Immediately execute AI summarization & project/agent routing for `chat_jid`.
    pub async fn triage_chat_now(self: &Arc<Self>, chat_jid: &str) -> Result<ChatTriageRow> {
        let messages = self
            .store
            .search_inbound(Some(chat_jid), None, 15, None)
            .await
            .unwrap_or_default();
        let labels = self.store.get_chat_labels(chat_jid).await.unwrap_or_default();
        let existing = self.store.get_chat_triage(chat_jid).await?;

        let last_msg_id = messages
            .first()
            .map(|m| m.message_id.clone())
            .or_else(|| existing.as_ref().map(|e| e.last_message_id.clone()))
            .unwrap_or_default();
        let chat_name = self
            .chat_names
            .get(chat_jid)
            .map(|s| s.clone())
            .or_else(|| existing.as_ref().map(|e| e.chat_name.clone()))
            .unwrap_or_else(|| chat_jid.to_string());
        let unread_count = existing.as_ref().map(|e| e.unread_count + 1).unwrap_or(1);

        // Emit 'triaging' state first so GPUI shows live activity
        let triaging_row = ChatTriageRow {
            chat_jid: chat_jid.to_string(),
            chat_name: chat_name.clone(),
            last_message_id: last_msg_id.clone(),
            unread_count,
            rolling_summary: existing
                .as_ref()
                .map(|e| e.rolling_summary.clone())
                .unwrap_or_else(|| "Summarizing incoming messages...".into()),
            project_id: existing.as_ref().and_then(|e| e.project_id.clone()),
            project_confidence: existing.as_ref().map(|e| e.project_confidence).unwrap_or(0.0),
            matched_agent_id: existing.as_ref().and_then(|e| e.matched_agent_id.clone()),
            agent_prompt: existing.as_ref().and_then(|e| e.agent_prompt.clone()),
            agent_status: "triaging".into(),
            agent_draft: existing.as_ref().and_then(|e| e.agent_draft.clone()),
            agent_error: None,
            updated_at: now_secs(),
            labels: labels.clone(),
        };
        let saved_triaging = self.store.upsert_chat_triage(&triaging_row).await?;
        let _ = self
            .event_tx
            .send(Arc::new(BridgeEvent::TriageUpdate(Box::new(saved_triaging))));

        let cfg = self.config();
        let llm_out = run_llm_or_heuristic_triage(chat_jid, &chat_name, &labels, &messages, &cfg).await;

        let next_status = if llm_out.quick_reply_draft.is_some() {
            "draft_ready"
        } else {
            "idle"
        };

        let final_row = ChatTriageRow {
            chat_jid: chat_jid.to_string(),
            chat_name,
            last_message_id: last_msg_id,
            unread_count,
            rolling_summary: llm_out.rolling_summary,
            project_id: llm_out.project_id.clone(),
            project_confidence: llm_out.project_confidence,
            matched_agent_id: llm_out.matched_agent_id.clone(),
            agent_prompt: llm_out.suggested_agent_prompt.clone(),
            agent_status: next_status.into(),
            agent_draft: llm_out.quick_reply_draft,
            agent_error: None,
            updated_at: now_secs(),
            labels,
        };

        let hydrated = self.store.upsert_chat_triage(&final_row).await?;
        let _ = self.event_tx.send(Arc::new(BridgeEvent::TriageUpdate(Box::new(
            hydrated.clone(),
        ))));

        // Check if matched CLI agent is configured with auto_run = true
        if let Some(ref agent_id) = hydrated.matched_agent_id {
            if let Some(agent_cfg) = cfg.agents.iter().find(|a| &a.id == agent_id) {
                if agent_cfg.auto_run {
                    let eng = self.clone();
                    let jid = chat_jid.to_string();
                    let aid = agent_id.clone();
                    let prompt = hydrated.agent_prompt.clone();
                    tokio::spawn(async move {
                        let _ = eng.run_cli_agent(&jid, Some(&aid), prompt.as_deref()).await;
                    });
                }
            }
        }

        Ok(hydrated)
    }

    /// Spawn the CLI subprocess agent in the matched project's `workdir` and capture its output as the draft.
    pub async fn run_cli_agent(
        &self,
        chat_jid: &str,
        override_agent_id: Option<&str>,
        override_prompt: Option<&str>,
    ) -> Result<ChatTriageRow> {
        let current = self
            .store
            .get_chat_triage(chat_jid)
            .await?
            .with_context(|| format!("chat '{chat_jid}' has no triage record yet"))?;

        let cfg = self.config();
        let agent_id = override_agent_id
            .map(|s| s.to_string())
            .or_else(|| current.matched_agent_id.clone())
            .or_else(|| cfg.agents.first().map(|a| a.id.clone()))
            .context("no CLI agent matched or configured in triage.toml")?;

        let agent = cfg
            .agents
            .iter()
            .find(|a| a.id == agent_id)
            .cloned()
            .with_context(|| format!("agent '{agent_id}' not found in triage.toml"))?;

        let project = current
            .project_id
            .as_ref()
            .and_then(|pid| cfg.projects.iter().find(|p| &p.id == pid))
            .or_else(|| cfg.projects.first())
            .cloned();

        let workdir = project
            .as_ref()
            .map(|p| PathBuf::from(&p.workdir))
            .filter(|p| p.exists())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let project_id = project
            .as_ref()
            .map(|p| p.id.clone())
            .unwrap_or_else(|| "default".into());

        let prompt = override_prompt
            .map(|s| s.to_string())
            .or_else(|| current.agent_prompt.clone())
            .unwrap_or_else(|| {
                format!(
                    "Draft a concise, helpful WhatsApp reply for project '{}' based on this summary: {}",
                    project_id, current.rolling_summary
                )
            });

        // Update state to running_agent and broadcast
        let mut running_row = current.clone();
        running_row.matched_agent_id = Some(agent.id.clone());
        running_row.agent_prompt = Some(prompt.clone());
        running_row.agent_status = "running_agent".into();
        running_row.agent_error = None;
        running_row.updated_at = now_secs();
        let saved_running = self.store.upsert_chat_triage(&running_row).await?;
        let _ = self
            .event_tx
            .send(Arc::new(BridgeEvent::TriageUpdate(Box::new(saved_running))));

        // Substitute template variables in args
        let rendered_args: Vec<String> = agent
            .args
            .iter()
            .map(|arg| {
                substitute_template(
                    arg,
                    &prompt,
                    chat_jid,
                    &current.chat_name,
                    &current.rolling_summary,
                    &project_id,
                    &workdir.to_string_lossy(),
                )
            })
            .collect();

        info!(
            chat_jid = %chat_jid,
            agent = %agent.id,
            command = %agent.command,
            workdir = %workdir.display(),
            "spawning CLI subprocess agent"
        );

        let mut cmd = tokio::process::Command::new(&agent.command);
        cmd.args(&rendered_args)
            .current_dir(&workdir)
            .env("WHATSRUST_CHAT_JID", chat_jid)
            .env("WHATSRUST_CHAT_NAME", &current.chat_name)
            .env("WHATSRUST_PROJECT_ID", &project_id)
            .env("WHATSRUST_SUMMARY", &current.rolling_summary)
            .env("WHATSRUST_PROMPT", &prompt)
            .kill_on_drop(true);

        let exec_res = tokio::time::timeout(
            Duration::from_secs(agent.timeout_secs.max(5)),
            cmd.output(),
        )
        .await;

        let updated = match exec_res {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if output.status.success() && !stdout.is_empty() {
                    self.store
                        .update_agent_status(chat_jid, "draft_ready", Some(&stdout), None)
                        .await?
                } else if output.status.success() {
                    let fallback = if stderr.is_empty() {
                        "(Agent finished with empty stdout)".to_string()
                    } else {
                        stderr
                    };
                    self.store
                        .update_agent_status(chat_jid, "draft_ready", Some(&fallback), None)
                        .await?
                } else {
                    let err_msg = if stderr.is_empty() {
                        format!("Agent exited with status {}", output.status)
                    } else {
                        stderr
                    };
                    self.store
                        .update_agent_status(chat_jid, "failed", None, Some(&err_msg))
                        .await?
                }
            }
            Ok(Err(e)) => {
                let err_msg = format!("Failed to spawn '{}': {e}", agent.command);
                self.store
                    .update_agent_status(chat_jid, "failed", None, Some(&err_msg))
                    .await?
            }
            Err(_) => {
                let err_msg = format!(
                    "CLI agent '{}' timed out after {}s",
                    agent.id, agent.timeout_secs
                );
                self.store
                    .update_agent_status(chat_jid, "failed", None, Some(&err_msg))
                    .await?
            }
        };

        let final_row = updated.context("triage row disappeared after CLI agent execution")?;
        let _ = self.event_tx.send(Arc::new(BridgeEvent::TriageUpdate(Box::new(
            final_row.clone(),
        ))));
        Ok(final_row)
    }

    /// Assign or remove a local/WhatsApp tag on a chat and broadcast `LabelUpdate`.
    pub async fn assign_chat_tag(
        &self,
        chat_jid: &str,
        tag_name: &str,
        color: Option<i32>,
        labeled: bool,
    ) -> Result<Vec<LabelRow>> {
        let label_id = if tag_name.starts_with("tag:") || tag_name.chars().all(|c| c.is_ascii_digit()) {
            tag_name.to_string()
        } else {
            format!("tag:{}", tag_name.to_lowercase())
        };
        let label_row = if labeled {
            Some(
                self.store
                    .upsert_label(&label_id, tag_name, color.unwrap_or(1), "local")
                    .await?,
            )
        } else {
            None
        };
        self.store
            .set_chat_label(chat_jid, &label_id, labeled)
            .await?;

        let _ = self
            .event_tx
            .send(Arc::new(BridgeEvent::LabelUpdate(LabelUpdateEvent {
                label_id,
                chat_jid: Some(chat_jid.to_string()),
                labeled: Some(labeled),
                label: label_row,
            })));

        // If chat already has a triage entry, broadcast its refreshed labels
        if let Ok(Some(triage_row)) = self.store.get_chat_triage(chat_jid).await {
            let _ = self
                .event_tx
                .send(Arc::new(BridgeEvent::TriageUpdate(Box::new(triage_row))));
        }

        Ok(self.store.get_chat_labels(chat_jid).await?)
    }
}

pub fn substitute_template(
    template: &str,
    prompt: &str,
    chat_jid: &str,
    chat_name: &str,
    summary: &str,
    project_id: &str,
    workdir: &str,
) -> String {
    template
        .replace("{{prompt}}", prompt)
        .replace("{{chat_jid}}", chat_jid)
        .replace("{{chat_name}}", chat_name)
        .replace("{{summary}}", summary)
        .replace("{{project_id}}", project_id)
        .replace("{{workdir}}", workdir)
}

async fn run_llm_or_heuristic_triage(
    chat_jid: &str,
    chat_name: &str,
    labels: &[LabelRow],
    messages: &[InboundRow],
    cfg: &TriageFileConfig,
) -> LlmTriageOutput {
    let api_key = read_env_or_dotenv(&cfg.llm.api_key_env)
        .or_else(|| read_env_or_dotenv("GEMINI_API_KEY"))
        .or_else(|| read_env_or_dotenv("GOOGLE_API_KEY"))
        .or_else(|| read_env_or_dotenv("OPENAI_API_KEY"))
        .or_else(|| read_env_or_dotenv("TR_LLM_API_KEY"));

    // Local Ollama / vLLM endpoints often don't require an API key
    let is_local_llm =
        cfg.llm.base_url.contains("localhost") || cfg.llm.base_url.contains("127.0.0.1");

    if api_key.is_some() || is_local_llm {
        let key = api_key.unwrap_or_else(|| "ollama".into());
        let base_url = cfg.llm.base_url.clone();
        let model = cfg.llm.model.clone();
        let jid = chat_jid.to_string();
        let cname = chat_name.to_string();
        let lbls = labels.to_vec();
        let msgs = messages.to_vec();
        let cfg_clone = cfg.clone();

        let res = tokio::task::spawn_blocking(move || {
            call_openai_compatible_triage(
                &base_url, &key, &model, &jid, &cname, &lbls, &msgs, &cfg_clone,
            )
        })
        .await;

        match res {
            Ok(Ok(out)) => return out,
            Ok(Err(e)) => {
                warn!(error = %e, "LLM triage request failed, falling back to heuristic triage")
            }
            Err(e) => {
                warn!(error = %e, "LLM triage task panicked, falling back to heuristic triage")
            }
        }
    }

    heuristic_triage(chat_jid, chat_name, labels, messages, cfg)
}

fn call_openai_compatible_triage(
    base_url: &str,
    api_key: &str,
    model: &str,
    chat_jid: &str,
    chat_name: &str,
    labels: &[LabelRow],
    messages: &[InboundRow],
    cfg: &TriageFileConfig,
) -> Result<LlmTriageOutput> {
    let trimmed_base = base_url.trim_end_matches('/');
    // If the user configured `https://generativelanguage.googleapis.com/v1beta`, automatically
    // route to `/v1beta/openai/chat/completions` so the OpenAI-compatible payload works seamlessly.
    let url = if trimmed_base.ends_with("generativelanguage.googleapis.com/v1beta") {
        format!("{trimmed_base}/openai/chat/completions")
    } else {
        format!("{trimmed_base}/chat/completions")
    };

    let projects_desc: Vec<serde_json::Value> = cfg
        .projects
        .iter()
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "tags": p.tags,
                "jids": p.jids,
                "keywords": p.keywords,
                "default_agent": p.default_agent,
            })
        })
        .collect();

    let agents_desc: Vec<serde_json::Value> = cfg
        .agents
        .iter()
        .map(|a| {
            serde_json::json!({
                "id": a.id,
                "name": a.name,
                "description": a.description,
            })
        })
        .collect();

    // Format messages chronologically (oldest to newest)
    let mut chronological = messages.to_vec();
    chronological.reverse();
    let transcript = chronological
        .iter()
        .map(|m| {
            let body = m.body_text.as_deref().unwrap_or("[media]");
            format!("[{}] {}: {}", m.content_kind, m.sender_jid, body)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let label_names: Vec<&str> = labels.iter().map(|l| l.name.as_str()).collect();

    let system_prompt = "You are an AI executive triage assistant for incoming WhatsApp messages.
Analyze the chat labels and recent message transcript, then output strictly valid JSON with these keys:
- \"rolling_summary\": concise 1-2 sentence summary (resume) of what the sender is asking or reporting.
- \"project_id\": string ID of the best matching project from the provided projects list, or null.
- \"project_confidence\": float between 0.0 and 1.0.
- \"matched_agent_id\": string ID of the best CLI AI agent from the provided agents list that can answer or investigate this, or null.
- \"suggested_agent_prompt\": specific instruction to pass to the CLI agent in the project's directory to investigate or draft the answer, or null.
- \"quick_reply_draft\": a polite, ready-to-send WhatsApp reply draft based on the context, or null.";

    let user_prompt = serde_json::json!({
        "chat_jid": chat_jid,
        "chat_name": chat_name,
        "whatsapp_tags": label_names,
        "available_projects": projects_desc,
        "available_agents": agents_desc,
        "recent_messages": transcript,
    });

    let body = serde_json::json!({
        "model": model,
        "temperature": 0.2,
        "response_format": { "type": "json_object" },
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": serde_json::to_string_pretty(&user_prompt)? }
        ]
    });

    let mut resp = ureq::post(&url)
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .send_json(&body)
        .context("HTTP POST to LLM endpoint failed")?;

    let resp_json: serde_json::Value = resp
        .body_mut()
        .read_json()
        .context("failed to parse LLM JSON response")?;

    let content_str = resp_json["choices"][0]["message"]["content"]
        .as_str()
        .context("LLM response missing choices[0].message.content")?;

    let clean_json = strip_markdown_json_fence(content_str);
    let parsed: LlmTriageOutput =
        serde_json::from_str(clean_json).context("failed to deserialize LlmTriageOutput JSON")?;
    Ok(parsed)
}

/// Strip optional ` ```json ... ``` ` markdown fences returned by some LLMs.
pub fn strip_markdown_json_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest).trim_start();
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim();
        }
    }
    trimmed
}

/// Read an environment variable from `std::env::var`, or fall back to parsing `.env` in the workspace.
pub fn read_env_or_dotenv(var_name: &str) -> Option<String> {
    if let Ok(val) = std::env::var(var_name) {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    let env_content = std::fs::read_to_string(".env").ok()?;
    for line in env_content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == var_name {
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Canonicalize phone digits so Brazilian 12-digit (`55` + DDD + 8 digits) and 13-digit
/// (`55` + DDD + `9` + 8 digits) mobile numbers match each other.
pub fn canonicalize_phone_digits(input: &str) -> String {
    let user_part = input.split('@').next().unwrap_or(input);
    let user_part = user_part.split(':').next().unwrap_or(user_part);
    let digits: String = user_part.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() == 13 && digits.starts_with("55") && digits.as_bytes().get(4) == Some(&b'9') {
        format!("{}{}", &digits[..4], &digits[5..])
    } else {
        digits
    }
}

/// Match a configured JID or phone number pattern against a chat JID or sender JID.
pub fn jid_or_phone_matches(pattern: &str, candidate: &str) -> bool {
    if pattern.is_empty() || candidate.is_empty() {
        return false;
    }
    if pattern.eq_ignore_ascii_case(candidate) {
        return true;
    }
    let p_digits = canonicalize_phone_digits(pattern);
    let c_digits = canonicalize_phone_digits(candidate);
    !p_digits.is_empty() && p_digits.len() >= 8 && p_digits == c_digits
}

/// Deterministic fallback when no LLM API key is configured or when offline.
pub fn heuristic_triage(
    chat_jid: &str,
    chat_name: &str,
    labels: &[LabelRow],
    messages: &[InboundRow],
    cfg: &TriageFileConfig,
) -> LlmTriageOutput {
    let mut chronological = messages.to_vec();
    chronological.reverse();

    let combined_text = chronological
        .iter()
        .filter_map(|m| m.body_text.as_deref())
        .collect::<Vec<_>>()
        .join(" | ");

    let rolling_summary = if combined_text.is_empty() {
        format!("New activity in chat {chat_name} ({chat_jid}).")
    } else {
        let preview: String = combined_text.chars().take(220).collect();
        format!(
            "{} ({} msg{}): {}",
            chat_name,
            chronological.len(),
            if chronological.len() == 1 { "" } else { "s" },
            preview
        )
    };

    let lower_text = combined_text.to_lowercase();

    // Match project by: 1) JID/Phone, 2) Chat Tag/Label, 3) Keyword/Name in message text
    let mut matched_project: Option<(&ProjectConfig, f64)> = None;
    for proj in &cfg.projects {
        if proj.jids.iter().any(|j| {
            jid_or_phone_matches(j, chat_jid)
                || messages
                    .iter()
                    .any(|m| jid_or_phone_matches(j, &m.sender_jid))
        }) {
            matched_project = Some((proj, 0.98));
            break;
        }
        if labels.iter().any(|l| {
            proj.tags
                .iter()
                .any(|pt| pt.eq_ignore_ascii_case(&l.name) || pt.eq_ignore_ascii_case(&l.label_id))
        }) {
            matched_project = Some((proj, 0.85));
            break;
        }
        if proj
            .keywords
            .iter()
            .any(|kw| !kw.is_empty() && lower_text.contains(&kw.to_lowercase()))
            || lower_text.contains(&proj.id.to_lowercase())
            || lower_text.contains(&proj.name.to_lowercase())
        {
            matched_project = Some((proj, 0.78));
            break;
        }
    }

    if matched_project.is_none() {
        if let Some(first_proj) = cfg.projects.first() {
            matched_project = Some((first_proj, 0.45));
        }
    }

    let project_id = matched_project.map(|(p, _)| p.id.clone());
    let project_confidence = matched_project.map(|(_, c)| c).unwrap_or(0.0);
    let matched_agent_id = matched_project
        .and_then(|(p, _)| p.default_agent.clone())
        .or_else(|| cfg.agents.first().map(|a| a.id.clone()));

    let suggested_agent_prompt = matched_project.map(|(p, _)| {
        format!(
            "Sender '{}' in WhatsApp chat '{}' sent: \"{}\". Inspect project '{}' and draft a concise WhatsApp reply.",
            chat_name, chat_jid, combined_text, p.name
        )
    });

    let quick_reply_draft = Some(format!(
        "Received your message regarding {} — checking on it now.",
        matched_project
            .map(|(p, _)| p.name.as_str())
            .unwrap_or("your request")
    ));

    LlmTriageOutput {
        rolling_summary,
        project_id,
        project_confidence,
        matched_agent_id,
        suggested_agent_prompt,
        quick_reply_draft,
    }
}

fn load_or_create_config(path: &Path) -> TriageFileConfig {
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(s) => match toml::from_str::<TriageFileConfig>(&s) {
                Ok(cfg) => return cfg,
                Err(e) => warn!(path = %path.display(), error = %e, "failed to parse triage.toml, using defaults"),
            },
            Err(e) => warn!(path = %path.display(), error = %e, "failed to read triage.toml, using defaults"),
        }
    } else {
        let def = TriageFileConfig::default();
        if let Ok(toml_str) = toml::to_string_pretty(&def) {
            if let Err(e) = std::fs::write(path, toml_str) {
                debug!(path = %path.display(), error = %e, "could not write default triage.toml");
            } else {
                info!(path = %path.display(), "created default triage.toml configuration");
            }
        }
        return def;
    }
    TriageFileConfig::default()
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_template() {
        let out = substitute_template(
            "Ask {{project_id}} about {{summary}} from {{chat_name}} ({{chat_jid}}) in {{workdir}}: {{prompt}}",
            "fix bug",
            "123@s.whatsapp.net",
            "Alice",
            "login error",
            "whatsrust",
            "/tmp/proj",
        );
        assert_eq!(
            out,
            "Ask whatsrust about login error from Alice (123@s.whatsapp.net) in /tmp/proj: fix bug"
        );
    }

    #[test]
    fn test_heuristic_triage_matches_tag_and_keyword() {
        let cfg = TriageFileConfig::default();
        let labels = vec![LabelRow {
            label_id: "tag:work".into(),
            name: "Work".into(),
            color: 1,
            source: "whatsapp".into(),
            updated_at: 100,
        }];
        let messages = vec![InboundRow {
            id: 1,
            chat_jid: "5511999999999@s.whatsapp.net".into(),
            sender_jid: "5511999999999@s.whatsapp.net".into(),
            message_id: "MSG1".into(),
            content_kind: "text".into(),
            body_text: Some("Can we check the gpui bridge status?".into()),
            timestamp: 1000,
            media_path: None,
        }];

        let out = heuristic_triage(
            "5511999999999@s.whatsapp.net",
            "Ivens",
            &labels,
            &messages,
            &cfg,
        );
        assert_eq!(out.project_id.as_deref(), Some("whatsrust"));
        assert!(out.project_confidence >= 0.8);
        assert_eq!(out.matched_agent_id.as_deref(), Some("claude-cli"));
        assert!(out.rolling_summary.contains("gpui bridge status"));
    }

    #[test]
    fn test_brazil_phone_and_jid_matching() {
        assert!(jid_or_phone_matches(
            "557791229001",
            "557791229001@s.whatsapp.net"
        ));
        assert!(jid_or_phone_matches(
            "557791229001@s.whatsapp.net",
            "5577991229001@s.whatsapp.net"
        ));
        assert!(jid_or_phone_matches("5577991229001", "557791229001"));
        assert!(!jid_or_phone_matches(
            "557791229001",
            "5511999999999@s.whatsapp.net"
        ));
    }

    #[tokio::test]
    async fn test_triage_engine_and_cli_agent_subprocess() {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("whatsrust-triage-engine-{ts}"));
        std::fs::create_dir_all(&root).unwrap();
        let db_path = root.join("test.db");

        let store = Store::new(&db_path).unwrap();
        let (event_tx, _event_rx) = crate::bridge_events::new_event_bus();
        let tq = Arc::new(TranscriptionQueue::new());
        let engine = TriageEngine::start(store.clone(), event_tx, tq);

        // Assign tag + insert inbound message
        engine
            .assign_chat_tag("5511999999999@s.whatsapp.net", "Work", Some(1), true)
            .await
            .unwrap();
        store
            .insert_inbound(
                "5511999999999@s.whatsapp.net",
                "5511999999999@s.whatsapp.net",
                "M1",
                "text",
                Some("Please check the whatsrust gpui implementation"),
                1000,
                None,
            )
            .await
            .unwrap();

        let triaged = engine
            .triage_chat_now("5511999999999@s.whatsapp.net")
            .await
            .unwrap();
        assert_eq!(triaged.project_id.as_deref(), Some("whatsrust"));
        assert_eq!(triaged.labels.len(), 1);

        // Run the 'quick-echo' CLI subprocess agent in the project's workdir
        let after_agent = engine
            .run_cli_agent(
                "5511999999999@s.whatsapp.net",
                Some("quick-echo"),
                Some("Test prompt"),
            )
            .await
            .unwrap();
        assert_eq!(after_agent.agent_status, "draft_ready");
        assert!(after_agent
            .agent_draft
            .as_deref()
            .unwrap_or("")
            .contains("[Draft for whatsrust]"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
