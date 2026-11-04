//! Audio transcription with Whisper using a strictly sequential, single-worker queue.
//!
//! Features:
//! - Strictly 1 transcription running at a time (FIFO queue)
//! - Backends:
//!   1. OpenAI-compatible Whisper API (`WHISPER_API_KEY` / `OPENAI_API_KEY`, optional `WHISPER_API_URL`, `WHISPER_MODEL`)
//!   2. Local Whisper CLI / binary (`WHISPER_CMD` or `whisper-cli` / `whisper` in PATH)
//! - Optional audio format conversion via `ffmpeg` if needed
//! - Synchronous or asynchronous job execution and status tracking

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

/// User-configurable options for transcription.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptionOptions {
    /// Target language code (e.g. "en", "pt", "es"). Auto-detected if omitted.
    pub language: Option<String>,
    /// Optional guiding prompt for vocabulary/formatting.
    pub prompt: Option<String>,
    /// Model name (defaults to "whisper-1" for API, or model path for local CLI).
    pub model: Option<String>,
    /// Sampling temperature (0.0 to 1.0).
    pub temperature: Option<f32>,
}

/// Transcription result payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub job_id: String,
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
    pub source_path: String,
    pub completed_at: i64,
}

/// Current status of a transcription job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum JobStatus {
    #[serde(rename = "queued")]
    Queued { queue_position: usize, enqueued_at: i64 },
    #[serde(rename = "processing")]
    Processing { started_at: i64 },
    #[serde(rename = "completed")]
    Completed { result: TranscriptionResult },
    #[serde(rename = "failed")]
    Failed { error: String, failed_at: i64 },
}

/// Overview of queue health and current activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatusOverview {
    pub queue_depth: usize,
    pub is_processing: bool,
    pub active_job_id: Option<String>,
    pub recent_jobs_count: usize,
}

struct InternalJob {
    job_id: String,
    audio_path: PathBuf,
    options: TranscriptionOptions,
    #[allow(dead_code)]
    enqueued_at: i64,
    response_tx: Option<oneshot::Sender<std::result::Result<TranscriptionResult, String>>>,
}

struct QueueState {
    jobs: HashMap<String, JobStatus>,
    job_order: VecDeque<String>,
    active_job: Option<String>,
}

const MAX_JOB_HISTORY: usize = 100;

/// Sequential transcription queue.
#[derive(Clone)]
pub struct TranscriptionQueue {
    tx: mpsc::Sender<InternalJob>,
    state: Arc<RwLock<QueueState>>,
    queue_counter: Arc<AtomicUsize>,
}

impl TranscriptionQueue {
    /// Create and start a new transcription queue with a single sequential worker.
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel::<InternalJob>(200);
        let state = Arc::new(RwLock::new(QueueState {
            jobs: HashMap::new(),
            job_order: VecDeque::new(),
            active_job: None,
        }));
        let queue_counter = Arc::new(AtomicUsize::new(0));

        let worker_state = state.clone();
        let worker_counter = queue_counter.clone();

        tokio::spawn(async move {
            info!("transcription queue worker started (concurrency: 1)");

            while let Some(job) = rx.recv().await {
                worker_counter.fetch_sub(1, Ordering::Relaxed);

                // Update status to Processing
                {
                    let mut s = worker_state.write();
                    s.active_job = Some(job.job_id.clone());
                    s.jobs.insert(
                        job.job_id.clone(),
                        JobStatus::Processing {
                            started_at: chrono::Utc::now().timestamp(),
                        },
                    );
                }

                info!(job_id = %job.job_id, path = %job.audio_path.display(), "processing transcription job");

                // Execute transcription strictly one at a time
                let res = execute_whisper_transcription(&job.audio_path, &job.options).await;

                // Update status to Completed or Failed
                let now = chrono::Utc::now().timestamp();
                match res {
                    Ok(text) => {
                        let result = TranscriptionResult {
                            job_id: job.job_id.clone(),
                            text,
                            language: job.options.language.clone(),
                            duration_secs: None,
                            source_path: job.audio_path.to_string_lossy().to_string(),
                            completed_at: now,
                        };

                        info!(job_id = %job.job_id, "transcription job completed successfully");

                        let mut s = worker_state.write();
                        s.jobs.insert(job.job_id.clone(), JobStatus::Completed { result: result.clone() });
                        s.active_job = None;

                        if let Some(reply) = job.response_tx {
                            let _ = reply.send(Ok(result));
                        }
                    }
                    Err(e) => {
                        let err_msg = e.to_string();
                        warn!(job_id = %job.job_id, error = %err_msg, "transcription job failed");

                        let mut s = worker_state.write();
                        s.jobs.insert(job.job_id.clone(), JobStatus::Failed {
                            error: err_msg.clone(),
                            failed_at: now,
                        });
                        s.active_job = None;

                        if let Some(reply) = job.response_tx {
                            let _ = reply.send(Err(err_msg));
                        }
                    }
                }
            }

            info!("transcription queue worker stopped");
        });

        Self {
            tx,
            state,
            queue_counter,
        }
    }

    /// Enqueue a transcription job asynchronously. Returns the job ID.
    pub async fn enqueue(
        &self,
        audio_path: PathBuf,
        options: TranscriptionOptions,
    ) -> Result<String> {
        let job_id = generate_job_id();
        let now = chrono::Utc::now().timestamp();
        let position = self.queue_counter.fetch_add(1, Ordering::Relaxed) + 1;

        {
            let mut s = self.state.write();
            if s.job_order.len() >= MAX_JOB_HISTORY {
                if let Some(old_id) = s.job_order.pop_front() {
                    s.jobs.remove(&old_id);
                }
            }
            s.job_order.push_back(job_id.clone());
            s.jobs.insert(
                job_id.clone(),
                JobStatus::Queued {
                    queue_position: position,
                    enqueued_at: now,
                },
            );
        }

        let job = InternalJob {
            job_id: job_id.clone(),
            audio_path,
            options,
            enqueued_at: now,
            response_tx: None,
        };

        self.tx.send(job).await
            .map_err(|_| anyhow::anyhow!("transcription queue channel closed"))?;

        Ok(job_id)
    }

    /// Enqueue a transcription job and wait for its completion up to `timeout`.
    pub async fn enqueue_and_wait(
        &self,
        audio_path: PathBuf,
        options: TranscriptionOptions,
        timeout: Duration,
    ) -> Result<TranscriptionResult> {
        let job_id = generate_job_id();
        let now = chrono::Utc::now().timestamp();
        let position = self.queue_counter.fetch_add(1, Ordering::Relaxed) + 1;

        let (reply_tx, reply_rx) = oneshot::channel();

        {
            let mut s = self.state.write();
            if s.job_order.len() >= MAX_JOB_HISTORY {
                if let Some(old_id) = s.job_order.pop_front() {
                    s.jobs.remove(&old_id);
                }
            }
            s.job_order.push_back(job_id.clone());
            s.jobs.insert(
                job_id.clone(),
                JobStatus::Queued {
                    queue_position: position,
                    enqueued_at: now,
                },
            );
        }

        let job = InternalJob {
            job_id: job_id.clone(),
            audio_path,
            options,
            enqueued_at: now,
            response_tx: Some(reply_tx),
        };

        self.tx.send(job).await
            .map_err(|_| anyhow::anyhow!("transcription queue channel closed"))?;

        match tokio::time::timeout(timeout, reply_rx).await {
            Ok(Ok(Ok(res))) => Ok(res),
            Ok(Ok(Err(err))) => anyhow::bail!("transcription failed: {err}"),
            Ok(Err(_)) => anyhow::bail!("transcription worker canceled request"),
            Err(_) => anyhow::bail!("transcription timed out after {}s (job ID: {job_id})", timeout.as_secs()),
        }
    }

    /// Look up the status of a specific job by ID.
    pub fn get_job_status(&self, job_id: &str) -> Option<JobStatus> {
        self.state.read().jobs.get(job_id).cloned()
    }

    /// Get an overview of queue activity and metrics.
    pub fn overview(&self) -> QueueStatusOverview {
        let s = self.state.read();
        QueueStatusOverview {
            queue_depth: self.queue_counter.load(Ordering::Relaxed),
            is_processing: s.active_job.is_some(),
            active_job_id: s.active_job.clone(),
            recent_jobs_count: s.jobs.len(),
        }
    }

    /// List recent jobs.
    pub fn recent_jobs(&self) -> Vec<(String, JobStatus)> {
        let s = self.state.read();
        s.job_order
            .iter()
            .filter_map(|id| s.jobs.get(id).map(|status| (id.clone(), status.clone())))
            .collect()
    }
}

impl Default for TranscriptionQueue {
    fn default() -> Self {
        Self::new()
    }
}

fn generate_job_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let ts = chrono::Utc::now().timestamp();
    format!("tx_{ts}_{hex}")
}

/// Execute whisper transcription using either API or local binary.
async fn execute_whisper_transcription(
    audio_path: &Path,
    options: &TranscriptionOptions,
) -> Result<String> {
    if !audio_path.exists() {
        anyhow::bail!("audio file not found: {}", audio_path.display());
    }

    // 1. Try OpenAI-compatible API if API key is provided via env
    let api_key = std::env::var("WHISPER_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .ok();

    if let Some(key) = api_key {
        let api_url = std::env::var("WHISPER_API_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/audio/transcriptions".to_string());
        let model = options.model.clone()
            .or_else(|| std::env::var("WHISPER_MODEL").ok())
            .unwrap_or_else(|| "whisper-1".to_string());

        let path_clone = audio_path.to_path_buf();
        let opts = options.clone();

        return tokio::task::spawn_blocking(move || {
            call_whisper_api(&api_url, &key, &model, &path_clone, &opts)
        })
        .await
        .map_err(|e| anyhow::anyhow!("API task panicked: {e}"))?;
    }

    // 2. Try local Whisper CLI if configured or present
    if let Some(whisper_cmd) = find_local_whisper_command() {
        return call_local_whisper(&whisper_cmd, audio_path, options).await;
    }

    // 3. Neither configured - provide clear setup instructions
    anyhow::bail!(
        "Whisper is not configured. Please set WHISPER_API_KEY or OPENAI_API_KEY (optional: WHISPER_API_URL, WHISPER_MODEL), \
         or install a local whisper CLI and set WHISPER_CMD (e.g. WHISPER_CMD=whisper-cli)."
    )
}

/// Find a local whisper executable if available.
fn find_local_whisper_command() -> Option<String> {
    if let Ok(cmd) = std::env::var("WHISPER_CMD") {
        if !cmd.trim().is_empty() {
            return Some(cmd);
        }
    }

    let candidates = ["whisper-cli", "whisper", "whisper-cpp", "main"];
    for name in candidates {
        #[cfg(windows)]
        let binary_name = format!("{name}.exe");
        #[cfg(not(windows))]
        let binary_name = name.to_string();

        if let Ok(path) = which(&binary_name) {
            return Some(path.to_string_lossy().to_string());
        }
    }
    None
}

/// Simple PATH search for an executable name.
fn which(name: &str) -> Result<PathBuf> {
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let full = dir.join(name);
            if full.is_file() {
                return Ok(full);
            }
        }
    }
    anyhow::bail!("executable '{name}' not found in PATH")
}

/// Call OpenAI-compatible Whisper API.
fn call_whisper_api(
    url: &str,
    api_key: &str,
    model: &str,
    audio_path: &Path,
    options: &TranscriptionOptions,
) -> Result<String> {
    let file_bytes = std::fs::read(audio_path)
        .with_context(|| format!("failed to read audio file {}", audio_path.display()))?;

    let filename = audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.ogg");

    let mime = crate::download::extension_for_mime(
        audio_path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ).map(|ext| format!("audio/{ext}")).unwrap_or_else(|| "audio/ogg".to_string());

    let boundary = format!("----WhatsrustBoundary{:x}", rand::random::<u64>());
    let mut body = Vec::new();

    // Field: model
    add_multipart_field(&mut body, &boundary, "model", model);

    // Field: language
    if let Some(ref lang) = options.language {
        add_multipart_field(&mut body, &boundary, "language", lang);
    }

    // Field: prompt
    if let Some(ref prompt) = options.prompt {
        add_multipart_field(&mut body, &boundary, "prompt", prompt);
    }

    // Field: temperature
    if let Some(temp) = options.temperature {
        add_multipart_field(&mut body, &boundary, "temperature", &temp.to_string());
    }

    // Field: file
    add_multipart_file(&mut body, &boundary, "file", filename, &mime, &file_bytes);

    // End multipart
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let mut response = ureq::post(url)
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body)
        .map_err(|e| anyhow::anyhow!("Whisper API request failed to {url}: {e}"))?;

    use std::io::Read;
    let mut resp_str = String::new();
    response.body_mut().as_reader().read_to_string(&mut resp_str)
        .context("failed reading Whisper API response body")?;

    #[derive(Deserialize)]
    struct WhisperResponse {
        text: Option<String>,
        error: Option<serde_json::Value>,
    }

    let parsed: WhisperResponse = serde_json::from_str(&resp_str)
        .with_context(|| format!("failed parsing Whisper API response: {resp_str}"))?;

    if let Some(err) = parsed.error {
        anyhow::bail!("Whisper API returned error: {err}");
    }

    parsed.text.context("Whisper API response missing 'text' field")
}

fn add_multipart_field(body: &mut Vec<u8>, boundary: &str, name: &str, value: &str) {
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
    );
}

fn add_multipart_file(
    body: &mut Vec<u8>,
    boundary: &str,
    name: &str,
    filename: &str,
    mime: &str,
    data: &[u8],
) {
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
        ).as_bytes(),
    );
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");
}

/// Call local whisper CLI binary.
async fn call_local_whisper(
    whisper_cmd: &str,
    audio_path: &Path,
    options: &TranscriptionOptions,
) -> Result<String> {
    // If input is OGG/Opus and ffmpeg is installed, convert to 16kHz WAV for whisper.cpp compatibility
    let wav_path = maybe_convert_to_wav(audio_path).await?;
    let target_audio = wav_path.as_deref().unwrap_or(audio_path);

    let mut cmd = tokio::process::Command::new(whisper_cmd);
    cmd.arg(target_audio);

    if let Some(ref model) = options.model {
        cmd.arg("-m").arg(model);
    }
    if let Some(ref lang) = options.language {
        cmd.arg("-l").arg(lang);
    }

    let output = cmd.output().await
        .with_context(|| format!("failed to execute whisper command '{whisper_cmd}'"))?;

    // Clean up temporary WAV file if created
    if let Some(ref temp) = wav_path {
        let _ = tokio::fs::remove_file(temp).await;
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("whisper command failed (status: {}): {stderr}", output.status);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.trim().to_string())
}

/// Convert audio to 16kHz mono WAV using ffmpeg if available and needed.
async fn maybe_convert_to_wav(audio_path: &Path) -> Result<Option<PathBuf>> {
    let ext = audio_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ext == "wav" {
        return Ok(None);
    }

    if which("ffmpeg").is_err() && which("ffmpeg.exe").is_err() {
        return Ok(None);
    }

    let temp_wav = audio_path.with_extension(format!("temp_{}.wav", rand::random::<u32>()));
    let status = tokio::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(audio_path)
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(&temp_wav)
        .output()
        .await;

    match status {
        Ok(out) if out.status.success() => Ok(Some(temp_wav)),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(error = %stderr, "ffmpeg audio conversion failed, falling back to original");
            Ok(None)
        }
        Err(e) => {
            warn!(error = %e, "ffmpeg failed to execute, falling back to original");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transcription_queue_processes_sequentially() {
        let queue = TranscriptionQueue::new();
        let overview = queue.overview();
        assert_eq!(overview.queue_depth, 0);
        assert!(!overview.is_processing);
    }

    #[test]
    fn test_job_id_format() {
        let id = generate_job_id();
        assert!(id.starts_with("tx_"));
        assert!(id.len() > 10);
    }
}
