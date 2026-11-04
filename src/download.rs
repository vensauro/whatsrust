//! File download and media caching utilities for whatsrust.
//!
//! Provides downloading from:
//! - WhatsApp inbound media messages by message ID
//! - External HTTP/HTTPS URLs
//! - Auto-saving received inbound attachments to disk

use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::bridge::{InboundContent, WhatsAppBridge};

/// Metadata returned after a successful download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedFileInfo {
    pub path: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64_data: Option<String>,
}

/// Get the base downloads directory (defaults to `./downloads`).
pub fn downloads_dir() -> PathBuf {
    let dir = std::env::var("WHATSRUST_DOWNLOADS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("downloads"));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Derive a file extension from a MIME type.
pub fn extension_for_mime(mime: &str) -> Option<&'static str> {
    let lower = mime.to_lowercase();
    let base_mime = lower.split(';').next().unwrap_or(&lower).trim();
    match base_mime {
        "audio/ogg" | "audio/opus" => Some("ogg"),
        "audio/mpeg" | "audio/mp3" => Some("mp3"),
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" | "audio/aac" => Some("m4a"),
        "audio/wav" | "audio/x-wav" | "audio/wave" => Some("wav"),
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "video/mp4" => Some("mp4"),
        "video/quicktime" => Some("mov"),
        "video/webm" => Some("webm"),
        "video/3gpp" => Some("3gp"),
        "application/pdf" => Some("pdf"),
        "application/json" => Some("json"),
        "text/plain" => Some("txt"),
        "application/zip" => Some("zip"),
        "application/octet-stream" => None,
        _ => None,
    }
}

/// Sanitize a string to be a safe single filename component (no slashes, no traversal).
pub fn sanitize_filename(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => '_',
            c if c.is_ascii_control() => '_',
            c => c,
        })
        .collect();
    let trimmed = clean.trim_matches(|c| c == '.' || c == ' ' || c == '_');
    if trimmed.is_empty() {
        "downloaded_file".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Auto-save inbound media to the downloads directory.
/// Returns the relative path as a string if saved.
pub fn auto_save_inbound_media(msg_id: &str, content: &InboundContent) -> Option<String> {
    let (data, suggested_name) = match content {
        InboundContent::Audio { data, mime, .. } => {
            let ext = extension_for_mime(mime).unwrap_or("ogg");
            (data.as_ref(), format!("{msg_id}.{ext}"))
        }
        InboundContent::Image { data, mime, .. } => {
            let ext = extension_for_mime(mime).unwrap_or("jpg");
            (data.as_ref(), format!("{msg_id}.{ext}"))
        }
        InboundContent::Video { data, mime, .. } => {
            let ext = extension_for_mime(mime).unwrap_or("mp4");
            (data.as_ref(), format!("{msg_id}.{ext}"))
        }
        InboundContent::Document { data, filename, .. } => {
            let safe_name = sanitize_filename(filename);
            (data.as_ref(), format!("{msg_id}_{safe_name}"))
        }
        InboundContent::Sticker { data, mime, .. } => {
            let ext = extension_for_mime(mime).unwrap_or("webp");
            (data.as_ref(), format!("{msg_id}.{ext}"))
        }
        _ => return None,
    };

    let dir = downloads_dir();
    let file_path = dir.join(suggested_name);

    if file_path.exists() {
        return Some(file_path.to_string_lossy().to_string());
    }

    match std::fs::write(&file_path, data) {
        Ok(_) => {
            debug!(path = %file_path.display(), size = data.len(), "auto-saved inbound media");
            Some(file_path.to_string_lossy().to_string())
        }
        Err(e) => {
            warn!(error = %e, path = %file_path.display(), "failed to auto-save inbound media");
            None
        }
    }
}

/// Check if a media file for a message ID already exists in the downloads directory.
pub fn find_existing_message_file(msg_id: &str) -> Option<PathBuf> {
    let dir = downloads_dir();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with(msg_id) {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Download media for a WhatsApp message (from disk cache or bridge client).
pub async fn download_from_message(
    bridge: &WhatsAppBridge,
    msg_id: &str,
    output_path: Option<&str>,
    include_base64: bool,
) -> Result<DownloadedFileInfo> {
    let target_dest = output_path.map(PathBuf::from);

    // 1. Check if already auto-saved to disk
    if let Some(existing) = find_existing_message_file(msg_id) {
        let dest = match target_dest {
            Some(d) => {
                if d != existing {
                    std::fs::copy(&existing, &d)
                        .with_context(|| format!("failed to copy from {} to {}", existing.display(), d.display()))?;
                    d
                } else {
                    existing
                }
            }
            None => existing,
        };

        let meta = std::fs::metadata(&dest)
            .with_context(|| format!("failed to read metadata for {}", dest.display()))?;
        let filename = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
        let mime = mime_guess_from_path(&dest);

        let base64_data = if include_base64 && meta.len() <= 10 * 1024 * 1024 {
            let bytes = std::fs::read(&dest)?;
            use base64::Engine;
            Some(base64::engine::general_purpose::STANDARD.encode(&bytes))
        } else {
            None
        };

        return Ok(DownloadedFileInfo {
            path: dest.to_string_lossy().to_string(),
            filename,
            mime_type: mime,
            size_bytes: meta.len(),
            base64_data,
        });
    }

    // 2. Fetch via bridge from WhatsApp servers
    let (data, mime, suggested_name) = bridge.download_media_for_message(msg_id).await?;
    let dest = match target_dest {
        Some(d) => {
            if d.is_dir() {
                d.join(&suggested_name)
            } else {
                d
            }
        }
        None => downloads_dir().join(&suggested_name),
    };

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&dest, &data)
        .with_context(|| format!("failed to save downloaded media to {}", dest.display()))?;

    let filename = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
    let base64_data = if include_base64 && data.len() <= 10 * 1024 * 1024 {
        use base64::Engine;
        Some(base64::engine::general_purpose::STANDARD.encode(&data))
    } else {
        None
    };

    info!(path = %dest.display(), bytes = data.len(), "message media downloaded");

    Ok(DownloadedFileInfo {
        path: dest.to_string_lossy().to_string(),
        filename,
        mime_type: mime,
        size_bytes: data.len() as u64,
        base64_data,
    })
}

/// Download an arbitrary file from an HTTP/HTTPS URL.
pub fn download_from_url(
    url: &str,
    output_path: Option<&str>,
    include_base64: bool,
) -> Result<DownloadedFileInfo> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        anyhow::bail!("invalid URL scheme: URL must start with http:// or https://");
    }

    let mut response = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("HTTP GET failed for {url}: {e}"))?;

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_string();

    let suggested_filename = {
        let path_no_query = url.split('?').next().unwrap_or(url);
        let from_path = path_no_query
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .map(sanitize_filename);

        match from_path {
            Some(name) if !name.is_empty() && name.contains('.') => name,
            _ => {
                let ext = extension_for_mime(&content_type).unwrap_or("bin");
                let ts = chrono::Utc::now().timestamp();
                format!("download_{ts}.{ext}")
            }
        }
    };

    let dest = match output_path {
        Some(p) => {
            let path = PathBuf::from(p);
            if path.is_dir() {
                path.join(&suggested_filename)
            } else {
                path
            }
        }
        None => downloads_dir().join(&suggested_filename),
    };

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    use std::io::Read;
    let mut reader = response.body_mut().as_reader();
    let mut data = Vec::new();
    reader.read_to_end(&mut data)
        .with_context(|| format!("failed reading HTTP response body from {url}"))?;

    std::fs::write(&dest, &data)
        .with_context(|| format!("failed saving file to {}", dest.display()))?;

    let filename = dest.file_name().and_then(|n| n.to_str()).unwrap_or("file").to_string();
    let base64_data = if include_base64 && data.len() <= 10 * 1024 * 1024 {
        use base64::Engine;
        Some(base64::engine::general_purpose::STANDARD.encode(&data))
    } else {
        None
    };

    info!(url = %url, path = %dest.display(), bytes = data.len(), "file downloaded from URL");

    Ok(DownloadedFileInfo {
        path: dest.to_string_lossy().to_string(),
        filename,
        mime_type: content_type,
        size_bytes: data.len() as u64,
        base64_data,
    })
}

/// Unified file download: handles either `message_id` or `url`.
pub async fn download_file(
    bridge: &WhatsAppBridge,
    message_id: Option<&str>,
    url: Option<&str>,
    output_path: Option<&str>,
    include_base64: bool,
) -> Result<DownloadedFileInfo> {
    match (message_id, url) {
        (Some(mid), None) => download_from_message(bridge, mid, output_path, include_base64).await,
        (None, Some(u)) => {
            let u_owned = u.to_string();
            let out = output_path.map(|s| s.to_string());
            tokio::task::spawn_blocking(move || {
                download_from_url(&u_owned, out.as_deref(), include_base64)
            })
            .await
            .map_err(|e| anyhow::anyhow!("download task panicked: {e}"))?
        }
        (Some(_), Some(_)) => anyhow::bail!("specify either 'message_id' or 'url', not both"),
        (None, None) => anyhow::bail!("must specify either 'message_id' or 'url'"),
    }
}

/// Simple MIME detection by extension.
fn mime_guess_from_path(path: &Path) -> String {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "ogg" | "opus" => "audio/ogg",
        "mp3" => "audio/mpeg",
        "m4a" | "aac" => "audio/mp4",
        "wav" => "audio/wav",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        "pdf" => "application/pdf",
        "json" => "application/json",
        "txt" => "text/plain",
        _ => "application/octet-stream",
    }.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("valid_name.txt"), "valid_name.txt");
        assert_eq!(sanitize_filename("../../etc/passwd"), "etc_passwd");
        assert_eq!(sanitize_filename("file:name?.mp3"), "file_name_.mp3");
    }

    #[test]
    fn test_extension_for_mime() {
        assert_eq!(extension_for_mime("audio/ogg; codecs=opus"), Some("ogg"));
        assert_eq!(extension_for_mime("audio/mpeg"), Some("mp3"));
        assert_eq!(extension_for_mime("image/jpeg"), Some("jpg"));
        assert_eq!(extension_for_mime("application/pdf"), Some("pdf"));
        assert_eq!(extension_for_mime("unknown/xyz"), None);
    }
}
