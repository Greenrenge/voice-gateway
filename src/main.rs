//! voice-gateway — an OpenAI-compatible `/v1/audio/transcriptions`
//! endpoint backed by MiniMax `asr-1.0` (`/v1/speech_to_text`).
//!
//! Lets any OpenAI-SDK-speaking client (e.g. self-hosted memos voice
//! transcription) use MiniMax ASR without code changes:
//!
//! ```text
//! client ──POST /v1/audio/transcriptions (multipart)──▶ bridge
//! bridge ──POST /v1/speech_to_text (multipart, asr-1.0)──▶ api.minimaxi.com
//! bridge ◀────────────────────── {"text": "..."} ─────────┘
//! client ◀────────────────────── {"text": "..."} ─────────┘
//! ```

use std::net::SocketAddr;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

/// Max upload accepted from clients (OpenAI caps transcriptions at 25 MB).
const MAX_BODY_BYTES: usize = 26 * 1024 * 1024;

/// OpenBot caps discovered model and voice names at 256 bytes (`VOICE_SERVER_LIMITS.name`).
const MAX_DISCOVERY_NAME_BYTES: usize = 256;

#[derive(Clone)]
struct AppState {
    http: reqwest::Client,
    upstream_base: String,
    upstream_model: String,
    tts_model: String,
    default_voice: String,
    minimax_key: String,
    /// Optional inbound bearer; when set, clients must present it.
    bridge_key: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let minimax_key = std::env::var("MINIMAX_API_KEY")
        .map_err(|_| anyhow::anyhow!("MINIMAX_API_KEY is required (MiniMax platform API key)"))?;
    let upstream_base = std::env::var("MINIMAX_API_BASE")
        .unwrap_or_else(|_| "https://api.minimaxi.com".to_string())
        .trim_end_matches('/')
        .to_string();
    let upstream_model = resolve_bounded_env("MINIMAX_MODEL", "asr-1.0")?;
    let tts_model = resolve_bounded_env("MINIMAX_TTS_MODEL", "speech-2.8-turbo")?;
    let default_voice = resolve_bounded_env("MINIMAX_VOICE", "female-shaonv")?;
    let bridge_key = std::env::var("GATEWAY_API_KEY")
        .ok()
        .filter(|k| !k.is_empty());
    let listen: SocketAddr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8090".to_string())
        .parse()?;

    let state = AppState {
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()?,
        upstream_base,
        upstream_model,
        tts_model,
        default_voice,
        minimax_key,
        bridge_key,
    };

    let app = router(state);
    tracing::info!(%listen, "listening");
    let listener = tokio::net::TcpListener::bind(listen).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn normalize_env_model(raw: Option<&str>, default: &str) -> String {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| default.to_string())
}

fn validate_bounded_env(var: &str, raw: Option<&str>, default: &str) -> anyhow::Result<String> {
    let val = normalize_env_model(raw, default);
    anyhow::ensure!(
        val.len() <= MAX_DISCOVERY_NAME_BYTES,
        "{var} must be at most {MAX_DISCOVERY_NAME_BYTES} bytes (got {})",
        val.len()
    );
    Ok(val)
}

fn resolve_bounded_env(var: &str, default: &str) -> anyhow::Result<String> {
    validate_bounded_env(var, std::env::var(var).ok().as_deref(), default)
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/audio/voices", get(list_voices))
        .route("/v1/audio/speech/voices", get(list_voices))
        .route("/v1/audio/transcriptions", post(transcribe))
        .route("/v1/audio/translations", post(transcribe))
        .route("/v1/audio/speech", post(speech))
        .route("/healthz", get(health))
        .route("/", get(index))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}

async fn health() -> &'static str {
    "ok"
}

async fn index() -> Json<Value> {
    Json(json!({
        "service": "voice-gateway",
        "endpoints": [
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/speech",
            "/v1/audio/voices",
            "/v1/audio/speech/voices",
            "/v1/models",
        ],
        "upstream_model_env": "MINIMAX_MODEL (default asr-1.0), MINIMAX_TTS_MODEL (default speech-2.8-turbo)",
    }))
}

async fn list_models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_inbound_auth(&state, &headers) {
        return resp;
    }
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [
                {
                    "id": state.upstream_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "minimax",
                    "task": "automatic-speech-recognition",
                },
                {
                    "id": state.tts_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "minimax",
                    "task": "text-to-speech",
                },
            ],
        })),
    )
        .into_response()
}

async fn list_voices(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = check_inbound_auth(&state, &headers) {
        return resp;
    }
    let mut voices = Vec::with_capacity(DEFAULT_MINIMAX_VOICES.len() + 1);
    let default = state.default_voice.trim();
    if !default.is_empty() && default.len() <= MAX_DISCOVERY_NAME_BYTES {
        voices.push(default.to_string());
    }
    for &voice in DEFAULT_MINIMAX_VOICES {
        if !voices.iter().any(|existing| existing == voice) {
            voices.push(voice.to_string());
        }
    }
    (StatusCode::OK, Json(json!({ "voices": voices }))).into_response()
}

#[derive(Debug)]
struct AudioUpload {
    filename: String,
    content_type: Option<String>,
    data: Bytes,
}

/// Extract the `file` part; other OpenAI fields (`model`, `language`,
/// `prompt`, `response_format`, …) are accepted and ignored — the upstream
/// model is fixed by `MINIMAX_MODEL`.
#[allow(clippy::result_large_err)]
async fn read_upload(mut multipart: Multipart) -> Result<AudioUpload, Response> {
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        openai_error(
            StatusCode::BAD_REQUEST,
            format!("invalid multipart body: {e}"),
        )
    })? {
        if field.name() == Some("file") {
            let filename = field.file_name().unwrap_or("audio").to_string();
            let content_type = field.content_type().map(str::to_string);
            let data = field.bytes().await.map_err(|e| {
                openai_error(StatusCode::BAD_REQUEST, format!("failed to read file: {e}"))
            })?;
            if data.is_empty() {
                return Err(openai_error(StatusCode::BAD_REQUEST, "file part is empty"));
            }
            return Ok(AudioUpload {
                filename,
                content_type,
                data,
            });
        }
    }
    Err(openai_error(
        StatusCode::BAD_REQUEST,
        "multipart field 'file' is required (as OpenAI's /v1/audio/transcriptions)",
    ))
}

async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Response {
    if let Err(resp) = check_inbound_auth(&state, &headers) {
        return resp;
    }

    let upload = match read_upload(multipart).await {
        Ok(u) => u,
        Err(resp) => return resp,
    };
    tracing::info!(
        filename = %upload.filename,
        content_type = upload.content_type.as_deref().unwrap_or("unknown"),
        bytes = upload.data.len(),
        "transcription request"
    );

    let part = reqwest::multipart::Part::bytes(upload.data.to_vec())
        .file_name(upload.filename)
        .mime_str(
            upload
                .content_type
                .as_deref()
                .unwrap_or("application/octet-stream"),
        )
        .expect("static mime string");
    let form = reqwest::multipart::Form::new()
        .text("model", state.upstream_model.clone())
        .part("file", part);

    let url = format!("{}/v1/speech_to_text", state.upstream_base);
    let resp = match state
        .http
        .post(&url)
        .bearer_auth(&state.minimax_key)
        .multipart(form)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "upstream request failed");
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            );
        }
    };

    let status = resp.status();
    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream read failed: {e}"),
            );
        }
    };

    match map_upstream_body(status.as_u16(), &body, wants_verbose(&headers)) {
        Ok(value) => (StatusCode::OK, Json(value)).into_response(),
        Err(err) => {
            tracing::warn!(upstream_status = status.as_u16(), error = %err, "upstream error");
            openai_error(StatusCode::BAD_GATEWAY, err)
        }
    }
}

#[allow(clippy::result_large_err)]
fn check_inbound_auth(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = state.bridge_key.as_deref() else {
        return Ok(());
    };
    let got = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match got {
        Some(k) if constant_time_eq(k.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(openai_error(
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key",
        )),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn wants_verbose(headers: &HeaderMap) -> bool {
    // response_format is normally a form field, but clients that pre-read it
    // into a header are honoured too; the multipart field is parsed upstream
    // and mapped the same way.
    headers
        .get("x-response-format")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == "verbose_json")
        .unwrap_or(false)
}

/// Map MiniMax `/v1/speech_to_text` response into an OpenAI-style response.
///
/// MiniMax success: `{"text": "...", "duration": 2.8, "trace_id": "..."}`
/// MiniMax failure: `{"base_resp": {"status_code": 1004, "status_msg": "..."}}`
fn map_upstream_body(status: u16, body: &str, verbose: bool) -> Result<Value, String> {
    let parsed: Value = serde_json::from_str(body)
        .map_err(|e| format!("upstream returned non-JSON body (HTTP {status}): {e}"))?;

    if let Some(base) = parsed.get("base_resp") {
        let code = base.get("status_code").and_then(Value::as_i64).unwrap_or(0);
        if code != 0 {
            let msg = base
                .get("status_msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown upstream error");
            return Err(format!("minimax error {code}: {msg}"));
        }
    }

    let text = parsed
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("upstream response has no 'text' field (HTTP {status})"))?;

    if verbose {
        let duration = parsed
            .get("duration")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Ok(json!({
            "task": "transcribe",
            "language": null,
            "duration": duration,
            "text": text,
            "segments": [],
        }))
    } else {
        Ok(json!({ "text": text }))
    }
}

fn openai_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": { "message": message.into(), "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

// ---------------- TTS: OpenAI /v1/audio/speech -> MiniMax /v1/t2a_v2 ----------------

#[derive(Debug, serde::Deserialize)]
struct TtsRequest {
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>, // accepted and ignored; fixed by MINIMAX_TTS_MODEL
    input: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f64>,
    #[serde(default)]
    response_format: Option<String>,
}

const DEFAULT_MINIMAX_VOICES: &[&str] = &[
    "female-shaonv",
    "female-tianmei",
    "female-yujie",
    "male-qn-qingse",
    "male-qn-jingying",
    "male-qn-badao",
    "presenter_female",
    "audiobook_female_1",
];

const OPENAI_VOICE_ALIASES: &[(&str, &str)] = &[
    ("alloy", "female-shaonv"),
    ("ash", "male-qn-jingying"),
    ("coral", "female-tianmei"),
    ("echo", "male-qn-qingse"),
    ("fable", "audiobook_female_1"),
    ("nova", "female-yujie"),
    ("onyx", "male-qn-badao"),
    ("sage", "presenter_female"),
    ("shimmer", "female-tianmei"),
];

/// Map empty -> default, standard OpenAI voices -> valid MiniMax voices,
/// preserve MiniMax voice ID shapes (including mixed-case system/cloned IDs),
/// and fall back to prefixing `female-` for bare shorthands like `shaonv`.
fn normalize_voice(voice: Option<&str>, default: &str) -> String {
    let Some(trimmed) = voice.map(str::trim).filter(|s| !s.is_empty()) else {
        return default.trim().to_string();
    };
    let lower = trimmed.to_lowercase();
    if let Some((_, mapped)) = OPENAI_VOICE_ALIASES
        .iter()
        .find(|(alias, _)| *alias == lower)
    {
        return (*mapped).to_string();
    }
    if lower.starts_with("female-")
        || lower.starts_with("male-")
        || lower == "presenter_female"
        || lower == "audiobook_female_1"
    {
        return lower;
    }
    if trimmed.contains('_') || trimmed.contains('-') {
        return trimmed.to_string();
    }
    format!("female-{lower}")
}

const TTS_FORMATS: &[&str] = &["mp3", "wav", "flac", "pcm", "opus"];

/// MiniMax t2a_v2 cannot emit opus; opus requests are synthesized as mp3 upstream
/// and transcoded to Ogg/Opus locally before being served.
fn upstream_t2a_format(requested: &str) -> &str {
    if requested == "opus" {
        "mp3"
    } else {
        requested
    }
}

fn content_type_for(format: &str) -> &'static str {
    match format {
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "pcm" => "audio/pcm",
        "opus" => "audio/ogg",
        _ => "audio/mpeg",
    }
}

fn build_t2a_body(
    model: &str,
    input: &str,
    voice: Option<&str>,
    speed: Option<f64>,
    default_voice: &str,
    format: &str,
) -> Value {
    json!({
        "model": model,
        "text": input,
        "stream": false,
        "language_boost": "auto",
        "output_format": "url",
        "voice_setting": {
            "voice_id": normalize_voice(voice, default_voice),
            "speed": speed.unwrap_or(1.0).clamp(0.5, 2.0),
            "vol": 1,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": format,
            "channel": 1,
        },
    })
}

async fn speech(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TtsRequest>,
) -> Response {
    if let Err(resp) = check_inbound_auth(&state, &headers) {
        return resp;
    }
    if req.input.trim().is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "'input' is required");
    }
    let format = match req.response_format.as_deref().map(str::to_lowercase) {
        None => "mp3".to_string(),
        Some(f) if TTS_FORMATS.contains(&f.as_str()) => f,
        Some(f) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("unsupported response_format '{f}' (supported: mp3, wav, flac, pcm, opus)"),
            );
        }
    };
    tracing::info!(
        bytes = req.input.len(),
        voice = %normalize_voice(req.voice.as_deref(), &state.default_voice),
        format = %format,
        "speech request"
    );

    let body = build_t2a_body(
        &state.tts_model,
        &req.input,
        req.voice.as_deref(),
        req.speed,
        &state.default_voice,
        upstream_t2a_format(&format),
    );
    let url = format!("{}/v1/t2a_v2", state.upstream_base);
    let resp = match state
        .http
        .post(&url)
        .bearer_auth(&state.minimax_key)
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "t2a_v2 request failed");
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream request failed: {e}"),
            );
        }
    };
    let status = resp.status();
    let payload: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream returned non-JSON body (HTTP {status}): {e}"),
            );
        }
    };

    if let Some(base) = payload.get("base_resp") {
        let code = base.get("status_code").and_then(Value::as_i64).unwrap_or(0);
        if code != 0 {
            let msg = base
                .get("status_msg")
                .and_then(Value::as_str)
                .unwrap_or("unknown upstream error");
            tracing::warn!(code, "minimax t2a_v2 error");
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("minimax error {code}: {msg}"),
            );
        }
    }

    let audio_url = match payload
        .get("data")
        .and_then(|d| d.get("audio"))
        .and_then(Value::as_str)
    {
        Some(u) => u.to_string(),
        None => {
            tracing::warn!("t2a_v2 parse failure: no 'data.audio' URL");
            return openai_error(
                StatusCode::BAD_GATEWAY,
                "upstream response has no 'data.audio' URL",
            );
        }
    };

    let audio = match state.http.get(&audio_url).send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("audio download failed with HTTP {}", r.status()),
            );
        }
        Err(e) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("audio download failed: {e}"),
            );
        }
    };
    let content_type = audio
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .filter(|ct| ct.starts_with("audio/"))
        .map(str::to_string)
        .unwrap_or_else(|| content_type_for(&format).to_string());
    let bytes = match audio.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return openai_error(StatusCode::BAD_GATEWAY, format!("audio read failed: {e}"));
        }
    };

    if format == "opus" {
        let (opus_bytes, ct) = match transcode_to_opus_ogg(&bytes).await {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(error = %e, "opus transcode failed");
                return openai_error(
                    StatusCode::BAD_GATEWAY,
                    format!("opus transcode failed: {e}"),
                );
            }
        };
        tracing::info!(bytes = opus_bytes.len(), "speech served (transcoded opus)");
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, ct)],
            opus_bytes,
        )
            .into_response();
    }

    tracing::info!(bytes = bytes.len(), "speech served");
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, content_type)],
        bytes,
    )
        .into_response()
}

/// Transcode upstream audio (mp3) to Telegram-style Ogg/Opus (mono, 24 kHz, 64 kbps)
/// via ffmpeg, matching the production Hermes wrapper's conversion settings.
async fn transcode_to_opus_ogg(input: &[u8]) -> anyhow::Result<(Vec<u8>, &'static str)> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let src = std::env::temp_dir().join(format!("vg-tts-{stamp}.mp3"));
    let dst = std::env::temp_dir().join(format!("vg-tts-{stamp}.ogg"));
    tokio::fs::write(&src, input).await?;
    let out = tokio::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&src)
        .args([
            "-ac", "1", "-ar", "24000", "-c:a", "libopus", "-b:a", "64k", "-f", "ogg",
        ])
        .arg(&dst)
        .output()
        .await;
    // Remove temp files regardless of outcome.
    let _ = tokio::fs::remove_file(&src).await;
    let result = match out {
        Ok(o) if o.status.success() => match tokio::fs::read(&dst).await {
            Ok(data) if !data.is_empty() => Ok((data, "audio/ogg" as &'static str)),
            Ok(_) => Err(anyhow::anyhow!("ffmpeg produced empty output")),
            Err(e) => Err(anyhow::anyhow!("ffmpeg output unreadable: {e}")),
        },
        Ok(o) => Err(anyhow::anyhow!(
            "ffmpeg exit {}: {}",
            o.status,
            String::from_utf8_lossy(&o.stderr).trim()
        )),
        Err(e) => Err(anyhow::anyhow!("ffmpeg spawn failed: {e}")),
    };
    let _ = tokio::fs::remove_file(&dst).await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn maps_success_to_openai_shape() {
        let body = r#"{"text":"你好，世界","duration":2.8,"trace_id":"t"}"#;
        let v = map_upstream_body(200, body, false).unwrap();
        assert_eq!(v["text"], "你好，世界");
        assert!(v.get("duration").is_none());
    }

    #[tokio::test]
    async fn maps_verbose_shape() {
        let body = r#"{"text":"hi","duration":1.5}"#;
        let v = map_upstream_body(200, body, true).unwrap();
        assert_eq!(v["task"], "transcribe");
        assert_eq!(v["duration"], 1.5);
    }

    #[tokio::test]
    async fn surfaces_base_resp_errors() {
        let body = r#"{"base_resp":{"status_code":1004,"status_msg":"invalid api key"}}"#;
        let err = map_upstream_body(200, body, false).unwrap_err();
        assert!(err.contains("minimax error 1004"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_body_without_text() {
        let err = map_upstream_body(200, r#"{"foo":1}"#, false).unwrap_err();
        assert!(err.contains("no 'text' field"));
    }

    #[tokio::test]
    async fn missing_file_part_is_400() {
        let state = test_state();
        let app = router(state);
        // multipart without a "file" part
        let body =
            "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["error"]["message"]
                .as_str()
                .unwrap()
                .contains("'file' is required")
        );
    }

    #[tokio::test]
    async fn end_to_end_with_mock_upstream() {
        // stub MiniMax upstream
        let upstream = Router::new().route(
            "/v1/speech_to_text",
            post(|headers: HeaderMap, mut mp: Multipart| async move {
                assert_eq!(
                    headers.get("authorization").unwrap(),
                    "Bearer minimax-secret"
                );
                let mut got_model = None;
                let mut got_audio: Option<usize> = None;
                while let Some(f) = mp.next_field().await.unwrap() {
                    match f.name() {
                        Some("model") => got_model = Some(f.text().await.unwrap()),
                        Some("file") => {
                            got_audio = Some(f.bytes().await.unwrap().len());
                        }
                        _ => {}
                    }
                }
                assert_eq!(got_model.as_deref(), Some("asr-1.0"));
                assert!(got_audio.unwrap_or(0) > 0);
                Json(json!({"text":"mock 转写结果","duration":1.0}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let state = AppState {
            http: reqwest::Client::new(),
            upstream_base: format!("http://{addr}"),
            upstream_model: "asr-1.0".into(),
            tts_model: "speech-2.8-turbo".into(),
            default_voice: "female-shaonv".into(),
            minimax_key: "minimax-secret".into(),
            bridge_key: Some("bridge-key".into()),
        };
        let app = router(state);

        let body = "--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFxxxx\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["text"], "mock 转写结果");
    }

    #[tokio::test]
    async fn rejects_bad_inbound_key() {
        let state = test_state();
        let app = router(state);
        let body = "--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nx\r\n--B--\r\n";
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer wrong")
            .header("content-type", "multipart/form-data; boundary=B")
            .body(Body::from(body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    fn test_state() -> AppState {
        AppState {
            http: reqwest::Client::new(),
            upstream_base: "http://127.0.0.1:1".into(),
            upstream_model: "asr-1.0".into(),
            tts_model: "speech-2.8-turbo".into(),
            default_voice: "female-shaonv".into(),
            minimax_key: "k".into(),
            bridge_key: Some("bridge-key".into()),
        }
    }

    #[test]
    fn voice_normalization_matches_wrapper() {
        assert_eq!(normalize_voice(None, "female-shaonv"), "female-shaonv");
        assert_eq!(normalize_voice(Some(""), "female-shaonv"), "female-shaonv");
        assert_eq!(
            normalize_voice(Some("shaonv"), "female-shaonv"),
            "female-shaonv"
        );
        assert_eq!(
            normalize_voice(Some("FEMALE-Tianmei"), "female-shaonv"),
            "female-tianmei"
        );
        assert_eq!(
            normalize_voice(Some("male-qn-qingse"), "female-shaonv"),
            "male-qn-qingse"
        );
        assert_eq!(
            normalize_voice(Some("presenter_female"), "female-shaonv"),
            "presenter_female"
        );
    }

    #[test]
    fn t2a_body_shape() {
        let body = build_t2a_body(
            "speech-2.8-turbo",
            "你好",
            Some("shaonv"),
            Some(1.5),
            "d",
            "mp3",
        );
        assert_eq!(body["model"], "speech-2.8-turbo");
        assert_eq!(body["voice_setting"]["voice_id"], "female-shaonv");
        assert_eq!(body["voice_setting"]["speed"], 1.5);
        assert_eq!(body["audio_setting"]["format"], "mp3");
        assert_eq!(body["output_format"], "url");
    }

    #[tokio::test]
    async fn end_to_end_speech_with_mock_upstream() {
        // stub MiniMax t2a_v2 + audio URL on one server; the t2a stub embeds
        // its own absolute audio URL (resolved after binding).
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let audio_url = format!("http://{addr}/audio/x.mp3");
        let upstream = Router::new()
            .route(
                "/v1/t2a_v2",
                post(move |Json(body): Json<Value>| {
                    let audio_url = audio_url.clone();
                    async move {
                        assert_eq!(body["model"], "speech-2.8-turbo");
                        assert_eq!(body["voice_setting"]["voice_id"], "female-shaonv");
                        Json(json!({"data": {"audio": audio_url}, "base_resp": {"status_code": 0}}))
                    }
                }),
            )
            .route(
                "/audio/x.mp3",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "audio/mpeg")],
                        Bytes::from_static(b"ID3DATA"),
                    )
                }),
            );
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let state = AppState {
            http: reqwest::Client::new(),
            upstream_base: format!("http://{addr}"),
            upstream_model: "asr-1.0".into(),
            tts_model: "speech-2.8-turbo".into(),
            default_voice: "female-shaonv".into(),
            minimax_key: "minimax-secret".into(),
            bridge_key: None,
        };
        let app = router(state);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": "tts-1",
                    "voice": "shaonv",
                    "input": "你好世界",
                    "response_format": "mp3"
                }))
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"ID3DATA");
        assert!(ct.starts_with("audio/"));
    }

    #[tokio::test]
    async fn speech_rejects_empty_input() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"  "}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn speech_rejects_bad_format() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer bridge-key")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"input":"hi","response_format":"aiff"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn opus_requests_synthesize_as_mp3_upstream() {
        assert_eq!(upstream_t2a_format("opus"), "mp3");
        assert_eq!(upstream_t2a_format("wav"), "wav");
        assert_eq!(upstream_t2a_format("mp3"), "mp3");
    }

    #[tokio::test]
    async fn models_returns_configured_asr_and_tts() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer bridge-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(!resp.status().is_redirection());
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.len() < 1024, "payload too large: {}", bytes.len());
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);

        assert_eq!(data[0]["id"], "asr-1.0");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["created"], 0);
        assert_eq!(data[0]["owned_by"], "minimax");
        assert_eq!(data[0]["task"], "automatic-speech-recognition");

        assert_eq!(data[1]["id"], "speech-2.8-turbo");
        assert_eq!(data[1]["object"], "model");
        assert_eq!(data[1]["created"], 0);
        assert_eq!(data[1]["owned_by"], "minimax");
        assert_eq!(data[1]["task"], "text-to-speech");

        for entry in data {
            let id = entry["id"].as_str().unwrap();
            assert!(!id.is_empty() && id.len() <= 256);
        }
    }

    #[tokio::test]
    async fn models_reflects_custom_env_models() {
        let mut state = test_state();
        state.upstream_model = "custom-asr-v2".into();
        state.tts_model = "custom-tts-v3".into();
        let app = router(state);
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer bridge-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "custom-asr-v2");
        assert_eq!(data[0]["task"], "automatic-speech-recognition");
        assert_eq!(data[1]["id"], "custom-tts-v3");
        assert_eq!(data[1]["task"], "text-to-speech");
    }

    #[tokio::test]
    async fn models_rejects_missing_or_wrong_key() {
        let req_missing = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp_missing = router(test_state()).oneshot(req_missing).await.unwrap();
        assert_eq!(resp_missing.status(), StatusCode::UNAUTHORIZED);
        let bytes_missing = resp_missing.into_body().collect().await.unwrap().to_bytes();
        let err_missing: Value = serde_json::from_slice(&bytes_missing).unwrap();
        assert!(err_missing["error"]["message"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false));

        let req_wrong = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer wrong")
            .body(Body::empty())
            .unwrap();
        let resp_wrong = router(test_state()).oneshot(req_wrong).await.unwrap();
        assert_eq!(resp_wrong.status(), StatusCode::UNAUTHORIZED);
        let bytes_wrong = resp_wrong.into_body().collect().await.unwrap().to_bytes();
        let err_wrong: Value = serde_json::from_slice(&bytes_wrong).unwrap();
        assert!(err_wrong["error"]["message"]
            .as_str()
            .map(|s| !s.is_empty())
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn models_allows_open_mode() {
        let mut state = test_state();
        state.bridge_key = None;
        let app = router(state);
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn index_includes_models_endpoint() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let endpoints = v["endpoints"].as_array().unwrap();
        assert!(endpoints.iter().any(|e| e == "/v1/models"));
    }

    #[test]
    fn model_env_normalization_trims_and_falls_back() {
        assert_eq!(normalize_env_model(None, "asr-1.0"), "asr-1.0");
        assert_eq!(normalize_env_model(Some(""), "asr-1.0"), "asr-1.0");
        assert_eq!(
            normalize_env_model(Some("  \t\n "), "speech-2.8-turbo"),
            "speech-2.8-turbo"
        );
        assert_eq!(
            normalize_env_model(Some("  custom-asr  "), "asr-1.0"),
            "custom-asr"
        );
    }

    #[tokio::test]
    async fn models_allows_identical_asr_and_tts_ids() {
        let mut state = test_state();
        state.upstream_model = "shared-voice-model".into();
        state.tts_model = "shared-voice-model".into();
        let app = router(state);
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("authorization", "Bearer bridge-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "shared-voice-model");
        assert_eq!(data[0]["task"], "automatic-speech-recognition");
        assert_eq!(data[1]["id"], "shared-voice-model");
        assert_eq!(data[1]["task"], "text-to-speech");
    }

    #[tokio::test]
    async fn voices_returns_default_first_and_deduplicated() {
        let mut state = test_state();
        state.default_voice = "English_Graceful_Lady".into();
        for path in ["/v1/audio/voices", "/v1/audio/speech/voices"] {
            let req = axum::http::Request::builder()
                .method("GET")
                .uri(path)
                .header("authorization", "Bearer bridge-key")
                .body(Body::empty())
                .unwrap();
            let resp = router(state.clone()).oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(!resp.status().is_redirection());
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let v: Value = serde_json::from_slice(&bytes).unwrap();
            let voices: Vec<&str> = v["voices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap())
                .collect();
            assert_eq!(voices.first().copied(), Some("English_Graceful_Lady"));
            assert!(voices.contains(&"female-shaonv"));
            assert!(voices
                .iter()
                .all(|voice| !voice.is_empty() && voice.len() <= 256));
        }
    }

    #[tokio::test]
    async fn voices_deduplicates_default_when_already_listed() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/v1/audio/voices")
            .header("authorization", "Bearer bridge-key")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let voices: Vec<&str> = v["voices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let count = voices.iter().filter(|&&x| x == "female-shaonv").count();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn voices_rejects_missing_or_wrong_key() {
        for path in ["/v1/audio/voices", "/v1/audio/speech/voices"] {
            let req = axum::http::Request::builder()
                .method("GET")
                .uri(path)
                .header("authorization", "Bearer wrong")
                .body(Body::empty())
                .unwrap();
            let resp = router(test_state()).oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn index_includes_voice_endpoints() {
        let app = router(test_state());
        let req = axum::http::Request::builder()
            .method("GET")
            .uri("/")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let endpoints = v["endpoints"].as_array().unwrap();
        assert!(endpoints.iter().any(|e| e == "/v1/audio/voices"));
        assert!(endpoints.iter().any(|e| e == "/v1/audio/speech/voices"));
    }

    #[test]
    fn voice_normalization_maps_openai_and_preserves_minimax_ids() {
        assert_eq!(normalize_voice(Some("alloy"), "female-shaonv"), "female-shaonv");
        assert_eq!(normalize_voice(Some("ALLOY"), "female-shaonv"), "female-shaonv");
        assert_eq!(normalize_voice(Some("echo"), "female-shaonv"), "male-qn-qingse");
        assert_eq!(normalize_voice(Some("onyx"), "female-shaonv"), "male-qn-badao");
        assert_eq!(normalize_voice(Some("nova"), "female-shaonv"), "female-yujie");
        assert_eq!(normalize_voice(Some("shimmer"), "female-shaonv"), "female-tianmei");
        assert_eq!(normalize_voice(Some("ash"), "female-shaonv"), "male-qn-jingying");
        assert_eq!(normalize_voice(Some("coral"), "female-shaonv"), "female-tianmei");
        assert_eq!(normalize_voice(Some("fable"), "female-shaonv"), "audiobook_female_1");
        assert_eq!(normalize_voice(Some("sage"), "female-shaonv"), "presenter_female");
        assert_eq!(
            normalize_voice(Some("English_Graceful_Lady"), "female-shaonv"),
            "English_Graceful_Lady"
        );
        assert_eq!(
            normalize_voice(Some("moss_audio_1234"), "female-shaonv"),
            "moss_audio_1234"
        );
        assert_eq!(
            normalize_voice(None, "English_Graceful_Lady"),
            "English_Graceful_Lady"
        );
    }

    #[test]
    fn validate_bounded_env_rejects_overlong_values() {
        assert_eq!(
            validate_bounded_env("MINIMAX_MODEL", Some("  "), "asr-1.0").unwrap(),
            "asr-1.0"
        );
        let exact = "a".repeat(256);
        assert_eq!(
            validate_bounded_env("MINIMAX_MODEL", Some(&exact), "asr-1.0").unwrap(),
            exact
        );
        let too_long = "a".repeat(257);
        assert!(validate_bounded_env("MINIMAX_MODEL", Some(&too_long), "asr-1.0").is_err());
    }

    #[tokio::test]
    async fn speech_wav_falls_back_from_octet_stream_content_type() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let audio_url = format!("http://{addr}/audio/out.wav");
        let upstream = Router::new()
            .route(
                "/v1/t2a_v2",
                post(move |Json(body): Json<Value>| {
                    let audio_url = audio_url.clone();
                    async move {
                        assert_eq!(body["voice_setting"]["voice_id"], "female-shaonv");
                        assert_eq!(body["audio_setting"]["format"], "wav");
                        Json(json!({"data": {"audio": audio_url}, "base_resp": {"status_code": 0}}))
                    }
                }),
            )
            .route(
                "/audio/out.wav",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "application/octet-stream")],
                        Bytes::from_static(b"RIFF\x24\x00\x00\x00WAVEfmt "),
                    )
                }),
            );
        tokio::spawn(async move { axum::serve(listener, upstream).await.unwrap() });

        let state = AppState {
            http: reqwest::Client::new(),
            upstream_base: format!("http://{addr}"),
            upstream_model: "asr-1.0".into(),
            tts_model: "speech-2.8-turbo".into(),
            default_voice: "female-shaonv".into(),
            minimax_key: "minimax-secret".into(),
            bridge_key: None,
        };
        let app = router(state);
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&json!({
                    "model": "speech-2.8-turbo",
                    "voice": "alloy",
                    "input": "Hello OpenBot",
                    "response_format": "wav"
                }))
                .unwrap(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("audio/wav")
        );
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.starts_with(b"RIFF"));
    }
}
