#!/usr/bin/env node
// free-pi OpenAI-compatible proxy
// -------------------------------
// Exposes a local OpenAI-compatible API (chat completions + models + tool
// calling) backed by the free-pi backend, reusing the JWT from your
// `npx free-pi-cli` sign-in. Works with any OpenAI client (openai SDK,
// LangChain, LlamaIndex, curl, ...).
//
// Usage:   node proxy.mjs            (env: PORT=8085, HOST=127.0.0.1,
//                                     FREEPI_BASE_URL, PROXY_API_KEY)
// Client:  baseURL="http://127.0.0.1:8085/v1", apiKey="anything"

import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import zlib from "node:zlib";
import { Readable, Transform } from "node:stream";
import { execFile } from "node:child_process";
import { McpManager } from "./mcp.mjs";

const PORT = Number(process.env.PORT || 8085);
const HOST = process.env.HOST || "127.0.0.1";
const BASE = (process.env.FREEPI_BASE_URL || "https://api.freepi.ai").replace(/\/+$/, "");
const PROXY_API_KEY = process.env.PROXY_API_KEY || ""; // set to require Bearer auth on the proxy

// Maximum context / reasoning defaults (match free-pi's own provider config).
// Used as fallback when a model's real specs are unknown; per-model specs are
// fetched from freepi.ai's public model pages (e.g. glm-5.3-flash: 1M ctx,
// 131072 max output).
const CONTEXT_WINDOW = 200000;
const MAX_OUTPUT_TOKENS = 8192;
const SPECS_URL = process.env.FREEPI_SPECS_URL || "https://freepi.ai/models.md";
const SPECS_TTL_MS = 24 * 60 * 60 * 1000;
const SPECS_CACHE_FILE = path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "model-specs-cache.json");

// Specs for the free models, embedded from their official pages
// (freepi.ai/models/<id>.md) so they work with zero network access.
const BUILT_IN_SPECS = {
  "glm-5.3-flash": {
    contextWindow: 1048576, maxOutput: 131072,
    inputs: ["text", "image", "video"], toolCalling: true, reasoningEffort: true,
  },
  "deepseek-v4.1-flash": {
    contextWindow: 1048576, maxOutput: 943718,
    inputs: ["text", "image"], toolCalling: true, reasoningEffort: true,
  },
};
// Reasoning effort injected when the client doesn't send one. The backend
// accepts "high" (verified: produces reasoning tokens); "xhigh" is accepted
// with HTTP 200 but silently does nothing, so "high" is the effective max.
// Override with FREEPI_PROXY_REASONING (off|minimal|low|medium|high), or set
// FREEPI_PROXY_REASONING=client to never inject.
const REASONING_EFFORT = process.env.FREEPI_PROXY_REASONING || "high";
// FREEPI_PROXY_FORCE=1 also overrides a client's own (smaller) choices.
const FORCE_MAXIMA = process.env.FREEPI_PROXY_FORCE === "1";

// Default max_tokens when the client sends none. free-pi's per-model max output
// is huge (glm-5.3-flash: 131,072 / deepseek-v4.1-flash: 943,718) and the
// backend counts prompt + requested output against the model window, so
// reserving the whole output budget turns otherwise-fine requests into HTTP 413
// "context_too_long". The fallback therefore stays modest; override with
// FREEPI_PROXY_MAX_DEFAULT_OUTPUT, or set FREEPI_PROXY_FORCE=1 to go maximal.
const MAX_DEFAULT_OUTPUT = Number(process.env.FREEPI_PROXY_MAX_DEFAULT_OUTPUT || 32768);

// Context-overflow recovery ("This conversation is too long — run /compact or
// start a new session with /new", HTTP 413 / code `context_too_long`).
// The proxy forwards the client's history untouched, so once a conversation
// outgrows the model window every following turn fails the same way. Instead of
// failing forever, drop the oldest turns (keeping the system prompt and the
// newest turns), retry, and repeat until the backend accepts the request — the
// proxy-side equivalent of /compact. Tunables:
//   FREEPI_PROXY_COMPACT_TRIES       retries after a 413 (default 4)
//   FREEPI_PROXY_COMPACT_KEEP_RATIO  history shrinks to this fraction (0.5)
//   FREEPI_PROXY_COMPACT_MIN_TOKENS  floor for the post-trim estimate (8000)
const COMPACT_MAX_TRIES = Number(process.env.FREEPI_PROXY_COMPACT_TRIES || 4);
const COMPACT_KEEP_RATIO = Number(process.env.FREEPI_PROXY_COMPACT_KEEP_RATIO || 0.5);
const COMPACT_MIN_TOKENS = Number(process.env.FREEPI_PROXY_COMPACT_MIN_TOKENS || 8000);
const COMPACT_NOTE =
  "[proxy] Earlier messages were removed automatically so the conversation fits the model's context window. Ask the user to re-send anything essential from the trimmed part if it is still needed.";
const CREDENTIALS_PATH =
  process.env.FREEPI_CREDENTIALS_PATH || path.join(os.homedir(), ".free-pi", "agent", "credentials.json");

const SESSION_ID = `proxy-${crypto.randomUUID()}`;
// Any request with a missing/unknown model name is routed here, so generic
// OpenAI clients (which send e.g. "gpt-4o") work without knowing the catalog.
const DEFAULT_MODEL = process.env.FREEPI_PROXY_DEFAULT_MODEL || "glm-5.3-flash";

// MCP: auto = connect to configured servers, inject their tools, execute
// mcp__* tool calls server-side in an agentic loop; off = ignore MCP config.
const MCP_MODE = process.env.MCP_MODE || "auto";
const MCP_MAX_ITERATIONS = Number(process.env.MCP_MAX_ITERATIONS || 5);

const mcp = new McpManager((msg) => console.log(msg));

// ---------- usage tracking + dashboard ----------

const STARTED_AT = Date.now();
const USAGE_FILE = path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "usage-stats.json");

const emptyUsage = () => ({ requests: 0, prompt_tokens: 0, completion_tokens: 0, reasoning_tokens: 0 });
let usageStats = (() => {
  try {
    const j = JSON.parse(fs.readFileSync(USAGE_FILE, "utf8"));
    if (j && j.today && j.allTime) return j;
  } catch {}
  return { day: new Date().toISOString().slice(0, 10), today: emptyUsage(), byModelToday: {}, allTime: emptyUsage() };
})();

function recordUsage(model, u) {
  try {
    u = u ?? {};
    const day = new Date().toISOString().slice(0, 10);
    if (usageStats.day !== day) {
      usageStats = { day, today: emptyUsage(), byModelToday: {}, allTime: usageStats.allTime ?? emptyUsage() };
    }
    const add = (b) => {
      b.requests += 1;
      b.prompt_tokens += u.prompt_tokens || 0;
      b.completion_tokens += u.completion_tokens || 0;
      b.reasoning_tokens += u.reasoning_tokens || u.completion_tokens_details?.reasoning_tokens || 0;
    };
    add(usageStats.today);
    add(usageStats.allTime);
    const mb = (usageStats.byModelToday[model] ??= emptyUsage());
    add(mb);
    fs.writeFileSync(USAGE_FILE, JSON.stringify(usageStats));
  } catch {}
}

// free-pi account allowance from /me (polled, 60s TTL)
let allowance = { data: null, at: 0 };
async function fetchAllowance() {
  if (allowance.data && Date.now() - allowance.at < 60000) return allowance.data;
  const token = loadToken();
  if (!token) return null;
  try {
    const res = await fetch(`${BASE}/me`, { headers: upstreamHeaders(token) });
    if (res.ok) allowance = { data: await res.json(), at: Date.now() };
  } catch {}
  return allowance.data;
}

// Push the freshly saved credentials.json content to the configured GitHub
// Actions secret via the gh CLI (available locally and on GitHub runners).
// Config: GH_TOKEN (or GITHUB_TOKEN) + GH_REPO, optional GH_SECRET_NAME
// (default FREEPI_CREDENTIALS_JSON).
function syncTokenToGithub() {
  return new Promise((resolve) => {
    const ghToken = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
    const ghRepo = process.env.GH_REPO;
    const secretName = process.env.GH_SECRET_NAME || "FREEPI_CREDENTIALS_JSON";
    if (!ghToken || !ghRepo) {
      return resolve({ attempted: false, ok: false, secret: secretName, detail: "set GH_TOKEN and GH_REPO env to enable auto-sync" });
    }
    let body;
    try {
      body = fs.readFileSync(CREDENTIALS_PATH, "utf8");
      JSON.parse(body); // never upload unparseable content
    } catch (err) {
      return resolve({ attempted: false, ok: false, secret: secretName, detail: `cannot read credentials file: ${err.message}` });
    }
    execFile(
      "gh",
      ["secret", "set", secretName, "-R", ghRepo, "--body", body],
      { timeout: 60000, env: { ...process.env, GH_TOKEN: ghToken, GH_REPO: ghRepo }, windowsHide: true },
      (err, stdout, stderr) => {
        if (err) {
          resolve({ attempted: true, ok: false, secret: secretName, detail: (stderr || err.message).trim().slice(0, 300) });
        } else {
          resolve({ attempted: true, ok: true, secret: secretName, detail: `updated ${ghRepo}/${secretName}` });
        }
      }
    );
  });
}

function tokenFp(t) { return t ? t.length + " chars, ends ..." + t.slice(-8) : "none"; }
function statsSnapshot() {
  const secs = Math.floor((Date.now() - STARTED_AT) / 1000);
  return {
    status: {
      signed_in: Boolean(loadToken()),
      token_fp: tokenFp(loadToken()),
      backend: BASE,
      client_version: getClientVersion(),
      session: SESSION_ID.slice(0, 14) + "…",
      default_model: DEFAULT_MODEL,
      reasoning_effort: REASONING_EFFORT,
      uptime_seconds: secs,
      mcp_servers: [...mcp.servers.entries()].map(([name, { tools }]) => ({ name, tools: tools.length })),
    },
    ad: lastAd && Date.now() - lastAd.at < 600000 ? lastAd.ad : null,
    ad_debug: adDebug,
    allowance: allowance.data && Date.now() - allowance.at < 300000
      ? {
          tier: allowance.data.tier,
          remaining_credits_today: allowance.data.remaining_credits_today,
          allowance_credits: allowance.data.allowance_credits,
          refill_at: allowance.data.refill_at,
          remaining_usd_today: allowance.data.remaining_usd_today,
          cap_usd_today: allowance.data.cap_usd_today,
          spent_usd_today: allowance.data.spent_usd_today,
        }
      : null,
    usage: {
      day: usageStats.day,
      today: usageStats.today,
      by_model_today: usageStats.byModelToday,
      all_time: usageStats.allTime,
    },
  };
}

// ---------- client version (self-healing, never stale) ----------

// The backend rejects chat requests whose x-client-version is below its
// required minimum with HTTP 426 upgrade_required. Version sources, in
// priority order:
//   1. FREEPI_CLIENT_VERSION env           (explicit override)
//   2. ~/.free-pi/agent/last-seen-version  (what the installed CLI last reported)
//   3. client-version-state.json           (last version this proxy resolved)
//   4. backend /client-version "latest"    (authoritative; always >= its "min")
// The backend publishes { min, latest } on GET /client-version and accepts any
// version on that endpoint, so the proxy can always learn the current minimum.
// If the version file is missing or below the backend minimum, the proxy uses
// the backend's latest and records it; a 426 triggers a forced refresh plus
// one retry, so outdated-version errors self-heal instead of surfacing.
const VERSION_FILE = path.join(os.homedir(), ".free-pi", "agent", "last-seen-version");
const PROXY_DIR = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const VERSION_STATE_FILE = path.join(PROXY_DIR, "client-version-state.json");
const VERSION_INFO_TTL_MS = 10 * 60 * 1000;
let currentVersion = null;
let envOverrideBypassed = false; // set when FREEPI_CLIENT_VERSION is below the backend minimum
let versionInfo = { min: null, latest: null, at: 0 };
let versionInfoPromise = null;

function cmpVersions(a, b) {
  const pa = String(a ?? "").split(".").map((n) => parseInt(n, 10) || 0);
  const pb = String(b ?? "").split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) { const d = (pa[i] ?? 0) - (pb[i] ?? 0); if (d) return d < 0 ? -1 : 1; }
  return 0;
}

function fileVersion() {
  try { return fs.readFileSync(VERSION_FILE, "utf8").trim() || null; } catch { return null; }
}

function stateVersion() {
  try { return JSON.parse(fs.readFileSync(VERSION_STATE_FILE, "utf8")).version || null; } catch { return null; }
}

function setClientVersion(version, source) {
  if (!version) return;
  currentVersion = version;
  // Always persist, even when unchanged, so the state file is a reliable
  // fallback if last-seen-version disappears and the backend is unreachable.
  try {
    fs.writeFileSync(VERSION_STATE_FILE, JSON.stringify({ version, source, at: new Date().toISOString() }, null, 2));
  } catch { /* non-fatal: in-memory value still applies */ }
}

function getClientVersion() {
  if (process.env.FREEPI_CLIENT_VERSION && !envOverrideBypassed) return process.env.FREEPI_CLIENT_VERSION;
  return currentVersion ?? "0.0.0";
}

async function fetchVersionInfo({ force = false } = {}) {
  if (!force && versionInfo.at && Date.now() - versionInfo.at < VERSION_INFO_TTL_MS) return versionInfo;
  if (versionInfoPromise) return versionInfoPromise;
  versionInfoPromise = (async () => {
    try {
      const res = await fetch(`${BASE}/client-version`, { headers: { "x-client-version": getClientVersion() } });
      if (res.ok) {
        const body = await res.json().catch(() => null);
        if (body?.min || body?.latest) {
          versionInfo = { min: body.min ?? null, latest: body.latest ?? null, at: Date.now() };
        }
      }
    } catch { /* offline: keep previous info */ }
    versionInfoPromise = null;
    return versionInfo;
  })();
  return versionInfoPromise;
}

// Picks the best version to send and records it. Called at startup and on 426.
async function refreshClientVersion({ force = false } = {}) {
  const info = await fetchVersionInfo({ force });
  const envV = process.env.FREEPI_CLIENT_VERSION;
  if (envV && !envOverrideBypassed && (!info.min || cmpVersions(envV, info.min) >= 0)) {
    setClientVersion(envV, "FREEPI_CLIENT_VERSION");
    return currentVersion;
  }
  if (envV && info.min && cmpVersions(envV, info.min) < 0) {
    // An explicit override below the backend minimum would cause permanent
    // 426s, so bypass it and say so loudly instead of failing forever.
    envOverrideBypassed = true;
    console.warn(`[version] FREEPI_CLIENT_VERSION=${envV} is below the backend minimum ${info.min}; sending ${info.latest ?? info.min} instead. Unset the env var or run npx free-pi-cli@latest to stop seeing this warning.`);
  }
  const fv = fileVersion();
  // The installed CLI's own report is the most truthful sender, but only while
  // it still satisfies the backend minimum; otherwise send the backend latest.
  if (fv && (!info.min || cmpVersions(fv, info.min) >= 0)) {
    setClientVersion(fv, "last-seen-version");
  } else if (info.latest) {
    const why = fv && info.min && cmpVersions(fv, info.min) < 0
      ? `backend latest (version file ${fv} is below backend min ${info.min})`
      : "backend latest (no version file)";
    setClientVersion(info.latest, why);
  } else {
    const sv = stateVersion();
    if (sv) setClientVersion(sv, "state file");
  }
  return currentVersion;
}

async function initClientVersion() {
  const sv = stateVersion();
  if (sv) currentVersion = sv; // provisional, so early requests have something sane
  const fv = fileVersion();
  if (fv) currentVersion = fv; // the real CLI's report wins over the state file
  await refreshClientVersion();
}

// ---------- credentials (cached, refreshed on 401 or file change) ----------

let credCache = { token: null, mtime: 0 };

function loadToken() {
  try {
    const mtime = fs.statSync(CREDENTIALS_PATH).mtimeMs;
    if (!credCache.token || mtime !== credCache.mtime) {
      const json = JSON.parse(fs.readFileSync(CREDENTIALS_PATH, "utf8"));
      credCache = { token: json.token || null, mtime };
    }
  } catch {
    credCache.token = null;
  }
  return credCache.token;
}

function upstreamHeaders(token, turnId) {
  return {
    authorization: `Bearer ${token}`,
    "x-session-id": SESSION_ID,
    "x-client-version": getClientVersion(),
    ...(turnId ? { "x-turn-id": turnId } : {}),
    "content-type": "application/json",
  };
}

// ---------- per-model specs (context window / max output from freepi.ai) ----------

let specsCache = { specs: null, at: 0 };
let specsPromise = null;

// Node's undici can time out where curl succeeds (flaky routes); fall back to
// curl.exe, which ships with Windows.
async function fetchText(url) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
    if (!res.ok) throw new Error(`status ${res.status}`);
    return await res.text();
  } catch (err) {
    const { execFile } = await import("node:child_process");
    const { promisify } = await import("node:util");
    const run = promisify(execFile);
    const { stdout } = await run("curl", ["-s", "-L", "--max-time", "25", url], {
      windowsHide: true, maxBuffer: 10 * 1024 * 1024,
    });
    if (!stdout || stdout.startsWith("<!DOCTYPE html>")) throw new Error(`curl fallback failed for ${url} (${err.message})`);
    return stdout;
  }
}

function parseSpecsMd(md) {
  const num = (s) => Number(String(s).replace(/,/g, ""));
  const ctx = md.match(/Context window:\s*([\d,]+)\s*tokens/i);
  const out = md.match(/Max output:\s*([\d,]+)\s*tokens/i);
  const inputs = md.match(/Inputs:\s*([^\n]+)/i);
  const tools = md.match(/Tool calling:\s*(\w+)/i);
  const effort = md.match(/Reasoning effort:\s*(\w+)/i);
  if (!ctx && !out) return null;
  return {
    contextWindow: ctx ? num(ctx[1]) : CONTEXT_WINDOW,
    maxOutput: out ? num(out[1]) : MAX_OUTPUT_TOKENS,
    inputs: inputs ? inputs[1].split(",").map((s) => s.trim().toLowerCase()).filter(Boolean) : ["text"],
    toolCalling: tools ? tools[1].toLowerCase() === "yes" : true,
    reasoningEffort: effort ? effort[1].toLowerCase() === "yes" : true,
  };
}

async function fetchModelSpecs() {
  if (specsCache.specs && Date.now() - specsCache.at < SPECS_TTL_MS) return specsCache.specs;
  if (specsPromise) return specsPromise;
  specsPromise = (async () => {
    // Start from embedded free-model specs + any previous cache (even stale)
    // so repeated attempts fill gaps instead of replacing — freepi.ai can be
    // flaky/rate-limited. Fetched specs always win over embedded ones.
    let specs = { ...BUILT_IN_SPECS, ...specsCache.specs };
    try {
      const cached = JSON.parse(fs.readFileSync(SPECS_CACHE_FILE, "utf8"));
      if (cached.specs) {
        specs = { ...BUILT_IN_SPECS, ...cached.specs, ...specs };
        if (cached.at && Date.now() - cached.at < SPECS_TTL_MS) {
          specsCache = { specs, at: cached.at };
          return specs;
        }
      }
    } catch {}

    try {
      const index = await fetchText(SPECS_URL);
      const slugs = [...new Set([...index.matchAll(/https:\/\/freepi\.ai\/models\/([a-z0-9.-]+)\.md/g)].map((m) => m[1]))];
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      for (const slug of slugs) {
        // freepi.ai can be flaky / rate-limit rapid fetches; retry with backoff
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            const spec = parseSpecsMd(await fetchText(`https://freepi.ai/models/${slug}.md`));
            if (spec) specs[slug] = spec;
            break;
          } catch {
            if (attempt < 2) await sleep(2000 * (attempt + 1));
          }
        }
        await sleep(500);
      }
      if (Object.keys(specs).length) {
        specsCache = { specs, at: Date.now() };
        try { fs.writeFileSync(SPECS_CACHE_FILE, JSON.stringify(specsCache)); } catch {}
      }
    } catch {
      /* keep whatever we have */
    }
    return specs;
  })();
  try {
    return await specsPromise;
  } finally {
    specsPromise = null;
  }
}

function specFor(modelId, specs) {
  if (!modelId) return null;
  return specs[modelId] ?? specs[modelId.toLowerCase()] ?? null;
}

// ---------- model catalog ----------

let catalogCache = { models: null, at: 0 };
const CATALOG_TTL_MS = 5 * 60 * 1000;

async function fetchCatalog() {
  if (catalogCache.models && Date.now() - catalogCache.at < CATALOG_TTL_MS) return catalogCache.models;
  try {
    const res = await fetch(`${BASE}/client-version`, { headers: { "x-client-version": getClientVersion() } });
    if (!res.ok) throw new Error(`status ${res.status}`);
    const body = await res.json();
    const models = Array.isArray(body?.models) ? body.models : [];
    if (models.length) catalogCache = { models, at: Date.now() };
  } catch {
    /* keep stale cache */
  }
  return catalogCache.models ?? [{ id: DEFAULT_MODEL, name: DEFAULT_MODEL }];
}

function toOpenAIModel(m) {
  return {
    id: m.id,
    object: "model",
    created: 0,
    owned_by: "free-pi",
    // Context/reasoning metadata so context-aware clients pick up the full
    // window instead of guessing a conservative default.
    context_length: CONTEXT_WINDOW,
    max_model_len: CONTEXT_WINDOW,
    context_window: CONTEXT_WINDOW,
    max_output_tokens: MAX_OUTPUT_TOKENS,
    reasoning: true,
    // free-pi extras (harmless for OpenAI clients)
    free_pi_name: m.name,
    free_pi_section: m.section ?? "main",
    free_pi_tier: m.tier ?? "regular",
  };
}

// ---------- request normalization (client quirks -> backend format) ----------

// Fill in maximum context / reasoning settings.
//  - reasoning_effort: inject REASONING_EFFORT when the client sends none
//    (or always, with FREEPI_PROXY_FORCE=1)
//  - max_tokens: default to MAX_DEFAULT_OUTPUT when the client sends none
//    (or raise smaller values to the model max, with FREEPI_PROXY_FORCE=1)
// Message content is never rewritten here: the body the client sends is
// forwarded as-is, except for context-overflow compaction (see
// fetchUpstreamWithCompaction) which only kicks in after the backend says the
// conversation is too long.
function applyMaxima(body) {
  const out = { ...body };
  return fetchModelSpecs().then((specs) => {
    const spec = specFor(out.model, specs);
    const maxOut = spec?.maxOutput ?? MAX_OUTPUT_TOKENS;
    if (REASONING_EFFORT !== "client") {
      if (out.reasoning_effort == null || FORCE_MAXIMA) out.reasoning_effort = REASONING_EFFORT;
    }
    // Never reserve more output than a client actually asked for (or than
    // MAX_DEFAULT_OUTPUT) unless FORCE_MAXIMA is set: prompt + reserved output
    // is what the backend measures against the context window.
    if (out.max_tokens == null) out.max_tokens = Math.min(maxOut, MAX_DEFAULT_OUTPUT);
    else if (FORCE_MAXIMA && out.max_tokens < maxOut) out.max_tokens = maxOut;
    return out;
  });
}

function normalizeBody(body) {
  const out = { ...body };

  // Backend has no developer role: fold it into system.
  if (Array.isArray(out.messages)) {
    const msgs = out.messages.map((m) => {
      if (m?.role === "developer") return { ...m, role: "system" };
      return m;
    });
    // Merge a leading developer/system pair produced by that fold.
    const merged = [];
    for (const m of msgs) {
      const prev = merged[merged.length - 1];
      if (prev && prev.role === "system" && m.role === "system" && prev.content && m.content) {
        merged[merged.length - 1] = { ...prev, content: `${prev.content}\n\n${m.content}` };
      } else {
        merged.push(m);
      }
    }
    out.messages = merged;
  }

  // Newer clients send max_completion_tokens; older backends expect max_tokens.
  if (out.max_tokens == null && out.max_completion_tokens != null) {
    out.max_tokens = out.max_completion_tokens;
  }

  // Legacy function-calling -> tools, for older OpenAI clients.
  if (!out.tools && Array.isArray(out.functions) && out.functions.length) {
    out.tools = out.functions.map((f) => ({ type: "function", function: f }));
    if (out.function_call === "none" || out.function_call === "auto") out.tool_choice = out.function_call;
    else if (out.function_call && typeof out.function_call === "object" && out.function_call.name) {
      out.tool_choice = { type: "function", function: { name: out.function_call.name } };
    }
    delete out.functions;
    delete out.function_call;
  }

  // Client-only fields must not reach the backend.
  delete out.mcp;

  // Image content normalization across client dialects.
  if (Array.isArray(out.messages)) {
    for (const m of out.messages) {
      if (!Array.isArray(m?.content)) continue;
      m.content = m.content.map((part) => {
        if (!part || typeof part !== "object") return part;
        // image_url given as a plain string -> wrap it
        if (part.type === "image_url" && typeof part.image_url === "string") {
          return { ...part, image_url: { url: part.image_url } };
        }
        // Anthropic-style image -> OpenAI image_url data URL
        if (part.type === "image" && part.source?.type === "base64" && part.source?.data) {
          return {
            type: "image_url",
            image_url: { url: `data:${part.source.media_type || "image/png"};base64,${part.source.data}` },
          };
        }
        return part;
      });
    }
  }

  return out;
}

// ---------- context-overflow recovery ----------

// Rough token estimate. It only decides which turns to drop; the backend stays
// the authority, so anything we think is fine is still sent as-is first.
//
// Media cost model, measured against the real backend: images are billed by
// PIXEL DIMENSIONS, not by byte size. Two 768x768 images whose base64 bodies
// were 570 KB vs 57 KB both reported exactly 799 prompt_tokens, and observed
// cost tracks ~px/763 + 35 (64x64 -> 31, 512x512 -> 376, 1024x1024 -> 1384,
// 2048x2048 -> 5491). So a 1-4 MB screenshot is only a few thousand tokens.
// Charging the base64 length instead (an earlier bug) made every attachment
// look like an overflow -- the opposite failure. Bill pixels.
const MEDIA_TOKENS_PER_PIXEL = 1 / 763;
const MEDIA_FIXED_TOKENS = 35;
const MEDIA_UNKNOWN_TOKENS = 800; // no parseable dimensions

// Read WxH from PNG/JPEG/GIF/WebP headers in a data URL -- no image decoding.
function imagePixelCount(buf) {
  try {
    if (buf.length > 24 && buf.readUInt32BE(0) === 0x89504e47) {
      return buf.readUInt32BE(16) * buf.readUInt32BE(20); // PNG IHDR
    }
    if (buf.length > 10 && buf.toString("ascii", 0, 3) === "GIF") {
      return buf.readUInt16LE(6) * buf.readUInt16LE(8);
    }
    if (buf.length > 30 && buf.toString("ascii", 0, 4) === "RIFF" &&
        buf.toString("ascii", 8, 12) === "WEBP") {
      const fmt = buf.toString("ascii", 12, 16);
      if (fmt === "VP8X") return (buf.readUIntLE(24, 3) + 1) * (buf.readUIntLE(27, 3) + 1);
      if (fmt === "VP8 ") return (buf.readUInt16LE(26) & 0x3fff) * (buf.readUInt16LE(28) & 0x3fff);
      if (fmt === "VP8L") {
        const b = buf.readUInt32LE(21);
        return ((b & 0x3fff) + 1) * (((b >> 14) & 0x3fff) + 1);
      }
    }
    if (buf.length > 4 && buf[0] === 0xff && buf[1] === 0xd8) { // JPEG SOFn
      let i = 2;
      while (i + 9 < buf.length) {
        if (buf[i] !== 0xff) { i++; continue; }
        const marker = buf[i + 1];
        if (marker >= 0xc0 && marker <= 0xcf &&
            marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
          return buf.readUInt16BE(i + 7) * buf.readUInt16BE(i + 5);
        }
        i += 2 + buf.readUInt16BE(i + 2);
      }
    }
  } catch { /* truncated header: fall through */ }
  return 0;
}

// The backend rejects a request body of roughly 0.94 MB and up with the
// misleading "This conversation is too long" 413 -- measured: a 200x200 image
// (79 prompt tokens) succeeded at a 0.951 MB body and failed at 0.959 MB, while
// identical pixels reached 413 once the base64 padding crossed ~1 MB. So the cap
// is BYTE SIZE of the request, not pixels and not tokens. Compaction can never
// fix this (the attachment survives every trim), so detect it up front.
const REQUEST_BODY_LIMIT_BYTES = 940_000;

function mediaTokens(url) {
  if (typeof url !== "string" || !url) return MEDIA_UNKNOWN_TOKENS;
  if (!url.startsWith("data:")) {
    // Remote URL: the backend fetches it and bills its real dimensions, which
    // we cannot see. Assume a moderately large image rather than zero.
    return Math.ceil(1024 * 1024 * MEDIA_TOKENS_PER_PIXEL) + MEDIA_FIXED_TOKENS;
  }
  const comma = url.indexOf(",");
  if (comma < 0) return MEDIA_UNKNOWN_TOKENS;
  const meta = url.slice(5, comma);
  let px = 0;
  if (/;base64/i.test(meta)) {
    try { px = imagePixelCount(Buffer.from(url.slice(comma + 1), "base64")); } catch { px = 0; }
  }
  if (!px) return MEDIA_UNKNOWN_TOKENS;
  return Math.ceil(px * MEDIA_TOKENS_PER_PIXEL) + MEDIA_FIXED_TOKENS;
}

// Largest inline attachment in a body, in bytes of base64 payload.
function largestMediaBytes(body) {
  let worst = 0, worstUrl = "";
  const walk = (v) => {
    if (v == null) return;
    if (Array.isArray(v)) { for (const x of v) walk(x); return; }
    if (typeof v !== "object") return;
    const t = v.type;
    if (t === "image_url" || t === "image" || t === "input_image" || t === "video_url") {
      const raw = v.image_url ?? v.source ?? v;
      const url = typeof raw === "string" ? raw : (typeof raw?.url === "string" ? raw.url : "");
      const data = typeof raw?.data === "string" ? raw.data : "";
      const len = Math.max(url.length, data.length);
      if (len > worst) { worst = len; worstUrl = url || "(inline base64)"; }
      return;
    }
    for (const x of Object.values(v)) walk(x);
  };
  walk(body?.messages);
  return { bytes: worst, url: worstUrl };
}


// ---------- attachment downscaling ----------
//
// The backend rejects a request body over ~0.94 MB (see
// REQUEST_BODY_LIMIT_BYTES), and base64 costs 4/3 of the file size, so inline
// attachments must stay under roughly 700 KB. Rather than reject the user's
// 1-4 MB screenshot, shrink it here: decode PNG in pure JS (node:zlib only, no
// native dependency so the workflow needs no npm install), box-filter down to a
// sane pixel budget, and re-encode as PNG. Images we cannot decode (JPEG/WebP)
// fall back to the size-limit error.
const DOWNSCALE_ENABLED = process.env.FREEPI_PROXY_DOWNSCALE !== "0";
// Vision cost tracks px/763; ~1.15 MP keeps quality high while costing ~1500
// tokens. Long side is capped separately so panoramas stay readable.
const DOWNSCALE_MAX_PIXELS = Number(process.env.FREEPI_PROXY_MAX_IMAGE_PIXELS || 1150000);
const DOWNSCALE_MAX_SIDE = Number(process.env.FREEPI_PROXY_MAX_IMAGE_SIDE || 1800);
// Target base64 bytes for a single attachment (leaves room for the rest of the
// body under the ~0.94 MB cap).
const DOWNSCALE_TARGET_B64 = Number(process.env.FREEPI_PROXY_IMAGE_BYTES || 620000);

const PNG_SIG = 0x89504e47;
const CRC_TBL = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TBL[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}

// Decode a non-interlaced 8-bit PNG (grey/RGB/RGBA) to raw pixels.
function decodePng(buf) {
  if (buf.length < 8 || buf.readUInt32BE(0) !== PNG_SIG) return null;
  let i = 8, w = 0, h = 0, depth = 0, ct = 0, inter = 0;
  const idat = [];
  while (i + 8 <= buf.length) {
    const len = buf.readUInt32BE(i);
    const type = buf.toString("ascii", i + 4, i + 8);
    const data = buf.subarray(i + 8, i + 8 + len);
    if (type === "IHDR") {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4);
      depth = data[8]; ct = data[9]; inter = data[12];
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    i += 12 + len;
  }
  if (!w || !h || depth !== 8 || inter) return null;
  const chan = ct === 2 ? 3 : ct === 6 ? 4 : ct === 0 ? 1 : null;
  if (!chan) return null;
  let raw;
  try { raw = zlib.inflateSync(Buffer.concat(idat)); } catch { return null; }
  const stride = w * chan;
  if (raw.length < (stride + 1) * h) return null;
  const out = Buffer.alloc(h * stride);
  let pos = 0;
  for (let y = 0; y < h; y++) {
    const ft = raw[pos++];
    const line = raw.subarray(pos, pos + stride); pos += stride;
    const start = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= chan ? out[start + x - chan] : 0;
      const b = y > 0 ? out[start - stride + x] : 0;
      const c = (y > 0 && x >= chan) ? out[start - stride + x - chan] : 0;
      let v = line[x];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      out[start + x] = v & 255;
    }
  }
  return { w, h, chan, data: out };
}

// Box-filter resize to (nw, nh); averaging avoids the aliasing that makes
// downscaled screenshots unreadable to a vision model.
function resizeBox(img, nw, nh) {
  const { w, h, chan, data } = img;
  const out = Buffer.alloc(nw * nh * 3);
  const xr = w / nw, yr = h / nh;
  for (let y = 0; y < nh; y++) {
    const y0 = Math.floor(y * yr), y1 = Math.max(y0 + 1, Math.floor((y + 1) * yr));
    for (let x = 0; x < nw; x++) {
      const x0 = Math.floor(x * xr), x1 = Math.max(x0 + 1, Math.floor((x + 1) * xr));
      let r = 0, g = 0, b = 0, n = 0;
      for (let sy = y0; sy < y1 && sy < h; sy++) {
        const row = sy * w * chan;
        for (let sx = x0; sx < x1 && sx < w; sx++) {
          const o = row + sx * chan;
          if (chan === 1) { const v = data[o]; r += v; g += v; b += v; }
          else { r += data[o]; g += data[o + 1]; b += data[o + 2]; }
          n++;
        }
      }
      const q = (y * nw + x) * 3;
      out[q] = (r / n) | 0; out[q + 1] = (g / n) | 0; out[q + 2] = (b / n) | 0;
    }
  }
  return out;
}

function encodePng(rgb, w, h) {
  const stride = w * 3;
  const raw = Buffer.alloc((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0; // filter: none
    rgb.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(raw, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
}

// Shrink one data URL if it is over the size limit or the pixel budget.
// Returns { url, changed, beforeB64, afterB64, beforePx, afterPx } or null when
// the image cannot be handled (non-PNG, or still too big after resizing).
function downscaleDataUrl(url) {
  if (!DOWNSCALE_ENABLED) return null;
  if (typeof url !== "string" || !url.startsWith("data:")) return null;
  const comma = url.indexOf(",");
  if (comma < 0) return null;
  const meta = url.slice(5, comma);
  if (!/;base64/i.test(meta)) return null;
  const b64 = url.slice(comma + 1);
  let buf;
  try { buf = Buffer.from(b64, "base64"); } catch { return null; }
  const img = decodePng(buf);
  if (!img) return null; // JPEG/WebP/etc: cannot shrink without a codec

  const px = img.w * img.h;
  const longSide = Math.max(img.w, img.h);
  const overBytes = b64.length > DOWNSCALE_TARGET_B64;
  const overPixels = px > DOWNSCALE_MAX_PIXELS || longSide > DOWNSCALE_MAX_SIDE;
  if (!overBytes && !overPixels) return null; // already fine, don't touch it

  let scale = Math.min(1, Math.sqrt(DOWNSCALE_MAX_PIXELS / px));
  if (longSide * scale > DOWNSCALE_MAX_SIDE) scale = DOWNSCALE_MAX_SIDE / longSide;

  // Shrink until the encoded payload fits; each step also drops pixels.
  let nw = Math.max(1, Math.round(img.w * scale));
  let nh = Math.max(1, Math.round(img.h * scale));
  let outBuf = null, outB64 = "";
  for (let attempt = 0; attempt < 6; attempt++) {
    const rgb = resizeBox(img, nw, nh);
    outBuf = encodePng(rgb, nw, nh);
    outB64 = outBuf.toString("base64");
    if (outB64.length <= DOWNSCALE_TARGET_B64) break;
    if (nw <= 32 || nh <= 32) break;
    const f = Math.sqrt(DOWNSCALE_TARGET_B64 / outB64.length) * 0.92;
    nw = Math.max(1, Math.round(nw * Math.min(f, 0.9)));
    nh = Math.max(1, Math.round(nh * Math.min(f, 0.9)));
  }
  if (!outB64 || outB64.length >= b64.length) return null; // no gain

  return {
    url: `data:image/png;base64,${outB64}`,
    changed: true,
    beforeB64: b64.length,
    afterB64: outB64.length,
    beforePx: px,
    afterPx: nw * nh,
  };
}

// Walk the body and downscale every inline image that is over budget.
function downscaleBodyImages(body) {
  const report = [];
  const walk = (v) => {
    if (v == null) return;
    if (Array.isArray(v)) { for (const x of v) walk(x); return; }
    if (typeof v !== "object") return;
    const t = v.type;
    if (t === "image_url" || t === "image" || t === "input_image") {
      const raw = v.image_url ?? v.source ?? v;
      const isStr = typeof raw === "string";
      const cur = isStr ? raw : (typeof raw?.url === "string" ? raw.url : "");
      const inline = typeof raw?.data === "string" ? `data:${raw.media_type || "image/png"};base64,${raw.data}` : "";
      const r = downscaleDataUrl(cur || inline);
      if (r) {
        if (isStr) v.image_url = r.url;
        else if (typeof v.image_url === "object" && v.image_url) v.image_url.url = r.url;
        // Anthropic-style: keep the part shape, replace the payload.
        else if (v.source && typeof v.source === "object") {
          v.source = { ...v.source, type: "base64", media_type: "image/png", data: r.url.split(",")[1] };
        }
        report.push(r);
      }
      return;
    }
    for (const x of Object.values(v)) walk(x);
  };
  walk(body?.messages);
  return report;
}

function estimateTokens(value) {
  if (value == null) return 0;
  if (typeof value === "string") return Math.ceil(value.length / 4);
  if (typeof value !== "object") return 1;
  if (Array.isArray(value)) return value.reduce((n, v) => n + estimateTokens(v), 0);
  if (
    value.type === "image_url" || value.type === "image" ||
    value.type === "input_image" || value.type === "video_url"
  ) {
    const raw = value.image_url ?? value.source ?? value;
    const url =
      typeof raw === "string" ? raw :
      typeof raw?.url === "string" ? raw.url : "";
    // Anthropic-style inline base64 keeps the bytes in source.data.
    const data = typeof raw?.data === "string"
      ? `data:${raw.media_type || "image/png"};base64,${raw.data}`
      : "";
    return mediaTokens(url || data);
  }
  let n = 0;
  for (const v of Object.values(value)) n += estimateTokens(v);
  return n;
}

const MESSAGE_OVERHEAD_TOKENS = 4;
const estimateBodyTokens = (body) =>
  estimateTokens(body?.messages) + estimateTokens(body?.tools) +
  estimateTokens(body?.response_format) + 16;

// Messages that mean "the prompt no longer fits the model window" — the backend
// answers with HTTP 413 and code `context_too_long` ("This conversation is too
// long — run /compact or start a new session with /new"). Mirrors free-pi-cli's
// own overflow detection so we don't retry on quota/rate-limit errors.
const OVERFLOW_PATTERNS = [
  /context_too_long/i,
  /context[_ ]length[_ ]exceeded/i,
  /prompt (?:is )?too long/i,
  /prompt exceeds max length/i,
  /request_too_large/i,
  /input is too long/i,
  /exceeds (?:the )?(?:model'?s )?maximum context length/i,
  /exceeds the context window/i,
  /input token count.*exceeds the maximum/i,
  /maximum context length is \d+ tokens/i,
  /too many tokens/i,
  /token limit exceeded/i,
  /conversation is too long/i,
];
const NON_OVERFLOW_PATTERNS = [/rate limit/i, /too many requests/i, /model_at_capacity/i];

function isContextOverflow(status, text) {
  const msg = String(text || "");
  if (NON_OVERFLOW_PATTERNS.some((p) => p.test(msg))) return false;
  if (OVERFLOW_PATTERNS.some((p) => p.test(msg))) return true;
  return status === 413; // bodyless 413 still means "payload too large"
}

// Drop the oldest turns until the request estimates <= budgetTokens. The system
// prompt and the newest turn are always preserved and tool-call pairing is kept
// valid by starting the retained tail on a user message.
function compactBody(body, budgetTokens) {
  const msgs = Array.isArray(body?.messages) ? body.messages : null;
  if (!msgs || !msgs.length) return null;

  const head = [];
  let i = 0;
  while (i < msgs.length && (msgs[i]?.role === "system" || msgs[i]?.role === "developer")) head.push(msgs[i++]);
  const tail = msgs.slice(i);
  if (!tail.length) return null;

  let room = budgetTokens - estimateTokens(body.tools) - estimateTokens(body.response_format) - estimateTokens(head);
  const kept = [];
  for (let j = tail.length - 1; j >= 0; j--) {
    const cost = estimateTokens(tail[j]) + MESSAGE_OVERHEAD_TOKENS;
    if (kept.length && room - cost < 0) break;
    kept.unshift(tail[j]);
    room -= cost;
  }
  // Start on a user turn: never leave orphaned tool results behind.
  while (kept.length > 1 && kept[0]?.role !== "user") kept.shift();
  if (!kept.length) kept.push(tail[tail.length - 1]);

  const dropped = msgs.length - head.length - kept.length;
  if (dropped <= 0) return null;
  return { ...body, messages: [...head, { role: "system", content: COMPACT_NOTE }, ...kept] };
}

// POST to the backend; on 413 / context-too-long shrink the history and retry.
// Returns { resp, text, body, compacted, overflow }; `text` is set only when the
// final response is not ok (its body has already been read).
async function fetchUpstreamWithCompaction(token, upstreamBody, turnId) {
  let body = upstreamBody;
  let compacted = false;
  for (let attempt = 0; ; attempt++) {
    const resp = await upstreamCall(token, body, turnId);
    if (resp.ok) return { resp, text: null, body, compacted };
    const text = await resp.text().catch(() => "");
    const overflow = isContextOverflow(resp.status, text);
    if (!overflow) return { resp, text, body, compacted };
    if (attempt >= COMPACT_MAX_TRIES) return { resp, text, body, compacted, overflow: true };

    const before = estimateBodyTokens(body);
    // Shrink to COMPACT_KEEP_RATIO of the current size, but never *above* it —
    // the floor only applies while it leaves room to shrink.
    const budget = Math.max(
      Math.min(COMPACT_MIN_TOKENS, Math.floor(before * 0.5)),
      Math.floor(before * COMPACT_KEEP_RATIO)
    );
    const next = compactBody(body, budget);
    const after = next ? estimateBodyTokens(next) : before;
    if (!next || after >= before) {
      return { resp, text, body, compacted, overflow: true };
    }
    // If trimming can only shave a sliver, the request is dominated by content
    // that survives every compaction (a very large attachment). Retrying just
    // burns the allowance and ends on the same 413, so report the real cause.
    if (before - after < Math.max(2000, before * 0.02)) {
      return { resp, text, body, compacted, overflow: true, mediaBound: true };
    }
    let code = "";
    try {
      const parsed = JSON.parse(text);
      code = parsed?.error?.code ?? parsed?.code ?? "";
    } catch {}
    console.log(
      `[compact] backend refused (${resp.status}${code ? " " + code : ""}): ` +
      `~${before} tokens -> ~${estimateBodyTokens(next)} ` +
      `(${body.messages.length - next.messages.length} older message(s) dropped)`
    );
    compacted = true;
    body = next;
  }
}

// ---------- MCP agentic loop ----------

function completionToSSE(completion) {
  const choice = completion.choices?.[0] ?? {};
  const msg = choice.message ?? {};
  const frame = (delta, finish = null, extra = {}) =>
    `data: ${JSON.stringify({ id: completion.id, object: "chat.completion.chunk", created: completion.created, model: completion.model, choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}`;
  const events = [frame({ role: msg.role ?? "assistant", ...(msg.content ? { content: msg.content } : {}) })];
  for (const [i, tc] of (msg.tool_calls ?? []).entries()) {
    events.push(frame({ tool_calls: [{ index: i, id: tc.id, type: "function", function: { name: tc.function?.name, arguments: tc.function?.arguments ?? "" } }] }));
  }
  events.push(frame({}, choice.finish_reason ?? "stop"));
  if (completion.usage) events.push(`data: ${JSON.stringify({ id: completion.id, object: "chat.completion.chunk", created: completion.created, model: completion.model, choices: [], usage: completion.usage })}`);
  events.push("data: [DONE]");
  return events.join("\n\n") + "\n\n";
}

function isMcpTool(name) {
  return typeof name === "string" && name.startsWith("mcp__");
}

// Run upstream chat completions against the backend with explicit body.
async function upstreamCall(token, upstreamBody, turnId) {
  return fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: upstreamHeaders(token, turnId),
    body: JSON.stringify(upstreamBody),
  });
}

// When MCP tools are active, run a server-side agentic loop: execute
// mcp__* tool calls ourselves and feed results back until the model answers
// or calls a client-owned tool (which is returned to the client untouched).
async function runWithMcp(token, upstreamBody, wantsStream, res, turnId) {
  let messages = [...upstreamBody.messages];
  let final = null;
  let compacted = false;
  for (let iter = 0; iter < MCP_MAX_ITERATIONS; iter++) {
    let call;
    try {
      call = await fetchUpstreamWithCompaction(token, { ...upstreamBody, messages, stream: false }, turnId);
    } catch (err) {
      return json(res, 502, openAIError(`Cannot reach free-pi backend: ${err.message}`, "api_error"));
    }
    // The proxy may have trimmed older turns to recover from a 413; keep looping
    // on the compacted history.
    messages = call.body.messages ?? messages;
    compacted = compacted || call.compacted;
    if (!call.resp.ok) {
      return json(res, call.overflow ? 413 : call.resp.status === 401 || call.resp.status === 429 ? call.resp.status : 502,
        openAIError(`free-pi backend ${call.resp.status}: ${(call.text ?? "").slice(0, 1000)}`, "api_error"));
    }
    const completion = await call.resp.json();
    recordUsage(completion.model ?? upstreamBody.model, completion.usage);
    const msg = completion.choices?.[0]?.message ?? {};
    const calls = msg.tool_calls ?? [];
    const mcpCalls = calls.filter((c) => isMcpTool(c.function?.name));
    if (!mcpCalls.length || calls.length !== mcpCalls.length) {
      // No MCP calls (or client-owned tools present) -> return as-is.
      final = completion;
      break;
    }
    const toolMsgs = [];
    for (const c of mcpCalls) {
      let result;
      try {
        result = await mcp.callTool(c.function.name, c.function.arguments);
      } catch (err) {
        result = `MCP tool error: ${err.message}`;
      }
      toolMsgs.push({ role: "tool", tool_call_id: c.id, content: String(result).slice(0, 100000) });
    }
    messages.push(msg, ...toolMsgs);
    if (iter === MCP_MAX_ITERATIONS - 1) { final = completion; }
  }
  if (!final) return json(res, 502, openAIError("MCP loop produced no completion", "api_error"));
  if (wantsStream) {
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      ...(compacted ? { "x-freepi-compacted": "1" } : {}),
    });
    return res.end(completionToSSE(final));
  }
  return json(res, 200, final);
}

// ---------- dashboard ----------

const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>free-pi proxy dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px; } .sub { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .card h3 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #8b949e; }
  .big { font-size: 26px; font-weight: 600; } .ok { color: #3fb950; } .bad { color: #f85149; }
  .kv { display: flex; justify-content: space-between; gap: 12px; padding: 2px 0; font-size: 13px; }
  .kv span:first-child { color: #8b949e; }
  .bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin: 8px 0; }
  .bar > div { height: 100%; background: #3fb950; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: 500; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .muted { color: #8b949e; } .pill { display:inline-block; background:#21262d; border-radius:10px; padding:1px 8px; margin:2px; font-size:12px; }
</style></head><body>
<h1>free-pi proxy</h1><div class="sub" id="updated">loading…</div>
<div class="grid">
  <div class="card"><h3>Status</h3><div id="status"></div></div>
  <div class="card"><h3>free-pi allowance today</h3><div id="allowance"></div></div>
  <div class="card"><h3>Tokens today</h3><div id="today"></div></div>
<div class="card"><h3>free-pi token</h3><div id="token"></div></div>
<div class="card"><h3>Sponsor</h3><div id="adcard" class="muted" style="font-size:13px">loading…</div></div>
</div>
<div class="card"><h3>Per model (today)</h3><div id="models"></div></div>
<p class="muted">Auto-refreshes every 5s. All-time totals included in /stats.</p>
<script>
function fmt(n) { return (n ?? 0).toLocaleString(); }
function hdrs() {
  var k = localStorage.getItem("freepi_proxy_key");
  return k ? { "Authorization": "Bearer " + k } : {};
}
function esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; }
function render(s) {
  var st = s.status || {};
  document.getElementById("status").innerHTML =
    "<div class=\\"kv\\"><span>Signed in</span><span class=\\"" + (st.signed_in ? "ok" : "bad") + "\\">" + (st.signed_in ? "yes" : "NO") + "</span></div>" +
    "<div class=\\"kv\\"><span>Uptime</span><span>" + Math.floor((st.uptime_seconds||0)/3600) + "h " + Math.floor(((st.uptime_seconds||0)%3600)/60) + "m</span></div>" +
    "<div class=\\"kv\\"><span>Default model</span><span>" + esc(st.default_model) + "</span></div>" +
    "<div class=\\"kv\\"><span>Reasoning effort</span><span>" + esc(st.reasoning_effort) + "</span></div>" +
    "<div class=\\"kv\\"><span>Backend</span><span>" + esc(st.backend) + "</span></div>" +
    "<div class=\\"kv\\"><span>MCP</span><span>" + ((st.mcp_servers||[]).map(function(m){return esc(m.name)+" ("+m.tools+")";}).join(", ") || "none") + "</span></div>" +
    "<div class=\\"kv\\"><span>Session</span><span class=\\"muted\\">" + esc(st.session) + "</span></div>";
  var a = s.allowance;
  document.getElementById("allowance").innerHTML = a ?
    "<div class=\\"big\\">" + fmt(a.remaining_credits_today) + " <span class=\\"muted\\" style=\\"font-size:14px\\">/ " + fmt(a.allowance_credits) + " credits</span></div>" +
    "<div class=\\"bar\\"><div style=\\"width:" + Math.max(0, Math.min(100, 100 * (a.remaining_credits_today||0) / Math.max(1, a.allowance_credits||1))) + "%\\"></div></div>" +
    "<div class=\\"kv\\"><span>Tier</span><span>" + esc(a.tier) + "</span></div>" +
    "<div class=\\"kv\\"><span>Refill</span><span>" + esc(a.refill_at) + "</span></div>" +
    "<div class=\\"kv\\"><span>USD today</span><span>$" + esc(a.spent_usd_today) + " <span class=\\"muted\\">cap $" + esc(a.cap_usd_today) + "</span></span></div>"
    : "<span class=\\"muted\\">not available (not signed in or /me unreachable)</span>";
  var u = s.usage || {};
  var t = u.today || {};
  document.getElementById("today").innerHTML =
    "<div class=\\"big\\">" + fmt((t.prompt_tokens||0) + (t.completion_tokens||0)) + " <span class=\\"muted\\" style=\\"font-size:14px\\">tokens</span></div>" +
    "<div class=\\"kv\\"><span>Requests</span><span>" + fmt(t.requests) + "</span></div>" +
    "<div class=\\"kv\\"><span>Prompt</span><span>" + fmt(t.prompt_tokens) + "</span></div>" +
    "<div class=\\"kv\\"><span>Completion</span><span>" + fmt(t.completion_tokens) + "</span></div>" +
    "<div class=\\"kv\\"><span>Reasoning</span><span>" + fmt(t.reasoning_tokens) + "</span></div>" +
    "<div class=\\"kv\\"><span>All time</span><span>" + fmt((u.all_time||{}).prompt_tokens) + " / " + fmt((u.all_time||{}).completion_tokens) + "</span></div>";
  var rows = "";
  var bm = u.by_model_today || {};
  Object.keys(bm).sort().forEach(function(mname) {
    var m = bm[mname];
    rows += "<tr><td>" + esc(mname) + "</td><td class=\\"num\\">" + fmt(m.requests) + "</td><td class=\\"num\\">" + fmt(m.prompt_tokens) + "</td><td class=\\"num\\">" + fmt(m.completion_tokens) + "</td><td class=\\"num\\">" + fmt(m.reasoning_tokens) + "</td></tr>";
  });
  document.getElementById("models").innerHTML = rows ?
    "<table><tr><th>model</th><th class=\\"num\\">req</th><th class=\\"num\\">prompt</th><th class=\\"num\\">completion</th><th class=\\"num\\">reasoning</th></tr>" + rows + "</table>"
    : "<span class=\\"muted\\">no requests yet today</span>";
  document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString() + " — day " + (u.day || "?");
  renderToken(s);
  renderAd(s.ad);
}
var tokenBuilt = false;
function renderToken(s) {
  var signedIn = s.status && s.status.signed_in;
  if (tokenBuilt) {
    var line = document.getElementById("toksignin");
    if (line) line.innerHTML = "<span>Signed in</span><span class=\\"" + (signedIn ? "ok" : "bad") + "\\">" + (signedIn ? "yes" : "NO") + "</span>";
    var fpEl = document.getElementById("tokfp");
    if (fpEl) fpEl.textContent = "In use: " + ((s.status && s.status.token_fp) || "?");
    return; // built once - never rebuild, so auto-refresh cannot wipe typed input
  }
  document.getElementById("token").innerHTML =
    "<div id=\\"toksignin\\" class=\\"kv\\"><span>Signed in</span><span class=\\"" + (signedIn ? "ok" : "bad") + "\\">" + (signedIn ? "yes" : "NO") + "</span></div>" +
    "<div id=\\"tokacct\\" class=\\"muted\\" style=\\"font-size:12px;margin:4px 0\\"></div>" +
    "<div id=\\"tokfp\\" class=\\"muted\\" style=\\"font-size:12px;margin:2px 0\\"></div>" +
    "<div style=\\"margin:8px 0 4px\\" class=\\"muted\\">Proxy API key (saved in this browser):</div>" +
    "<input id=\\"pkey\\" type=\\"text\\" placeholder=\\"PROXY_API_KEY (leave empty if none set)\\" style=\\"width:100%;box-sizing:border-box;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:6px;font-family:monospace\\">" +
    "<span id=\\"pkeystatus\\" class=\\"muted\\" style=\\"font-size:12px\\"></span>" +
    "<div style=\\"margin:8px 0 4px\\" class=\\"muted\\">Paste credentials.json contents (or just the token string) from any machine:</div>" +
    "<textarea id=\\"tok\\" rows=\\"3\\" placeholder=\\"{&quot;token&quot;:&quot;...&quot;}\\" style=\\"width:100%;box-sizing:border-box;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:6px;font-family:monospace;font-size:11px\\"></textarea>" +
    "<button id=\\"check\\" style=\\"margin-top:8px;background:#30363d;color:#e6edf3;border:1px solid #8b949e;border-radius:6px;padding:6px 14px;cursor:pointer;margin-right:6px\\">Check</button>" +
    "<button id=\\"save\\" style=\\"margin-top:8px;background:#238636;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer\\">Save token</button>" +
    "<span id=\\"tokmsg\\" class=\\"muted\\" style=\\"margin-left:10px;font-size:12px\\"></span>";
  var keyInput = document.getElementById("pkey");
  keyInput.value = localStorage.getItem("freepi_proxy_key") || "";
  keyInput.oninput = function() { localStorage.setItem("freepi_proxy_key", keyInput.value.trim()); };
  var probeKey = function() {
    fetch("/token", { method: "POST", headers: Object.assign({ "content-type": "application/json" }, hdrs()), body: "{}" })
      .then(function(r) {
        var el = document.getElementById("pkeystatus");
        if (!el) return;
        if (r.status === 400) el.innerHTML = "<span class=\\"ok\\">\u2713 proxy key accepted</span>";
        else if (r.status === 401) el.innerHTML = "<span class=\\"bad\\">\u2717 key missing or wrong \u2014 enter the proxy's PROXY_API_KEY above</span>";
      }).catch(function() {});
  };
  probeKey();
  keyInput.addEventListener("input", function() { clearTimeout(probeKey._t); probeKey._t = setTimeout(probeKey, 400); });
  var acctLine = document.getElementById("tokacct");
  var fmtAcct = function(a) {
    if (!a) return "";
    return "tier " + esc(a.tier) + " \u2014 " + fmt(a.remaining_credits_today) + "/" + fmt(a.allowance_credits) + " credits left, refill " + esc(a.refill_at);
  };
  var post = function(checkOnly) {
    var msg = document.getElementById("tokmsg");
    var val = document.getElementById("tok").value.trim();
    if (!val) { msg.textContent = "paste a token first"; return; }
    msg.textContent = checkOnly ? "checking..." : "validating...";
    fetch("/token", { method: "POST", headers: Object.assign({ "content-type": "application/json" }, hdrs()), body: JSON.stringify({ token: val, check_only: checkOnly }) })
      .then(function(r) { return r.json(); })
      .then(function(j) {
        if (j.error) { msg.textContent = "\u2717 " + j.error.message; return; }
        acctLine.textContent = fmtAcct(j.account);
        if (checkOnly) { msg.textContent = "\u2713 valid (not saved)"; return; }
        var sync = j.github_sync || {};
        var syncTxt = sync.attempted
          ? (sync.ok ? " \u00b7 GitHub secret " + sync.secret + " updated \u2713" : " \u00b7 GitHub sync FAILED: " + sync.detail)
          : " \u00b7 GitHub sync not configured";
        msg.textContent = "\u2713 saved & validated" + syncTxt;
        document.getElementById("tok").value = "";
      })
      .catch(function(e) { msg.textContent = "\u2717 " + e.message; });
  };
  document.getElementById("check").onclick = function() { post(true); };
  document.getElementById("save").onclick = function() { post(false); };
  tokenBuilt = true;
}
var shownAdId = null;
function ea(s) { return esc(s).replace(/"/g, "&quot;"); }
function renderAd(ad) {
  var el = document.getElementById("adcard");
  if (!el) return;
  if (!ad) { if (shownAdId !== null) { shownAdId = null; el.innerHTML = "<span class=\\"muted\\">no ad yet — ads appear here as the proxy serves requests</span>"; } return; }
  if (ad.ad_id === shownAdId) return;
  shownAdId = ad.ad_id;
  var c = ad.creative || {};
  el.innerHTML =
    "<div style=\\"font-weight:600\\">" + esc(c.brand_name || c.headline || "") + "</div>" +
    "<div style=\\"margin:2px 0\\">" + esc(c.headline || "") + "</div>" +
    "<div class=\\"muted\\" style=\\"margin:4px 0\\">" + esc(c.body || "") + "</div>" +
    "<a href=\\"" + ea(ad.click_url) + "\\" target=\\"_blank\\" rel=\\"noopener\\" style=\\"color:#58a6ff;background:#21262d;border-radius:6px;padding:4px 10px;display:inline-block;text-decoration:none\\">" + esc(c.cta || "Learn more") + "</a>" +
    "<div class=\\"muted\\" style=\\"margin-top:4px;font-size:11px\\">shown in exchange for proxy usage, like the CLI</div>";
}
function refresh() { fetch("/stats", { headers: hdrs() }).then(function(r){return r.json();}).then(render).catch(function(){}); }
refresh(); setInterval(refresh, 5000);
</script></body></html>`;

// ---------- ads (shown in the dashboard; the browser fires impressions on real display) ----------

// Ads are fetched only when an open dashboard asks for one, and the impression
// pixel is loaded by the user's browser when the ad is actually rendered — the
// same contract as the CLI, where the ad is shown to the user before it counts.
let currentAd = null;

async function fetchAdFromBackend(slot, trigger, turnId) {
  const token = loadToken();
  if (!token) return null;
  try {
    const res = await fetch(`${BASE}/ads/next?slot=${encodeURIComponent(slot)}`, {
      headers: {
        ...upstreamHeaders(token, turnId),
        "x-client-os": process.platform,
        "x-client-arch": os.arch(),
        ...(trigger ? { "x-ad-trigger": trigger } : {}),
      },
    });
    adDebug.fetch_status = res.status;
    if (res.status === 204 || !res.ok) return null;
    return await res.json();
  } catch (err) {
    adDebug.last_error = "fetch: " + err.message;
    return null;
  }
}

async function postAdImpressionToBackend(body) {
  const token = loadToken();
  if (!token) return false;
  try {
    const res = await fetch(`${BASE}/ads/impression`, {
      method: "POST",
      headers: upstreamHeaders(token),
      body: JSON.stringify(body),
    });
    adDebug.imp_status = res.status;
    return res.ok;
  } catch (err) {
    adDebug.last_error = "impression: " + err.message;
    return false;
  }
}

// Per-turn ad cycle, matching the CLI exactly: on each user message the CLI
// fetches a banner ad (trigger "user_message", fresh x-turn-id, >=60s between
// fetches), fires the gravity impression pixel, and reports the impression —
// with the ad rendered to the user. Here the ad is rendered in the dashboard.
function isGravityImpUrl(raw) {
  if (!raw) return false;
  try {
    const u = new URL(raw);
    return u.protocol === "https:" && /(^|\.)trygravity\.(ai|com)$/.test(u.hostname);
  } catch {
    return false;
  }
}
let lastAd = null; // { ad, at }
let launchAdAt = 0; // when the session-start ("launch") ad was fetched
let lastPostedToken = null; // CLI semantics: only impression NEW click_tokens
let adInFlight = null;
const adDebug = { fetch_status: null, imp_status: null, dup_polls: 0, last_error: null };
// Per-message ad cycle, matching the CLI: every user message fetches a banner
// ad (trigger "user_message"; "launch" for the first ad of a session) and only
// NEW ads (new click_token) are impressed — the CLI does the same via
// lastPostedToken. The backend serves a new ad at most once per ~20s and
// pauses requests whose turn has no fresh ad, so poll briefly for one; the ad
// is displayed in the dashboard the whole time.
const AD_POLL_MAX_MS = 45000;
const AD_POLL_INTERVAL_MS = 3000;
async function maybeShowAd(turnId, trigger) {
  if (!trigger) trigger = launchAdAt === 0 ? "launch" : "user_message";
  if (adInFlight) return adInFlight;
  adInFlight = (async () => {
    const deadline = Date.now() + AD_POLL_MAX_MS;
    for (;;) {
      const ad = await fetchAdFromBackend("banner", trigger, turnId);
      if (!ad) return;
      if (ad.click_token !== lastPostedToken) {
        lastAd = { ad, at: Date.now() };
        if (trigger === "launch") launchAdAt = Date.now();
        lastPostedToken = ad.click_token;
        let impFired = false;
        if (ad.source === "gravity" && isGravityImpUrl(ad.imp_url)) {
          impFired = await fetch(ad.imp_url, { signal: AbortSignal.timeout(3000) }).then((r) => r.ok).catch(() => false);
        }
        await postAdImpressionToBackend({ ad_id: ad.ad_id, click_token: ad.click_token, ...(impFired ? { imp_fired: true } : {}) });
        return;
      }
      adDebug.dup_polls += 1;
      if (Date.now() >= deadline) { lastAd = { ad, at: Date.now() }; return; }
      await new Promise((r) => setTimeout(r, AD_POLL_INTERVAL_MS));
    }
  })().catch(() => {}).finally(() => { adInFlight = null; });
  return adInFlight;
}

// ---------- proxy core ----------

function json(res, status, payload) {
  const data = JSON.stringify(payload);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(data) });
  res.end(data);
}

function openAIError(message, type = "invalid_request_error", code = null) {
  return { error: { message, type, code } };
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > 32 * 1024 * 1024) {
        reject(new Error("request body too large"));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function checkAuth(req, res) {
  if (!PROXY_API_KEY) return true;
  const got = (req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (got === PROXY_API_KEY) return true;
  json(res, 401, openAIError("Invalid proxy API key. Set the client's apiKey to the proxy's PROXY_API_KEY.", "authentication_error"));
  return false;
}

// Route unknown/missing model names to the default free model.
async function resolveModel(requested) {
  const wanted = typeof requested === "string" ? requested.trim() : "";
  if (!wanted) return DEFAULT_MODEL;
  const models = await fetchCatalog();
  if (models.length && !models.some((m) => m.id === wanted)) return DEFAULT_MODEL;
  return wanted;
}

async function handleChatCompletions(req, res) {
  const raw = await readBody(req);
  let body;
  try {
    body = JSON.parse(raw.toString("utf8") || "{}");
  } catch {
    return json(res, 400, openAIError("Request body is not valid JSON."));
  }

  const wantsStream = body.stream === true;
  const useMcp = body.mcp !== false; // clients can opt out per request
  body.model = await resolveModel(body.model);
  const upstreamBody = normalizeBody(await applyMaxima(body));
  const token = loadToken();
  if (!token) {
    return json(res, 401, openAIError(
      "No free-pi credentials found. Run `npx free-pi-cli@latest` once to sign in, then restart the proxy.",
      "authentication_error"
    ));
  }

  // One ad cycle per user message, exactly like the CLI: new turn id, banner
  // ad (>=60s refresh), impression reported, same turn id on the chat call.
  const turnId = crypto.randomUUID();

  // Oversized screenshots (1-4 MB is normal) would exceed the backend's body
  // cap. Shrink them here instead of failing the user's request.
  const shrunk = downscaleBodyImages(upstreamBody);
  if (shrunk.length) {
    const saved = shrunk.reduce((n, r) => n + (r.beforeB64 - r.afterB64), 0);
    console.log(
      `[downscale] ${shrunk.length} image(s): ` +
      shrunk.map((r) => `${r.beforePx}px/${(r.beforeB64 / 1048576).toFixed(2)}MB -> ` +
        `${r.afterPx}px/${(r.afterB64 / 1048576).toFixed(2)}MB`).join(", ") +
      ` (saved ${(saved / 1048576).toFixed(2)} MB of base64)`
    );
  }

  // Fail fast on an attachment the backend will reject regardless of history:
  // its own 413 says "conversation is too long", which sends users chasing the
  // wrong fix. The cap is the request body size (~0.94 MB measured).
  const media = largestMediaBytes(upstreamBody);
  const bodyBytes = Buffer.byteLength(JSON.stringify(upstreamBody));
  if (bodyBytes > REQUEST_BODY_LIMIT_BYTES) {
    const overBy = ((bodyBytes - REQUEST_BODY_LIMIT_BYTES) / 1024).toFixed(0);
    const mediaMb = (media.bytes / 1048576).toFixed(2);
    const tried = shrunk.length
      ? ` The proxy already downscaled ${shrunk.length} image(s) and the body is still over.`
      : ` The proxy could not downscale this attachment: it can only shrink PNGs.`;
    return json(res, 413, openAIError(
      `Request body is ${(bodyBytes / 1048576).toFixed(2)} MB, over the backend's ~${(REQUEST_BODY_LIMIT_BYTES / 1048576).toFixed(2)} MB limit ` +
      `(${overBy} KB too large). Largest attachment: ${mediaMb} MB of base64.` + tried +
      ` This is a payload-size cap, not a context-length problem — dropping history cannot help. ` +
      `Convert the image to PNG (auto-downscaled), or re-encode it as JPEG/WebP and resize to <=1568px.`,
      "context_too_long"));
  }

  await maybeShowAd(turnId);

  // MCP servers connected? Inject their tools and run the server-side loop.
  const mcpTools = useMcp && MCP_MODE !== "off" && mcp.connected() ? mcp.toOpenAITools() : [];
  if (mcpTools.length) {
    const have = new Set((upstreamBody.tools ?? []).map((t) => t.function?.name).filter(Boolean));
    upstreamBody.tools = [...(upstreamBody.tools ?? []), ...mcpTools.filter((t) => !have.has(t.function.name))];
    return runWithMcp(token, upstreamBody, wantsStream, res, turnId);
  }

  const sentVersion = getClientVersion();
  let call;
  try {
    call = await fetchUpstreamWithCompaction(token, upstreamBody, turnId);
  } catch (err) {
    return json(res, 502, openAIError(`Cannot reach free-pi backend: ${err.message}`, "api_error"));
  }
  let upstream = call.resp;

  // Expired sign-in or outdated client version: refresh credentials and ask
  // the backend for its current minimum/latest version, then retry once.
  if (upstream.status === 401 || upstream.status === 426) {
    credCache = { token: null, mtime: 0 };
    if (upstream.status === 426) await refreshClientVersion({ force: true });
    const fresh = loadToken();
    const freshVersion = getClientVersion();
    if (fresh && (fresh !== token || freshVersion !== sentVersion)) {
      try {
        call = await fetchUpstreamWithCompaction(fresh, call.body, turnId);
      } catch (err) {
        return json(res, 502, openAIError(`Cannot reach free-pi backend: ${err.message}`, "api_error"));
      }
      upstream = call.resp;
    }
  }

  if (!upstream.ok) {
    const text = call.text ?? await upstream.text().catch(() => "");
    let detail = text.slice(0, 2000);
    try {
      const parsed = JSON.parse(text);
      detail = parsed?.error?.message ?? parsed?.detail ?? JSON.stringify(parsed).slice(0, 2000);
    } catch {}
    const overflow = call.overflow === true || isContextOverflow(upstream.status, text);
    const hint =
      overflow
        ? call.mediaBound
          ? " (an attachment is so large it fills the model window on its own — remove or shrink it; dropping older messages cannot help)"
          : " (the proxy already dropped the oldest turns and the request still does not fit — start a new chat, or lower the attached files/context)"
        : upstream.status === 401
          ? " (free-pi sign-in expired — run `npx free-pi-cli@latest` to sign in again)"
          : upstream.status === 426
            ? ` (proxy sent client version ${sentVersion} and could not self-heal; run \`npx free-pi-cli@latest\` once so the CLI records its new version, or set FREEPI_CLIENT_VERSION, or check connectivity to ${BASE})`
            : upstream.status === 429
              ? " (daily free-pi allowance spent — /usage in free-pi, or /buy-credits)"
              : "";
    return json(res, overflow ? 413 : upstream.status === 401 || upstream.status === 429 ? upstream.status : 502,
      openAIError(`free-pi backend ${upstream.status}: ${detail}${hint}`, overflow ? "context_too_long" : "api_error"));
  }

  if (!wantsStream) {
    const data = call.text ?? await upstream.text();
    try {
      const parsed = JSON.parse(data);
      recordUsage(parsed.model ?? upstreamBody.model, parsed.usage);
    } catch {}
    res.writeHead(upstream.status, {
      "content-type": "application/json",
      ...(call.compacted ? { "x-freepi-compacted": "1" } : {}),
    });
    return res.end(data);
  }

  // Streaming: pipe SSE through untouched so tool_call deltas stay intact;
  // a tap scans events to record usage from the final chunk.
  res.writeHead(200, {
    "content-type": upstream.headers.get("content-type") || "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...(call.compacted ? { "x-freepi-compacted": "1" } : {}),
  });
  const tap = makeSseScanner((ev) => {
    if (ev.usage) recordUsage(ev.model ?? upstreamBody.model, ev.usage);
  });
  Readable.fromWeb(upstream.body).pipe(tap).pipe(res);
}

// Transform that lets SSE bytes pass through untouched while parsing
// `data: {...}` lines for the stats tap.
function makeSseScanner(onEvent) {
  let buf = "";
  return new Transform({
    transform(chunk, enc, cb) {
      buf += chunk.toString("utf8");
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data !== "[DONE]") {
            try { onEvent(JSON.parse(data)); } catch {}
          }
        }
      }
      cb(null, chunk);
    },
  });
}

async function handleModels(res, singleId) {
  const models = await fetchCatalog();
  const specs = await fetchModelSpecs();
  const toModel = (m) => {
    const spec = specFor(m.id, specs);
    const o = toOpenAIModel(m);
    if (spec) {
      o.context_length = spec.contextWindow;
      o.max_model_len = spec.contextWindow;
      o.context_window = spec.contextWindow;
      o.max_output_tokens = spec.maxOutput;
      o.input_modalities = spec.inputs;
      o.free_pi_tool_calling = spec.toolCalling;
      o.free_pi_reasoning_effort = spec.reasoningEffort;
    }
    return o;
  };
  if (singleId != null) {
    const m = models.find((m) => m.id === singleId);
    if (!m) return json(res, 404, openAIError(`Model '${singleId}' not found in free-pi catalog.`, "invalid_request_error", "model_not_found"));
    return json(res, 200, toModel(m));
  }
  json(res, 200, { object: "list", data: models.map(toModel) });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  const p = url.pathname.replace(/\/+$/, "") || "/";

  if ((p === "/" || p === "/dashboard") && req.method === "GET") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    return res.end(DASHBOARD_HTML);
  }

  if (p === "/stats" && req.method === "GET") {
    fetchAllowance().catch(() => {}); // refresh in background, serve cache
    return json(res, 200, statsSnapshot());
  }

  // Save (or check-only) a free-pi token pasted into the dashboard. Write
  // endpoint -> requires the proxy API key when one is configured. Accepts
  // either the raw token string or a full credentials.json body. Validated
  // against /me before saving; on success the token is auto-synced to the
  // configured GitHub Actions secret (env GH_TOKEN + GH_REPO).
  if (p === "/token" && req.method === "POST") {
    if (!checkAuth(req, res)) return;
    const raw = await readBody(req);
    let body = {};
    let token = raw.toString("utf8").trim();
    // normalize smart quotes that chat apps / editors inject into pasted JSON
    token = token.replace(/[\u201c\u201d\u201e\u201f]/g, '"').replace(/[\u2018\u2019]/g, "'");
    try {
      const parsed = JSON.parse(token);
      if (typeof parsed === "string") token = parsed.trim();
      else {
        body = parsed ?? {};
        if (typeof parsed?.token === "string") token = parsed.token.trim();
        else if (typeof parsed?.jwt === "string") token = parsed.jwt.trim();
      }
    } catch { /* plain token string */ }
    // fallback: extract the JWT from a mangled paste (prose, partial JSON, wrapped text)
    if (!/^eyJ[A-Za-z0-9_-]+\./.test(token)) {
      const m = token.match(/eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/);
      if (m) token = m[0];
    }
    token = token.replace(/\s+/g, ""); // strip line wraps / stray whitespace inside pasted JWTs
    const checkOnly = body.check_only === true;
    if (!token || token.length < 20) {
      return json(res, 400, openAIError("Paste the contents of credentials.json or the raw token string."));
    }
    // Validate against the backend before anything else. A bad token never
    // touches the credentials file.
    let account = null;
    try {
      const check = await fetch(`${BASE}/me`, { headers: upstreamHeaders(token) });
      if (!check.ok) {
        return json(res, 401, openAIError(`Token rejected by free-pi backend (HTTP ${check.status}). Not saved. [pasted token: ${token.length} chars, ends ...${token.slice(-8)}]`, "authentication_error"));
      }
      try { account = await check.json(); } catch {}
    } catch (err) {
      return json(res, 502, openAIError(`Cannot reach free-pi backend to validate token: ${err.message}`, "api_error"));
    }
    const accountInfo = account && {
      tier: account.tier,
      remaining_credits_today: account.remaining_credits_today,
      allowance_credits: account.allowance_credits,
      refill_at: account.refill_at,
    };
    if (checkOnly) {
      return json(res, 200, { ok: true, check_only: true, account: accountInfo, message: "Token is valid. Not saved (check only)." });
    }
    try {
      fs.mkdirSync(path.dirname(CREDENTIALS_PATH), { recursive: true });
      fs.writeFileSync(CREDENTIALS_PATH, JSON.stringify({ token }) + "\n");
      credCache = { token, mtime: fs.statSync(CREDENTIALS_PATH).mtimeMs };
      allowance = { data: null, at: 0 }; // re-poll with the new token
      console.log("[token] credentials updated via dashboard");
    } catch (err) {
      return json(res, 500, openAIError(`Could not write credentials file: ${err.message}`, "api_error"));
    }
    // Auto-sync to GitHub Actions secret so VM runs pick up the fresh token.
    const sync = await syncTokenToGithub();
    if (sync.attempted) console.log(`[token] github sync: ${sync.ok ? "OK" : "FAILED"} — ${sync.detail}`);
    return json(res, 200, {
      ok: true,
      signed_in: true,
      account: accountInfo,
      github_sync: sync,
      message: sync.ok
        ? `Token validated and saved \u2014 now in use: ${tokenFp(token)}. GitHub secret ${sync.secret} updated.`
        : sync.attempted
          ? `Token validated and saved \u2014 now in use: ${tokenFp(token)} \u2014 but GitHub sync failed: ${sync.detail}`
          : `Token validated and saved \u2014 now in use: ${tokenFp(token)} (GitHub sync not configured).`,
    });
  }

  // Fetch an ad to display in the dashboard (only called by an open dashboard).
  if (p === "/ads/next" && req.method === "GET") {
    if (!checkAuth(req, res)) return;
    const slot = url.searchParams.get("slot") === "inline" ? "inline" : "banner";
    const trigger = url.searchParams.get("trigger") || undefined;
    const ad = await fetchAdFromBackend(slot, trigger);
    if (!ad) return json(res, 200, { ad: null, slot });
    currentAd = { ad, at: Date.now(), slot };
    return json(res, 200, { ad, slot });
  }

  // Report an impression after the dashboard actually displayed the ad.
  if (p === "/ads/impression" && req.method === "POST") {
    if (!checkAuth(req, res)) return;
    const raw = await readBody(req);
    let body = {};
    try { body = JSON.parse(raw.toString("utf8") || "{}"); } catch {}
    if (!body.ad_id || !body.click_token) {
      return json(res, 400, openAIError("ad_id and click_token are required."));
    }
    const ok = await postAdImpressionToBackend({
      ad_id: String(body.ad_id),
      click_token: String(body.click_token),
      ...(body.imp_fired ? { imp_fired: true } : {}),
    });
    return json(res, ok ? 200 : 502, { ok });
  }

  if (p === "/health") {
    const token = loadToken();
    return json(res, 200, { ok: true, signed_in: Boolean(token), backend: BASE, session: SESSION_ID, mcp_servers: mcp.connected() ? [...mcp.servers.keys()] : [] });
  }

  if (!checkAuth(req, res)) return;

  if (req.method === "POST" && (p === "/v1/chat/completions" || p === "/chat/completions")) {
    try {
      await handleChatCompletions(req, res);
    } catch (err) {
      if (!res.headersSent) json(res, 500, openAIError(`Proxy error: ${err.message}`, "api_error"));
      else res.destroy();
    }
    return;
  }

  if (req.method === "GET" && (p === "/v1/models" || p === "/models")) return handleModels(res);
  if (req.method === "GET" && (p === "/v1/models/" || p.startsWith("/v1/models/"))) {
    return handleModels(res, decodeURIComponent(p.split("/").pop()));
  }

  json(res, 404, openAIError(`Unknown route ${req.method} ${p}. Use POST /v1/chat/completions or GET /v1/models.`));
});

server.listen(PORT, HOST, async () => {
  await initClientVersion();
  const token = loadToken();
  console.log(`free-pi proxy  http://${HOST}:${PORT}/v1`);
  console.log(`  backend      ${BASE}`);
  console.log(`  client ver   ${getClientVersion()} (backend min ${versionInfo.min ?? "?"}, latest ${versionInfo.latest ?? "?"})`);
  console.log(`  credentials  ${CREDENTIALS_PATH} (${token ? "signed in" : "NOT SIGNED IN — run npx free-pi-cli@latest"})`);
  console.log(`  session      ${SESSION_ID}`);
  console.log(`  auth         ${PROXY_API_KEY ? "PROXY_API_KEY required" : "open (binds to localhost only)"}`);
  console.log(`  default model ${DEFAULT_MODEL} (unknown model names route here)`);
  console.log(`  dashboard     http://${HOST === "0.0.0.0" ? "<this-host>" : HOST}:${PORT}/`);
  const models = await fetchCatalog();
  fetchModelSpecs().catch(() => {}); // background: don't block startup on flaky fetches
  const specs = { ...BUILT_IN_SPECS, ...(specsCache.specs ?? {}) };
  const glm = specs["glm-5.3-flash"];
  console.log(`  maxima       context=${glm?.contextWindow ?? CONTEXT_WINDOW} max_tokens=${glm?.maxOutput ?? MAX_OUTPUT_TOKENS} reasoning_effort=${REASONING_EFFORT}${FORCE_MAXIMA ? " (forced)" : ""} (${Object.keys(specs).length} model specs loaded${specsCache.at ? "" : " — fetching in background"})`);
  console.log(`  models       ${models.filter((m) => m.section !== "defense").map((m) => m.id).join(", ")}`);
  console.log(`  overflow     drop oldest turns after a 413 (up to ${COMPACT_MAX_TRIES} retries, keep ${Math.round(COMPACT_KEEP_RATIO * 100)}%; overrides FREEPI_PROXY_COMPACT_*)`);
  if (MCP_MODE !== "off") {
    mcp.connectAll().then((n) => console.log(`  mcp          ${n} server(s) connected${n ? ": " + [...mcp.servers.keys()].join(", ") : " (no mcp.json found)"}`)).catch(() => {});
  } else {
    console.log("  mcp          disabled (MCP_MODE=off)");
  }
});
