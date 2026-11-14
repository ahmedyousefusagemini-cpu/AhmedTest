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
// misleading "This conversation is too long" 413. Measured against the live
// backend, the cap is a flat limit on the JSON CHARACTER COUNT of the request
// body -- not bytes, pixels or tokens. Binary-searched boundary: 999,907 chars
// succeeded, 1,000,407 chars returned 413. That is why ASCII prose (1 byte per
// char) fails near 1 MB while CJK (3 bytes per char) fits 2.5 MB of text inside
// the same 1M-character budget. Compaction can never fix an oversized ATTACHMENT
// (it survives every trim), so compare characters, not bytes.
const REQUEST_BODY_LIMIT_CHARS = 1_000_000;
// Only used to decide whether an attachment dominates the body.
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
// The backend rejects a request body over 1,000,000 JSON characters (see
// REQUEST_BODY_LIMIT_CHARS). Base64 expands bytes by 4/3, so an inline
// attachment quickly eats that character budget. Rather than reject the user's
// 1-4 MB screenshot, shrink it here: decode PNG in pure JS (node:zlib only, no
// native dependency so the workflow needs no npm install), box-filter down to a
// sane pixel budget, and re-encode as PNG. Images we cannot decode (JPEG/WebP)
// fall back to the size-limit error.
const DOWNSCALE_ENABLED = process.env.FREEPI_PROXY_DOWNSCALE !== "0";
// Vision cost tracks px/763; ~1.15 MP keeps quality high while costing ~1500
// tokens. Long side is capped separately so panoramas stay readable.
const DOWNSCALE_MAX_PIXELS = Number(process.env.FREEPI_PROXY_MAX_IMAGE_PIXELS || 1150000);
const DOWNSCALE_MAX_SIDE = Number(process.env.FREEPI_PROXY_MAX_IMAGE_SIDE || 1800);
// Target base64 characters for a single attachment (leaves room for the rest of
// the body under the 1,000,000-character cap).
const DOWNSCALE_TARGET_B64 = Number(process.env.FREEPI_PROXY_IMAGE_BYTES || 620000);
// Quality used when re-encoding a JPEG attachment. 85 is visually near-lossless
// for screenshots and photos while keeping the payload small.
const DOWNSCALE_JPEG_QUALITY = Number(process.env.FREEPI_PROXY_JPEG_QUALITY || 85);

const PNG_SIG = 0x89504e47;

// SOI marker. Covers every JPEG variant we can decode (JFIF, EXIF, progressive).
function isJpeg(buf) {
  return buf.length > 3 && buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff;
}
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
  // Only 8- and 16-bit, non-interlaced images are supported. Palette (ct 3)
  // images are rare for screenshots and are left to the caller.
  if (!w || !h || inter) return null;
  if (depth !== 8 && depth !== 16) return null;
  const chan = ct === 2 ? 3 : ct === 6 ? 4 : ct === 0 ? 1 : ct === 4 ? 2 : null;
  if (!chan) return null;
  let raw;
  try { raw = zlib.inflateSync(Buffer.concat(idat)); } catch { return null; }
  // Filtering operates on bytes, so for 16-bit images the stride is in bytes and
  // each sample is two bytes (high byte first).
  const bps = depth === 16 ? 2 : 1;
  const stride = w * chan * bps;
  if (raw.length < (stride + 1) * h) return null;
  const filt = Buffer.alloc(h * stride);
  let pos = 0;
  for (let y = 0; y < h; y++) {
    const ft = raw[pos++];
    const line = raw.subarray(pos, pos + stride); pos += stride;
    const start = y * stride;
    for (let x = 0; x < stride; x++) {
      const a = x >= bps ? filt[start + x - bps] : 0;
      const b = y > 0 ? filt[start - stride + x] : 0;
      const c = (y > 0 && x >= bps) ? filt[start - stride + x - bps] : 0;
      let v = line[x];
      if (ft === 1) v += a;
      else if (ft === 2) v += b;
      else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
      filt[start + x] = v & 255;
    }
  }
  if (depth === 8) return { w, h, chan, data: filt };
  // 16-bit: keep the high byte of each big-endian sample.
  const out = Buffer.alloc(w * h * chan);
  for (let i = 0, j = 0; i < filt.length; i += 2, j++) out[j] = filt[i];
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

// ---------------------------------------------------------------------------
// JPEG support for downscaling.
//
// Baseline JPEGs are decoded by decodeJpeg() below. Progressive and other exotic
// JPEGs use the fuller decoder vendored from jpeg-js, because the deploy workflow
// cannot install npm packages.
//
// Copyright 2011 notmasteryet. Licensed under the Apache License, Version 2.0:
//   http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
//
// Upstream code, minified and otherwise unmodified.
// ---------------------------------------------------------------------------
var JpegImage=function(){"use strict";var e=new Int32Array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63]),r=4017,n=799,o=3406,a=2276,t=1567,s=3784,i=5793,c=2896;function f(){}function l(e,r){for(var n,o,a=0,t=[],s=16;s>0&&!e[s-1];)s--;t.push({children:[],index:0});var i,c=t[0];for(n=0;n<s;n++){for(o=0;o<e[n];o++){for((c=t.pop()).children[c.index]=r[a];c.index>0;){if(0===t.length)throw new Error("Could not recreate Huffman Table");c=t.pop()}for(c.index++,t.push(c);t.length<=n;)t.push(i={children:[],index:0}),c.children[c.index]=i.children,c=i;a++}n+1<s&&(t.push(i={children:[],index:0}),c.children[c.index]=i.children,c=i)}return t[0].children}function h(r,n,o,a,t,s,i,c,f,l){o.precision,o.samplesPerLine,o.scanLines;var h=o.mcusPerLine,u=o.progressive,m=(o.maxH,o.maxV,n),d=0,p=0;function v(){if(p>0)return p--,d>>p&1;if(255==(d=r[n++])){var e=r[n++];if(e)throw new Error("unexpected marker: "+(d<<8|e).toString(16))}return p=7,d>>>7}function w(e){for(var r,n=e;null!==(r=v());){if("number"==typeof(n=n[r]))return n;if("object"!=typeof n)throw new Error("invalid huffman sequence")}return null}function b(e){for(var r=0;e>0;){var n=v();if(null===n)return;r=r<<1|n,e--}return r}function g(e){var r=b(e);return r>=1<<e-1?r:r+(-1<<e)+1}var y=0;var k,x=0;function A(e,r,n,o,a){var t=n%h,s=(n/h|0)*e.v+o,i=t*e.h+a;void 0===e.blocks[s]&&l.tolerantDecoding||r(e,e.blocks[s][i])}function P(e,r,n){var o=n/e.blocksPerLine|0,a=n%e.blocksPerLine;void 0===e.blocks[o]&&l.tolerantDecoding||r(e,e.blocks[o][a])}var M,E,T,C,I,L,X=a.length;L=u?0===s?0===c?function(e,r){var n=w(e.huffmanTableDC),o=0===n?0:g(n)<<f;r[0]=e.pred+=o}:function(e,r){r[0]|=v()<<f}:0===c?function(r,n){if(y>0)y--;else for(var o=s,a=i;o<=a;){var t=w(r.huffmanTableAC),c=15&t,l=t>>4;if(0!==c)n[e[o+=l]]=g(c)*(1<<f),o++;else{if(l<15){y=b(l)+(1<<l)-1;break}o+=16}}}:function(r,n){for(var o=s,a=i,t=0;o<=a;){var c=e[o],l=n[c]<0?-1:1;switch(x){case 0:var h=w(r.huffmanTableAC),u=15&h;if(t=h>>4,0===u)t<15?(y=b(t)+(1<<t),x=4):(t=16,x=1);else{if(1!==u)throw new Error("invalid ACn encoding");k=g(u),x=t?2:3}continue;case 1:case 2:n[c]?n[c]+=(v()<<f)*l:0===--t&&(x=2==x?3:0);break;case 3:n[c]?n[c]+=(v()<<f)*l:(n[c]=k<<f,x=0);break;case 4:n[c]&&(n[c]+=(v()<<f)*l)}o++}4===x&&0===--y&&(x=0)}:function(r,n){var o=w(r.huffmanTableDC),a=0===o?0:g(o);n[0]=r.pred+=a;for(var t=1;t<64;){var s=w(r.huffmanTableAC),i=15&s,c=s>>4;if(0!==i)n[e[t+=c]]=g(i),t++;else{if(c<15)break;t+=16}}};var U,B,D,R,Y=0;for(B=1==X?a[0].blocksPerLine*a[0].blocksPerColumn:h*o.mcusPerColumn,t||(t=B);Y<B;){for(E=0;E<X;E++)a[E].pred=0;if(y=0,1==X)for(M=a[0],I=0;I<t;I++)P(M,L,Y),Y++;else for(I=0;I<t;I++){for(E=0;E<X;E++)for(D=(M=a[E]).h,R=M.v,T=0;T<R;T++)for(C=0;C<D;C++)A(M,L,Y,T,C);if(++Y===B)break}if(Y===B)do{if(255===r[n]&&0!==r[n+1])break;n+=1}while(n<r.length-2);if(p=0,(U=r[n]<<8|r[n+1])<65280)throw new Error("marker was not found");if(!(U>=65488&&U<=65495))break;n+=2}return n-m}function u(e,f){var l,h,u=[],m=f.blocksPerLine,d=f.blocksPerColumn,p=m<<3,w=new Int32Array(64),b=new Uint8Array(64);function g(e,l,h){var u,m,d,p,v,w,b,g,y,k,x=f.quantizationTable,A=h;for(k=0;k<64;k++)A[k]=e[k]*x[k];for(k=0;k<8;++k){var P=8*k;0!=A[1+P]||0!=A[2+P]||0!=A[3+P]||0!=A[4+P]||0!=A[5+P]||0!=A[6+P]||0!=A[7+P]?(u=i*A[0+P]+128>>8,m=i*A[4+P]+128>>8,d=A[2+P],p=A[6+P],v=c*(A[1+P]-A[7+P])+128>>8,g=c*(A[1+P]+A[7+P])+128>>8,w=A[3+P]<<4,b=A[5+P]<<4,y=u-m+1>>1,u=u+m+1>>1,m=y,y=d*s+p*t+128>>8,d=d*t-p*s+128>>8,p=y,y=v-b+1>>1,v=v+b+1>>1,b=y,y=g+w+1>>1,w=g-w+1>>1,g=y,y=u-p+1>>1,u=u+p+1>>1,p=y,y=m-d+1>>1,m=m+d+1>>1,d=y,y=v*a+g*o+2048>>12,v=v*o-g*a+2048>>12,g=y,y=w*n+b*r+2048>>12,w=w*r-b*n+2048>>12,b=y,A[0+P]=u+g,A[7+P]=u-g,A[1+P]=m+b,A[6+P]=m-b,A[2+P]=d+w,A[5+P]=d-w,A[3+P]=p+v,A[4+P]=p-v):(y=i*A[0+P]+512>>10,A[0+P]=y,A[1+P]=y,A[2+P]=y,A[3+P]=y,A[4+P]=y,A[5+P]=y,A[6+P]=y,A[7+P]=y)}for(k=0;k<8;++k){var M=k;0!=A[8+M]||0!=A[16+M]||0!=A[24+M]||0!=A[32+M]||0!=A[40+M]||0!=A[48+M]||0!=A[56+M]?(u=i*A[0+M]+2048>>12,m=i*A[32+M]+2048>>12,d=A[16+M],p=A[48+M],v=c*(A[8+M]-A[56+M])+2048>>12,g=c*(A[8+M]+A[56+M])+2048>>12,w=A[24+M],b=A[40+M],y=u-m+1>>1,u=u+m+1>>1,m=y,y=d*s+p*t+2048>>12,d=d*t-p*s+2048>>12,p=y,y=v-b+1>>1,v=v+b+1>>1,b=y,y=g+w+1>>1,w=g-w+1>>1,g=y,y=u-p+1>>1,u=u+p+1>>1,p=y,y=m-d+1>>1,m=m+d+1>>1,d=y,y=v*a+g*o+2048>>12,v=v*o-g*a+2048>>12,g=y,y=w*n+b*r+2048>>12,w=w*r-b*n+2048>>12,b=y,A[0+M]=u+g,A[56+M]=u-g,A[8+M]=m+b,A[48+M]=m-b,A[16+M]=d+w,A[40+M]=d-w,A[24+M]=p+v,A[32+M]=p-v):(y=i*h[k+0]+8192>>14,A[0+M]=y,A[8+M]=y,A[16+M]=y,A[24+M]=y,A[32+M]=y,A[40+M]=y,A[48+M]=y,A[56+M]=y)}for(k=0;k<64;++k){var E=128+(A[k]+8>>4);l[k]=E<0?0:E>255?255:E}}v(p*d*8);for(var y=0;y<d;y++){var k=y<<3;for(l=0;l<8;l++)u.push(new Uint8Array(p));for(var x=0;x<m;x++){g(f.blocks[y][x],b,w);var A=0,P=x<<3;for(h=0;h<8;h++){var M=u[k+h];for(l=0;l<8;l++)M[P+l]=b[A++]}}}return u}function m(e){return e<0?0:e>255?255:e}f.prototype={load:function(e){var r=new XMLHttpRequest;r.open("GET",e,!0),r.responseType="arraybuffer",r.onload=function(){var e=new Uint8Array(r.response||r.mozResponseArrayBuffer);this.parse(e),this.onload&&this.onload()}.bind(this),r.send(null)},parse:function(r){var n=1e3*this.opts.maxResolutionInMP*1e3,o=0;r.length;function a(){var e=r[o]<<8|r[o+1];return o+=2,e}function t(){var e=a(),n=r.subarray(o,o+e-2);return o+=n.length,n}function s(e){var r,n,o=1,a=1;for(n in e.components)e.components.hasOwnProperty(n)&&(o<(r=e.components[n]).h&&(o=r.h),a<r.v&&(a=r.v));var t=Math.ceil(e.samplesPerLine/8/o),s=Math.ceil(e.scanLines/8/a);for(n in e.components)if(e.components.hasOwnProperty(n)){r=e.components[n];var i=Math.ceil(Math.ceil(e.samplesPerLine/8)*r.h/o),c=Math.ceil(Math.ceil(e.scanLines/8)*r.v/a),f=t*r.h,l=s*r.v,h=[];v(256*(l*f));for(var u=0;u<l;u++){for(var m=[],d=0;d<f;d++)m.push(new Int32Array(64));h.push(m)}r.blocksPerLine=i,r.blocksPerColumn=c,r.blocks=h}e.maxH=o,e.maxV=a,e.mcusPerLine=t,e.mcusPerColumn=s}var i,c,f=null,m=null,d=[],p=[],w=[],b=[],g=a(),y=-1;if(this.comments=[],65496!=g)throw new Error("SOI not found");for(g=a();65497!=g;){switch(g){case 65280:break;case 65504:case 65505:case 65506:case 65507:case 65508:case 65509:case 65510:case 65511:case 65512:case 65513:case 65514:case 65515:case 65516:case 65517:case 65518:case 65519:case 65534:var k=t();if(65534===g){var x=String.fromCharCode.apply(null,k);this.comments.push(x)}65504===g&&74===k[0]&&70===k[1]&&73===k[2]&&70===k[3]&&0===k[4]&&(f={version:{major:k[5],minor:k[6]},densityUnits:k[7],xDensity:k[8]<<8|k[9],yDensity:k[10]<<8|k[11],thumbWidth:k[12],thumbHeight:k[13],thumbData:k.subarray(14,14+3*k[12]*k[13])}),65505===g&&69===k[0]&&120===k[1]&&105===k[2]&&102===k[3]&&0===k[4]&&(this.exifBuffer=k.subarray(5,k.length)),65518===g&&65===k[0]&&100===k[1]&&111===k[2]&&98===k[3]&&101===k[4]&&0===k[5]&&(m={version:k[6],flags0:k[7]<<8|k[8],flags1:k[9]<<8|k[10],transformCode:k[11]});break;case 65499:for(var A=a()+o-2;o<A;){var P=r[o++];v(256);var M=new Int32Array(64);if(P>>4){if(P>>4!=1)throw new Error("DQT: invalid table spec");for(Q=0;Q<64;Q++){M[e[Q]]=a()}}else for(Q=0;Q<64;Q++){M[e[Q]]=r[o++]}d[15&P]=M}break;case 65472:case 65473:case 65474:a(),(i={}).extended=65473===g,i.progressive=65474===g,i.precision=r[o++],i.scanLines=a(),i.samplesPerLine=a(),i.components={},i.componentsOrder=[];var E=i.scanLines*i.samplesPerLine;if(E>n){var T=Math.ceil((E-n)/1e6);throw new Error(`maxResolutionInMP limit exceeded by ${T}MP`)}var C,I=r[o++];for($=0;$<I;$++){C=r[o];var L=r[o+1]>>4,X=15&r[o+1],U=r[o+2];if(L<=0||X<=0)throw new Error("Invalid sampling factor, expected values above 0");i.componentsOrder.push(C),i.components[C]={h:L,v:X,quantizationIdx:U},o+=3}s(i),p.push(i);break;case 65476:var B=a();for($=2;$<B;){var D=r[o++],R=new Uint8Array(16),Y=0;for(Q=0;Q<16;Q++,o++)Y+=R[Q]=r[o];v(16+Y);var q=new Uint8Array(Y);for(Q=0;Q<Y;Q++,o++)q[Q]=r[o];$+=17+Y,(D>>4?w:b)[15&D]=l(R,q)}break;case 65501:a(),c=a();break;case 65500:a(),a();break;case 65498:a();var j=r[o++],G=[];for($=0;$<j;$++){W=i.components[r[o++]];var J=r[o++];W.huffmanTableDC=b[J>>4],W.huffmanTableAC=w[15&J],G.push(W)}var O=r[o++],S=r[o++],z=r[o++],H=h(r,o,i,G,c,O,S,z>>4,15&z,this.opts);o+=H;break;case 65535:255!==r[o]&&o--;break;default:if(255==r[o-3]&&r[o-2]>=192&&r[o-2]<=254){o-=3;break}if(224===g||225==g){if(-1!==y)throw new Error(`first unknown JPEG marker at offset ${y.toString(16)}, second unknown JPEG marker ${g.toString(16)} at offset ${(o-1).toString(16)}`);y=o-1;const e=a();if(255===r[o+e-2]){o+=e-2;break}}throw new Error("unknown JPEG marker "+g.toString(16))}g=a()}if(1!=p.length)throw new Error("only single frame JPEGs supported");for(var $=0;$<p.length;$++){var V=p[$].components;for(var Q in V)V[Q].quantizationTable=d[V[Q].quantizationIdx],delete V[Q].quantizationIdx}this.width=i.samplesPerLine,this.height=i.scanLines,this.jfif=f,this.adobe=m,this.components=[];for($=0;$<i.componentsOrder.length;$++){var W=i.components[i.componentsOrder[$]];this.components.push({lines:u(0,W),scaleX:W.h/i.maxH,scaleY:W.v/i.maxV})}},getData:function(e,r){var n,o,a,t,s,i,c,f,l,h,u,d,p,w,b,g,y,k,x,A,P,M=this.width/e,E=this.height/r,T=0,C=e*r*this.components.length;v(C);var I=new Uint8Array(C);switch(this.components.length){case 1:for(n=this.components[0],h=0;h<r;h++)for(s=n.lines[0|h*n.scaleY*E],l=0;l<e;l++)u=s[0|l*n.scaleX*M],I[T++]=u;break;case 2:for(n=this.components[0],o=this.components[1],h=0;h<r;h++)for(s=n.lines[0|h*n.scaleY*E],i=o.lines[0|h*o.scaleY*E],l=0;l<e;l++)u=s[0|l*n.scaleX*M],I[T++]=u,u=i[0|l*o.scaleX*M],I[T++]=u;break;case 3:for(P=!0,this.adobe&&this.adobe.transformCode?P=!0:void 0!==this.opts.colorTransform&&(P=!!this.opts.colorTransform),n=this.components[0],o=this.components[1],a=this.components[2],h=0;h<r;h++)for(s=n.lines[0|h*n.scaleY*E],i=o.lines[0|h*o.scaleY*E],c=a.lines[0|h*a.scaleY*E],l=0;l<e;l++)P?(u=s[0|l*n.scaleX*M],d=i[0|l*o.scaleX*M],k=m(u+1.402*((p=c[0|l*a.scaleX*M])-128)),x=m(u-.3441363*(d-128)-.71413636*(p-128)),A=m(u+1.772*(d-128))):(k=s[0|l*n.scaleX*M],x=i[0|l*o.scaleX*M],A=c[0|l*a.scaleX*M]),I[T++]=k,I[T++]=x,I[T++]=A;break;case 4:if(!this.adobe)throw new Error("Unsupported color mode (4 components)");for(P=!1,this.adobe&&this.adobe.transformCode?P=!0:void 0!==this.opts.colorTransform&&(P=!!this.opts.colorTransform),n=this.components[0],o=this.components[1],a=this.components[2],t=this.components[3],h=0;h<r;h++)for(s=n.lines[0|h*n.scaleY*E],i=o.lines[0|h*o.scaleY*E],c=a.lines[0|h*a.scaleY*E],f=t.lines[0|h*t.scaleY*E],l=0;l<e;l++)P?(u=s[0|l*n.scaleX*M],d=i[0|l*o.scaleX*M],p=c[0|l*a.scaleX*M],w=f[0|l*t.scaleX*M],b=255-m(u+1.402*(p-128)),g=255-m(u-.3441363*(d-128)-.71413636*(p-128)),y=255-m(u+1.772*(d-128))):(b=s[0|l*n.scaleX*M],g=i[0|l*o.scaleX*M],y=c[0|l*a.scaleX*M],w=f[0|l*t.scaleX*M]),I[T++]=255-b,I[T++]=255-g,I[T++]=255-y,I[T++]=255-w;break;default:throw new Error("Unsupported color mode")}return I},copyToImageData:function(e,r){var n,o,a,t,s,i,c,f,l,h=e.width,u=e.height,d=e.data,p=this.getData(h,u),v=0,w=0;switch(this.components.length){case 1:for(o=0;o<u;o++)for(n=0;n<h;n++)a=p[v++],d[w++]=a,d[w++]=a,d[w++]=a,r&&(d[w++]=255);break;case 3:for(o=0;o<u;o++)for(n=0;n<h;n++)c=p[v++],f=p[v++],l=p[v++],d[w++]=c,d[w++]=f,d[w++]=l,r&&(d[w++]=255);break;case 4:for(o=0;o<u;o++)for(n=0;n<h;n++)s=p[v++],i=p[v++],a=p[v++],c=255-m(s*(1-(t=p[v++])/255)+t),f=255-m(i*(1-t/255)+t),l=255-m(a*(1-t/255)+t),d[w++]=c,d[w++]=f,d[w++]=l,r&&(d[w++]=255);break;default:throw new Error("Unsupported color mode")}}};var d=0,p=0;function v(e=0){var r=d+e;if(r>p){var n=Math.ceil((r-p)/1024/1024);throw new Error(`maxMemoryUsageInMB limit exceeded by at least ${n}MB`)}d=r}return f.resetMaxMemoryUsage=function(e){d=0,p=e},f.getBytesAllocated=function(){return d},f.requestMemoryAllocation=v,f}();

// Decode any JPEG (baseline or progressive) into { w, h, chan: 3, data: RGB }.
function decodeJpegFull(buf) {
  try {
    JpegImage.resetMaxMemoryUsage(768 * 1024 * 1024);
    const img = new JpegImage();
    img.opts = { maxResolutionInMP: 120, maxMemoryUsageInMB: 768, tolerantDecoding: true };
    img.parse(buf);
    if (!img.width || !img.height) return null;
    if (img.width * img.height > 64e6) return null; // refuse absurd dimensions
    const rgba = new Uint8Array(img.width * img.height * 4);
    img.copyToImageData({ data: rgba, width: img.width, height: img.height }, true);
    const rgb = Buffer.allocUnsafe(img.width * img.height * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      rgb[j] = rgba[i];
      rgb[j + 1] = rgba[i + 1];
      rgb[j + 2] = rgba[i + 2];
    }
    return { w: img.width, h: img.height, chan: 3, data: rgb };
  } catch (err) {
    log("downscale", "jpeg decode failed:", err && err.message);
    return null;
  }
}

// ---------------------------------------------------------------------------
// JPEG encoder, also vendored from jpeg-js (Apache-2.0, see notice above).
// ---------------------------------------------------------------------------
var btoa=btoa||function(r){return Buffer.from(r).toString("base64")};function JPEGEncoder(r){Math.round;var e,n,o,a,t,f=Math.floor,i=new Array(64),v=new Array(64),d=new Array(64),u=new Array(64),c=new Array(65535),w=new Array(65535),h=new Array(64),g=new Array(64),y=[],m=0,A=7,s=new Array(64),l=new Array(64),p=new Array(64),E=new Array(256),j=new Array(2048),D=[0,1,5,6,14,15,27,28,2,4,7,13,16,26,29,42,3,8,12,17,25,30,41,43,9,11,18,24,31,40,44,53,10,19,23,32,39,45,52,54,20,22,33,38,46,51,55,60,21,34,37,47,50,56,59,61,35,36,48,49,57,58,62,63],I=[0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0],B=[0,1,2,3,4,5,6,7,8,9,10,11],C=[0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125],M=[1,2,3,0,4,17,5,18,33,49,65,6,19,81,97,7,34,113,20,50,129,145,161,8,35,66,177,193,21,82,209,240,36,51,98,114,130,9,10,22,23,24,25,26,37,38,39,40,41,42,52,53,54,55,56,57,58,67,68,69,70,71,72,73,74,83,84,85,86,87,88,89,90,99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,131,132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,162,163,164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,194,195,196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,225,226,227,228,229,230,231,232,233,234,241,242,243,244,245,246,247,248,249,250],b=[0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0],x=[0,1,2,3,4,5,6,7,8,9,10,11],T=[0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119],G=[0,1,2,3,17,4,5,33,49,6,18,65,81,7,97,113,19,34,50,129,8,20,66,145,161,177,193,9,35,51,82,240,21,98,114,209,10,22,36,52,225,37,241,23,24,25,26,38,39,40,41,42,53,54,55,56,57,58,67,68,69,70,71,72,73,74,83,84,85,86,87,88,89,90,99,100,101,102,103,104,105,106,115,116,117,118,119,120,121,122,130,131,132,133,134,135,136,137,138,146,147,148,149,150,151,152,153,154,162,163,164,165,166,167,168,169,170,178,179,180,181,182,183,184,185,186,194,195,196,197,198,199,200,201,202,210,211,212,213,214,215,216,217,218,226,227,228,229,230,231,232,233,234,242,243,244,245,246,247,248,249,250];function J(r,e){for(var n=0,o=0,a=new Array,t=1;t<=16;t++){for(var f=1;f<=r[t];f++)a[e[o]]=[],a[e[o]][0]=n,a[e[o]][1]=t,o++,n++;n*=2}return a}function P(r){for(var e=r[0],n=r[1]-1;n>=0;)e&1<<n&&(m|=1<<A),n--,--A<0&&(255==m?(S(255),S(0)):S(m),A=7,m=0)}function S(r){y.push(r)}function _(r){S(r>>8&255),S(255&r)}function F(r,e,n,o,a){for(var t,f=a[0],i=a[240],v=function(r,e){var n,o,a,t,f,i,v,d,u,c,w=0;for(u=0;u<8;++u){n=r[w],o=r[w+1],a=r[w+2],t=r[w+3],f=r[w+4],i=r[w+5],v=r[w+6];var g=n+(d=r[w+7]),y=n-d,m=o+v,A=o-v,s=a+i,l=a-i,p=t+f,E=t-f,j=g+p,D=g-p,I=m+s,B=m-s;r[w]=j+I,r[w+4]=j-I;var C=.707106781*(B+D);r[w+2]=D+C,r[w+6]=D-C;var M=.382683433*((j=E+l)-(B=A+y)),b=.5411961*j+M,x=1.306562965*B+M,T=.707106781*(I=l+A),G=y+T,J=y-T;r[w+5]=J+b,r[w+3]=J-b,r[w+1]=G+x,r[w+7]=G-x,w+=8}for(w=0,u=0;u<8;++u){n=r[w],o=r[w+8],a=r[w+16],t=r[w+24],f=r[w+32],i=r[w+40],v=r[w+48];var P=n+(d=r[w+56]),S=n-d,_=o+v,F=o-v,N=a+i,U=a-i,k=t+f,q=t-f,z=P+k,H=P-k,K=_+N,L=_-N;r[w]=z+K,r[w+32]=z-K;var O=.707106781*(L+H);r[w+16]=H+O,r[w+48]=H-O;var Q=.382683433*((z=q+U)-(L=F+S)),R=.5411961*z+Q,V=1.306562965*L+Q,W=.707106781*(K=U+F),X=S+W,Y=S-W;r[w+40]=Y+R,r[w+24]=Y-R,r[w+8]=X+V,r[w+56]=X-V,w++}for(u=0;u<64;++u)c=r[u]*e[u],h[u]=c>0?c+.5|0:c-.5|0;return h}(r,e),d=0;d<64;++d)g[D[d]]=v[d];var u=g[0]-n;n=g[0],0==u?P(o[0]):(P(o[w[t=32767+u]]),P(c[t]));for(var y=63;y>0&&0==g[y];y--);if(0==y)return P(f),n;for(var m,A=1;A<=y;){for(var s=A;0==g[A]&&A<=y;++A);var l=A-s;if(l>=16){m=l>>4;for(var p=1;p<=m;++p)P(i);l&=15}t=32767+g[A],P(a[(l<<4)+w[t]]),P(c[t]),A++}return 63!=y&&P(f),n}function N(r){if(r<=0&&(r=1),r>100&&(r=100),t!=r){(function(r){for(var e=[16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99],n=0;n<64;n++){var o=f((e[n]*r+50)/100);o<1?o=1:o>255&&(o=255),i[D[n]]=o}for(var a=[17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],t=0;t<64;t++){var c=f((a[t]*r+50)/100);c<1?c=1:c>255&&(c=255),v[D[t]]=c}for(var w=[1,1.387039845,1.306562965,1.175875602,1,.785694958,.5411961,.275899379],h=0,g=0;g<8;g++)for(var y=0;y<8;y++)d[h]=1/(i[D[h]]*w[g]*w[y]*8),u[h]=1/(v[D[h]]*w[g]*w[y]*8),h++})(r<50?Math.floor(5e3/r):Math.floor(200-2*r)),t=r}}this.encode=function(r,t){var f;(new Date).getTime();t&&N(t),y=new Array,m=0,A=7,_(65496),_(65504),_(16),S(74),S(70),S(73),S(70),S(0),S(1),S(1),S(0),_(1),_(1),S(0),S(0),void 0!==(f=r.comments)&&f.constructor===Array&&f.forEach(r=>{if("string"==typeof r){_(65534);var e,n=r.length;for(_(n+2),e=0;e<n;e++)S(r.charCodeAt(e))}}),function(r){if(r){_(65505),69===r[0]&&120===r[1]&&105===r[2]&&102===r[3]?_(r.length+2):(_(r.length+5+2),S(69),S(120),S(105),S(102),S(0));for(var e=0;e<r.length;e++)S(r[e])}}(r.exifBuffer),function(){_(65499),_(132),S(0);for(var r=0;r<64;r++)S(i[r]);S(1);for(var e=0;e<64;e++)S(v[e])}(),function(r,e){_(65472),_(17),S(8),_(e),_(r),S(3),S(1),S(17),S(0),S(2),S(17),S(1),S(3),S(17),S(1)}(r.width,r.height),function(){_(65476),_(418),S(0);for(var r=0;r<16;r++)S(I[r+1]);for(var e=0;e<=11;e++)S(B[e]);S(16);for(var n=0;n<16;n++)S(C[n+1]);for(var o=0;o<=161;o++)S(M[o]);S(1);for(var a=0;a<16;a++)S(b[a+1]);for(var t=0;t<=11;t++)S(x[t]);S(17);for(var f=0;f<16;f++)S(T[f+1]);for(var i=0;i<=161;i++)S(G[i])}(),_(65498),_(12),S(3),S(1),S(0),S(2),S(17),S(3),S(17),S(0),S(63),S(0);var c=0,w=0,h=0;m=0,A=7,this.encode.displayName="_encode_";for(var g,E,D,J,U,k,q,z,H,K=r.data,L=r.width,O=r.height,Q=4*L,R=0;R<O;){for(g=0;g<Q;){for(k=U=Q*R+g,q=-1,z=0,H=0;H<64;H++)k=U+(z=H>>3)*Q+(q=4*(7&H)),R+z>=O&&(k-=Q*(R+1+z-O)),g+q>=Q&&(k-=g+q-Q+4),E=K[k++],D=K[k++],J=K[k++],s[H]=(j[E]+j[D+256|0]+j[J+512|0]>>16)-128,l[H]=(j[E+768|0]+j[D+1024|0]+j[J+1280|0]>>16)-128,p[H]=(j[E+1280|0]+j[D+1536|0]+j[J+1792|0]>>16)-128;c=F(s,d,c,e,o),w=F(l,u,w,n,a),h=F(p,u,h,n,a),g+=32}R+=8}if(A>=0){var V=[];V[1]=A+1,V[0]=(1<<A+1)-1,P(V)}return _(65497),"undefined"==typeof module?new Uint8Array(y):Buffer.from(y)},function(){(new Date).getTime();r||(r=50),function(){for(var r=String.fromCharCode,e=0;e<256;e++)E[e]=r(e)}(),e=J(I,B),n=J(b,x),o=J(C,M),a=J(T,G),function(){for(var r=1,e=2,n=1;n<=15;n++){for(var o=r;o<e;o++)w[32767+o]=n,c[32767+o]=[],c[32767+o][1]=n,c[32767+o][0]=o;for(var a=-(e-1);a<=-r;a++)w[32767+a]=n,c[32767+a]=[],c[32767+a][1]=n,c[32767+a][0]=e-1+a;r<<=1,e<<=1}}(),function(){for(var r=0;r<256;r++)j[r]=19595*r,j[r+256|0]=38470*r,j[r+512|0]=7471*r+32768,j[r+768|0]=-11059*r,j[r+1024|0]=-21709*r,j[r+1280|0]=32768*r+8421375,j[r+1536|0]=-27439*r,j[r+1792|0]=-5329*r}(),N(r),(new Date).getTime()}()}function encode(r,e){return void 0===e&&(e=50),{data:new JPEGEncoder(e).encode(r,e),width:r.width,height:r.height}}function getImageDataFromImage(r){var e="string"==typeof r?document.getElementById(r):r,n=document.createElement("canvas");n.width=e.width,n.height=e.height;var o=n.getContext("2d");return o.drawImage(e,0,0),o.getImageData(0,0,n.width,n.height)}

// Encode RGB pixels as a JPEG at the given quality (1-100).
function encodeJpeg(rgb, w, h, quality) {
  const rgba = Buffer.allocUnsafe(w * h * 4);
  for (let i = 0, j = 0; j < rgba.length; i += 3, j += 4) {
    rgba[j] = rgb[i];
    rgba[j + 1] = rgb[i + 1];
    rgba[j + 2] = rgb[i + 2];
    rgba[j + 3] = 255;
  }
  const out = encode({ data: rgba, width: w, height: h }, quality);
  return Buffer.from(out.data);
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
  const srcJpeg = isJpeg(buf);
  let img = decodePng(buf);
  if (!img && srcJpeg) img = decodeJpegFull(buf);
  if (!img) return null; // WebP and other formats: cannot shrink without a codec

  // JPEG sources are re-encoded as JPEG: a lossless PNG of a photo is several
  // times larger and would force a much harsher resolution cut to stay under the
  // body limit. PNG sources keep their exact pixels.
  const encodeOut = (rgb, w, h) =>
    srcJpeg ? encodeJpeg(rgb, w, h, DOWNSCALE_JPEG_QUALITY) : encodePng(rgb, w, h);
  const outMime = srcJpeg ? "image/jpeg" : "image/png";

  const px = img.w * img.h;
  const longSide = Math.max(img.w, img.h);
  const overBytes = b64.length > DOWNSCALE_TARGET_B64;
  const overPixels = px > DOWNSCALE_MAX_PIXELS || longSide > DOWNSCALE_MAX_SIDE;
  if (!overBytes && !overPixels) return null; // already fine, don't touch it

  let scale = Math.min(1, Math.sqrt(DOWNSCALE_MAX_PIXELS / px));
  if (longSide * scale > DOWNSCALE_MAX_SIDE) scale = DOWNSCALE_MAX_SIDE / longSide;

  // Shrink until the encoded payload fits. The pixel target above is the goal;
  // the base64 target is only a hard ceiling, so step down gently to avoid
  // collapsing the image.
  let nw = Math.max(1, Math.round(img.w * scale));
  let nh = Math.max(1, Math.round(img.h * scale));
  let outBuf = null, outB64 = "";
  for (let attempt = 0; attempt < 8; attempt++) {
    const rgb = resizeBox(img, nw, nh);
    outBuf = encodeOut(rgb, nw, nh);
    outB64 = outBuf.toString("base64");
    if (outB64.length <= DOWNSCALE_TARGET_B64) break;
    if (nw <= 32 || nh <= 32) break;
    // Area scales with the square of the side, so shrink by the square root of
    // the overshoot, but never by more than 30% in one step.
    const f = Math.max(0.7, Math.min(0.9, Math.sqrt(DOWNSCALE_TARGET_B64 / outB64.length) * 0.95));
    nw = Math.max(1, Math.round(nw * f));
    nh = Math.max(1, Math.round(nh * f));
  }
  // Accept whenever the image actually got smaller in pixels; the base64 ceiling
  // above still keeps the request under the backend body limit.
  if (!outB64 || nw * nh >= px) return null;

  return {
    url: `data:${outMime};base64,${outB64}`,
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

  // Fail fast only when an ATTACHMENT alone blows the body cap: trimming history
  // cannot help there, and the backend's own 413 blames the conversation. A body
  // that is large because of text is different -- compaction really can fix it,
  // so those requests are still forwarded and left to the normal retry path.
  const media = largestMediaBytes(upstreamBody);
  const bodyJson = JSON.stringify(upstreamBody);
  const bodyChars = bodyJson.length;
  const bodyBytes = Buffer.byteLength(bodyJson);
  const attachmentDominates = media.bytes > REQUEST_BODY_LIMIT_BYTES * 0.5;
  if (bodyChars > REQUEST_BODY_LIMIT_CHARS && attachmentDominates) {
    const overBy = (bodyChars - REQUEST_BODY_LIMIT_CHARS).toLocaleString("en-US");
    const mediaMb = (media.bytes / 1048576).toFixed(2);
    const tried = shrunk.length
      ? ` The proxy already downscaled ${shrunk.length} image(s) and the body is still over.`
      : ` The proxy could not downscale this attachment (only PNG and JPEG can be resized).`;
    return json(res, 413, openAIError(
      `Request body is ${bodyChars.toLocaleString("en-US")} characters, over the backend's ` +
      `${REQUEST_BODY_LIMIT_CHARS.toLocaleString("en-US")}-character limit (${overBy} too many). ` +
      `Largest attachment: ${mediaMb} MB of base64.` + tried +
      ` This is a payload-size cap, not a context-length problem — dropping history cannot help. ` +
      `Convert the image to PNG or JPEG (both auto-downscaled), or resize it to <=1568px.`,
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
