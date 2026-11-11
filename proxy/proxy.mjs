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

function statsSnapshot() {
  const secs = Math.floor((Date.now() - STARTED_AT) / 1000);
  return {
    status: {
      signed_in: Boolean(loadToken()),
      backend: BASE,
      client_version: getClientVersion(),
      session: SESSION_ID.slice(0, 14) + "…",
      default_model: DEFAULT_MODEL,
      reasoning_effort: REASONING_EFFORT,
      uptime_seconds: secs,
      mcp_servers: [...mcp.servers.entries()].map(([name, { tools }]) => ({ name, tools: tools.length })),
    },
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

function upstreamHeaders(token) {
  return {
    authorization: `Bearer ${token}`,
    "x-session-id": SESSION_ID,
    "x-client-version": getClientVersion(),
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
  return catalogCache.models ?? [{ id: FALLBACK_MODEL, name: FALLBACK_MODEL }];
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
//  - max_tokens: default to MAX_OUTPUT_TOKENS when the client sends none
//    (or raise smaller values, with FREEPI_PROXY_FORCE=1)
// Message content is never touched: the full context the client sends is
// forwarded as-is (the backend window is CONTEXT_WINDOW).
function applyMaxima(body) {
  const out = { ...body };
  return fetchModelSpecs().then((specs) => {
    const spec = specFor(out.model, specs);
    const maxOut = spec?.maxOutput ?? MAX_OUTPUT_TOKENS;
    if (REASONING_EFFORT !== "client") {
      if (out.reasoning_effort == null || FORCE_MAXIMA) out.reasoning_effort = REASONING_EFFORT;
    }
    if (out.max_tokens == null) out.max_tokens = maxOut;
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
async function upstreamCall(token, upstreamBody) {
  return fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: upstreamHeaders(token),
    body: JSON.stringify(upstreamBody),
  });
}

// When MCP tools are active, run a server-side agentic loop: execute
// mcp__* tool calls ourselves and feed results back until the model answers
// or calls a client-owned tool (which is returned to the client untouched).
async function runWithMcp(token, upstreamBody, wantsStream, res) {
  const messages = [...upstreamBody.messages];
  let final = null;
  for (let iter = 0; iter < MCP_MAX_ITERATIONS; iter++) {
    let resp;
    try {
      resp = await upstreamCall(token, { ...upstreamBody, messages, stream: false });
    } catch (err) {
      return json(res, 502, openAIError(`Cannot reach free-pi backend: ${err.message}`, "api_error"));
    }
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      return json(res, resp.status === 401 || resp.status === 429 ? resp.status : 502,
        openAIError(`free-pi backend ${resp.status}: ${text.slice(0, 1000)}`, "api_error"));
    }
    const completion = await resp.json();
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
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
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
}
var tokenBuilt = false;
function renderToken(s) {
  var signedIn = s.status && s.status.signed_in;
  if (tokenBuilt) { var line = document.getElementById("toksignin"); if (line) line.innerHTML = "<span>Signed in</span><span class=\\"" + (signedIn ? "ok" : "bad") + "\\">" + (signedIn ? "yes" : "NO") + "</span>"; return; } // built once - never rebuild, so auto-refresh cannot wipe typed input
  document.getElementById("token").innerHTML =
    "<div id=\\"toksignin\\" class=\\"kv\\"><span>Signed in</span><span class=\\"" + (signedIn ? "ok" : "bad") + "\\">" + (signedIn ? "yes" : "NO") + "</span></div>" +
    "<div id=\\"tokacct\\" class=\\"muted\\" style=\\"font-size:12px;margin:4px 0\\"></div>" +
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
function refresh() { fetch("/stats", { headers: hdrs() }).then(function(r){return r.json();}).then(render).catch(function(){}); }
refresh(); setInterval(refresh, 5000);
</script></body></html>`;

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

  // MCP servers connected? Inject their tools and run the server-side loop.
  const mcpTools = useMcp && MCP_MODE !== "off" && mcp.connected() ? mcp.toOpenAITools() : [];
  if (mcpTools.length) {
    const have = new Set((upstreamBody.tools ?? []).map((t) => t.function?.name).filter(Boolean));
    upstreamBody.tools = [...(upstreamBody.tools ?? []), ...mcpTools.filter((t) => !have.has(t.function.name))];
    return runWithMcp(token, upstreamBody, wantsStream, res);
  }

  const sentVersion = getClientVersion();
  let upstream;
  try {
    upstream = await fetch(`${BASE}/v1/chat/completions`, {
      method: "POST",
      headers: upstreamHeaders(token),
      body: JSON.stringify(upstreamBody),
    });
  } catch (err) {
    return json(res, 502, openAIError(`Cannot reach free-pi backend: ${err.message}`, "api_error"));
  }

  // Expired sign-in or outdated client version: refresh credentials and ask
  // the backend for its current minimum/latest version, then retry once.
  if (upstream.status === 401 || upstream.status === 426) {
    credCache = { token: null, mtime: 0 };
    if (upstream.status === 426) await refreshClientVersion({ force: true });
    const fresh = loadToken();
    const freshVersion = getClientVersion();
    if (fresh && (fresh !== token || freshVersion !== sentVersion)) {
      upstream = await fetch(`${BASE}/v1/chat/completions`, {
        method: "POST",
        headers: upstreamHeaders(fresh),
        body: JSON.stringify(upstreamBody),
      });
    }
  }

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => "");
    let detail = text.slice(0, 2000);
    try {
      const parsed = JSON.parse(text);
      detail = parsed?.error?.message ?? parsed?.detail ?? JSON.stringify(parsed).slice(0, 2000);
    } catch {}
    const hint =
      upstream.status === 401
        ? " (free-pi sign-in expired — run `npx free-pi-cli@latest` to sign in again)"
        : upstream.status === 426
          ? ` (proxy sent client version ${sentVersion} and could not self-heal; run \`npx free-pi-cli@latest\` once so the CLI records its new version, or set FREEPI_CLIENT_VERSION, or check connectivity to ${BASE})`
          : upstream.status === 429
            ? " (daily free-pi allowance spent — /usage in free-pi, or /buy-credits)"
            : "";
    return json(res, upstream.status === 401 || upstream.status === 429 ? upstream.status : 502,
      openAIError(`free-pi backend ${upstream.status}: ${detail}${hint}`, "api_error"));
  }

  if (!wantsStream) {
    const data = await upstream.text();
    try {
      const parsed = JSON.parse(data);
      recordUsage(parsed.model ?? upstreamBody.model, parsed.usage);
    } catch {}
    res.writeHead(upstream.status, { "content-type": "application/json" });
    return res.end(data);
  }

  // Streaming: pipe SSE through untouched so tool_call deltas stay intact;
  // a tap scans events to record usage from the final chunk.
  res.writeHead(200, {
    "content-type": upstream.headers.get("content-type") || "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
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
    try {
      const parsed = JSON.parse(token);
      if (typeof parsed === "string") token = parsed.trim();
      else {
        body = parsed ?? {};
        if (typeof parsed?.token === "string") token = parsed.token.trim();
        else if (typeof parsed?.jwt === "string") token = parsed.jwt.trim();
      }
    } catch { /* plain token string */ }
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
        return json(res, 401, openAIError(`Token rejected by free-pi backend (HTTP ${check.status}). Not saved.`, "authentication_error"));
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
        ? `Token validated and saved. GitHub secret ${sync.secret} updated.`
        : sync.attempted
          ? `Token validated and saved, but GitHub sync failed: ${sync.detail}`
          : "Token validated and saved (GitHub sync not configured).",
    });
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
  if (MCP_MODE !== "off") {
    mcp.connectAll().then((n) => console.log(`  mcp          ${n} server(s) connected${n ? ": " + [...mcp.servers.keys()].join(", ") : " (no mcp.json found)"}`)).catch(() => {});
  } else {
    console.log("  mcp          disabled (MCP_MODE=off)");
  }
});
