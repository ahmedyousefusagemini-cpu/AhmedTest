// Minimal MCP (Model Context Protocol) client for the free-pi proxy.
// Supports stdio servers ({command, args, env}) and streamable-HTTP servers
// ({url}) — the same shapes free-pi's `mcp.json` uses. No SDK dependencies.
//
// Config files merged in order (later wins on name conflicts):
//   1. ~/.free-pi/agent/mcp.json   (shared with the free-pi CLI)
//   2. <proxy dir>/mcp.json        (proxy-specific overrides)

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";

const PROTOCOL_VERSION = "2025-03-26";
const INIT_TIMEOUT_MS = 30000;
const LIST_TIMEOUT_MS = 30000;
const CALL_TIMEOUT_MS = Number(process.env.MCP_CALL_TIMEOUT_MS || 120000);

class StdioTransport {
  constructor(name, cfg, log) {
    this.name = name;
    this.cfg = cfg;
    this.log = log;
    this.pending = new Map();
    this.buf = "";
    this.closed = false;
  }

  async start() {
    this.proc = spawn(this.cfg.command, this.cfg.args ?? [], {
      env: { ...process.env, ...(this.cfg.env ?? {}) },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (d) => this.#onData(d));
    this.proc.stderr.on("data", (d) => this.log?.(`[mcp:${this.name}] stderr: ${String(d).trim().slice(0, 300)}`));
    this.proc.on("exit", (code) => {
      this.closed = true;
      for (const { reject } of this.pending.values()) reject(new Error(`MCP server '${this.name}' exited (code ${code})`));
      this.pending.clear();
    });
    await this.#initialize();
  }

  #onData(d) {
    this.buf += d;
    let idx;
    while ((idx = this.buf.indexOf("\n")) >= 0) {
      const line = this.buf.slice(0, idx).trim();
      this.buf = this.buf.slice(idx + 1);
      if (!line) continue;
      try { this.#onMessage(JSON.parse(line)); } catch {}
    }
  }

  #onMessage(msg) {
    if (msg.id != null && this.pending.has(msg.id)) {
      const { resolve, timer } = this.pending.get(msg.id);
      clearTimeout(timer);
      this.pending.delete(msg.id);
      resolve(msg);
    }
  }

  #initialize() {
    return this.request("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "freepi-proxy", version: "1.0.0" },
    }, INIT_TIMEOUT_MS).then((resp) => {
      if (resp.error) throw new Error(`initialize failed: ${JSON.stringify(resp.error).slice(0, 200)}`);
      this.serverInfo = resp.result?.serverInfo;
      return this.notify("notifications/initialized");
    });
  }

  request(method, params, timeoutMs = CALL_TIMEOUT_MS) {
    if (this.closed) return Promise.reject(new Error(`MCP server '${this.name}' is not running`));
    const id = crypto.randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`MCP '${this.name}' ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.#send({ jsonrpc: "2.0", id, method, ...(params !== undefined ? { params } : {}) });
    });
  }

  notify(method, params) {
    this.#send({ jsonrpc: "2.0", method, ...(params !== undefined ? { params } : {}) });
  }

  #send(obj) {
    this.proc.stdin.write(JSON.stringify(obj) + "\n");
  }

  async listTools() {
    const resp = await this.request("tools/list", {}, LIST_TIMEOUT_MS);
    if (resp.error) throw new Error(`tools/list failed: ${JSON.stringify(resp.error).slice(0, 200)}`);
    return resp.result?.tools ?? [];
  }

  async callTool(name, args) {
    const resp = await this.request("tools/call", { name, arguments: args ?? {} });
    if (resp.error) throw new Error(JSON.stringify(resp.error).slice(0, 300));
    const r = resp.result ?? {};
    if (r.isError) throw new Error(textOfContent(r.content) || "MCP tool reported an error");
    return textOfContent(r.content);
  }

  async close() {
    try { this.notify("notifications/cancelled", {}) } catch {}
    try { this.proc?.kill(); } catch {}
  }
}

class HttpTransport {
  constructor(name, cfg, log) {
    this.name = name;
    this.url = cfg.url;
    this.headers = { ...(cfg.headers ?? {}) };
    this.log = log;
  }

  async #rpc(body, timeoutMs) {
    const res = await fetch(this.url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        ...this.headers,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    const sessionId = res.headers.get("mcp-session-id");
    if (sessionId) this.headers["mcp-session-id"] = sessionId;
    const ct = res.headers.get("content-type") || "";
    const text = await res.text();
    if (ct.includes("text/event-stream")) {
      // SSE: take the last data: line that parses as JSON-RPC with matching id
      let match = null;
      for (const line of text.split("\n")) {
        if (!line.startsWith("data:")) continue;
        try {
          const j = JSON.parse(line.slice(5).trim());
          if (j.id === body.id) match = j;
        } catch {}
      }
      if (!match) throw new Error(`no JSON-RPC response in SSE from ${this.url}`);
      return match;
    }
    return JSON.parse(text);
  }

  async start() {
    const resp = await this.#rpc({
      jsonrpc: "2.0", id: crypto.randomUUID(), method: "initialize",
      params: { protocolVersion: PROTOCOL_VERSION, capabilities: {}, clientInfo: { name: "freepi-proxy", version: "1.0.0" } },
    }, INIT_TIMEOUT_MS);
    if (resp.error) throw new Error(`initialize failed: ${JSON.stringify(resp.error).slice(0, 200)}`);
    this.serverInfo = resp.result?.serverInfo;
    await this.#rpc({ jsonrpc: "2.0", method: "notifications/initialized" }, INIT_TIMEOUT_MS).catch(() => {});
  }

  async listTools() {
    const resp = await this.#rpc({ jsonrpc: "2.0", id: crypto.randomUUID(), method: "tools/list" }, LIST_TIMEOUT_MS);
    if (resp.error) throw new Error(`tools/list failed: ${JSON.stringify(resp.error).slice(0, 200)}`);
    return resp.result?.tools ?? [];
  }

  async callTool(name, args) {
    const resp = await this.#rpc({ jsonrpc: "2.0", id: crypto.randomUUID(), method: "tools/call", params: { name, arguments: args ?? {} } }, CALL_TIMEOUT_MS);
    if (resp.error) throw new Error(JSON.stringify(resp.error).slice(0, 300));
    const r = resp.result ?? {};
    if (r.isError) throw new Error(textOfContent(r.content) || "MCP tool reported an error");
    return textOfContent(r.content);
  }

  async close() {}
}

function textOfContent(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((c) => (c.type === "text" ? c.text : c.type === "resource" ? JSON.stringify(c.resource ?? c) : JSON.stringify(c))).join("\n");
  }
  return String(content);
}

export class McpManager {
  constructor(log) {
    this.log = log;
    this.servers = new Map(); // name -> { transport, tools: [] }
    this.configPaths = [
      path.join(os.homedir(), ".free-pi", "agent", "mcp.json"),
      path.join(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")), "mcp.json"),
    ];
  }

  loadConfig() {
    const merged = {};
    for (const p of this.configPaths) {
      try {
        const j = JSON.parse(fs.readFileSync(p, "utf8"));
        Object.assign(merged, j);
      } catch {}
    }
    return merged;
  }

  async connectAll() {
    const cfg = this.loadConfig();
    for (const [name, serverCfg] of Object.entries(cfg)) {
      if (!serverCfg || typeof serverCfg !== "object") continue;
      try {
        const transport = serverCfg.url
          ? new HttpTransport(name, serverCfg, this.log)
          : new StdioTransport(name, serverCfg, this.log);
        await transport.start();
        const tools = await transport.listTools();
        this.servers.set(name, { transport, tools });
        this.log?.(`[mcp] connected '${name}' (${transport.serverInfo?.name ?? "unknown"}) — ${tools.length} tools`);
      } catch (err) {
        this.log?.(`[mcp] FAILED to connect '${name}': ${String(err.message).slice(0, 200)}`);
      }
    }
    return this.servers.size;
  }

  connected() { return this.servers.size > 0; }

  // OpenAI function tools for all MCP tools, namespaced mcp__<server>__<tool>
  toOpenAITools() {
    const tools = [];
    for (const [server, { tools: list }] of this.servers) {
      for (const t of list) {
        tools.push({
          type: "function",
          function: {
            name: `mcp__${server}__${t.name}`,
            description: t.description ? `[MCP:${server}] ${t.description}` : `[MCP:${server}] tool ${t.name}`,
            parameters: t.inputSchema && typeof t.inputSchema === "object" ? t.inputSchema : { type: "object", properties: {} },
          },
        });
      }
    }
    return tools;
  }

  async callTool(fqName, argsJson) {
    const m = /^mcp__([^_]+(?:__[^_]+)*?)__(.+)$/.exec(fqName);
    if (!m) throw new Error(`not an MCP tool: ${fqName}`);
    // server names may contain underscores; match against known servers first
    let server = null, tool = null;
    for (const key of this.servers.keys()) {
      if (fqName.startsWith(`mcp__${key}__`)) { server = key; tool = fqName.slice(`mcp__${key}__`.length); break; }
    }
    if (!server) { server = m[1]; tool = m[2]; }
    const entry = this.servers.get(server);
    if (!entry) throw new Error(`MCP server '${server}' is not connected`);
    let args = {};
    try { args = typeof argsJson === "string" ? JSON.parse(argsJson || "{}") : (argsJson ?? {}); } catch { args = {}; }
    return entry.transport.callTool(tool, args);
  }

  async closeAll() {
    for (const { transport } of this.servers.values()) await transport.close().catch(() => {});
    this.servers.clear();
  }
}
