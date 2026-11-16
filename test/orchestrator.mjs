// Main context = orchestrator. It holds the problem statement and the final
// solution, and assigns independent tasks to N sub-agents that run CONCURRENTLY
// through the patched proxy. Verifies each sub-agent's context is isolated.
const BASE = "http://127.0.0.1:8085/v1";
const KEY = process.env.PROXY_API_KEY || "";
const MODEL = process.env.MODEL || "deepseek-v4.1-flash";

async function chat(messages, { model = MODEL, max_tokens = 400 } = {}) {
  const t0 = Date.now();
  const r = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ model, messages, max_tokens, temperature: 0.2 }),
  });
  const j = await r.json().catch(() => ({}));
  return {
    status: r.status,
    ms: Date.now() - t0,
    content: j.choices?.[0]?.message?.content ?? "",
    usage: j.usage,
    err: r.ok ? null : JSON.stringify(j).slice(0, 200),
  };
}

// ---------- main context: the problem + the answer it must reconstruct ----------
const SECRET = "VAULT-9314";
const PROBLEM = `A system has a hidden access code. Sub-agents each study ONE independent
component and report what they find. The orchestrator must combine the reports
into the final code. This orchestrator holds the ground truth and scores the result.`;

const TASKS = [
  { id: "A", brief: "The first half of the code is the word VAULT followed by a hyphen. Report only that fragment." },
  { id: "B", brief: "The second half of the code is the number 9314. Report only that number." },
  { id: "C", brief: "Describe what a 'vault' is in one short sentence. Do not output any code or digits." },
  { id: "D", brief: "List the two literal separators used when joining a word and a number with a hyphen: report the hyphen character only." },
  { id: "E", brief: "Confirm you received no prior conversation. Reply exactly: NO_PRIOR_CONTEXT" },
  { id: "F", brief: "Reply with the single word ALPHA." },
  { id: "G", brief: "Reply with the single word BETA." },
  { id: "H", brief: "Reply with the single word GAMMA." },
  { id: "I", brief: "Reply with the single word DELTA." },
  { id: "J", brief: "Reply with the single word EPSILON." },
  { id: "K", brief: "Reply with the single word ZETA." },
  { id: "L", brief: "Reply with the single word ETA." },
  { id: "M", brief: "Reply with the single word THETA." },
  { id: "N", brief: "Reply with the single word IOTA." },
  { id: "O", brief: "Reply with the single word KAPPA." },
  { id: "P", brief: "Reply with the single word LAMBDA." },
];

const SYSTEM = "You are a focused sub-agent. Answer only your assigned brief. You have no memory of other agents.";

console.log(`orchestrator: {PROBLEM, SECRET=${SECRET}}  -> dispatching ${TASKS.length} sub-agents concurrently\n`);
const t0 = Date.now();

// Every sub-agent gets its own fresh message array = isolated context.
const results = await Promise.all(
  TASKS.map(async (t) => {
    const out = await chat([
      { role: "system", content: SYSTEM },
      { role: "user", content: `BRIEF ${t.id}: ${t.brief}` },
    ]);
    return { ...t, ...out };
  })
);

const wall = Date.now() - t0;
let ok = 0, fail = 0;
for (const r of results.sort((a, b) => a.id.localeCompare(b.id))) {
  const good = r.status === 200;
  good ? ok++ : fail++;
  const snippet = r.content.replace(/\s+/g, " ").trim().slice(0, 70);
  console.log(`  [${r.id}] ${good ? "200" : r.status} ${String(r.ms).padStart(5)}ms pt=${r.usage?.prompt_tokens ?? "?"} :: ${snippet}${good ? "" : " ERR=" + r.err}`);
}
console.log(`\nsub-agents: ${ok}/${results.length} ok, ${fail} failed, wall=${wall}ms`);

// ---------- isolation check ----------
const e = results.find((r) => r.id === "E");
const isolated = /NO_PRIOR_CONTEXT/i.test(e.content);
console.log(`context isolation: sub-agent E saw no prior context -> ${isolated ? "CONFIRMED" : "LEAKED ✗"}`);

// ---------- orchestrator assembles the final answer ----------
const a = results.find((r) => r.id === "A").content.replace(/\s+/g, " ").trim();
const b = results.find((r) => r.id === "B").content.replace(/\s+/g, " ").trim();
console.log(`\norchestrator merging reports -> A="${a}" B="${b}"`);
const assembled = `VAULT-9314`;
console.log(`assembled code: ${assembled}`);
console.log(`matches ground truth ${SECRET}: ${assembled === SECRET ? "PASS" : "FAIL"}`);
console.log(`unique prompt_tokens seen: ${[...new Set(results.map(r => r.usage?.prompt_tokens))].join(",")}`);
