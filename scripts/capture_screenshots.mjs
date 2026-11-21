// Captures README screenshots of the web viewer WITHOUT broadcast footage: only the
// court, context panel, timeline, charts and method section are photographed. The
// broadcast video and its stills are third-party material and stay out of the repo.
//
//   python web/serve.py   (in another terminal)
//   node scripts/capture_screenshots.mjs [http://localhost:8000] [docs/public/media/web]
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL_ = process.argv[2] || "http://localhost:8000";
const OUT = process.argv[3] || "docs/public/media/web";
const CHROME = process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9341;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync(OUT, { recursive: true });

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--hide-scrollbars", "--mute-audio",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "tvshots-"))}`, "about:blank"], { stdio: "ignore" });
for (let i = 0; i < 40; i++) { try { await fetch(`http://127.0.0.1:${PORT}/json/version`); break; } catch { await sleep(250); } }
const target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const js = async (e) => (await send("Runtime.evaluate", { expression: e, awaitPromise: true, returnByValue: true })).result.value;
const box = (sels) => js(`(() => { const rs = ${JSON.stringify(sels)}.map(s => document.querySelector(s).getBoundingClientRect());
  const x = Math.min(...rs.map(r => r.left)), y = Math.min(...rs.map(r => r.top)) + scrollY;
  return { x: x - 16, y: y - 16, width: Math.max(...rs.map(r => r.right)) - x + 32, height: Math.max(...rs.map(r => r.bottom)) + scrollY - y + 32, scale: 1 }; })()`);
async function shot(name, sels) {
  const clip = await box(sels);
  const r = await send("Page.captureScreenshot", { format: "png", clip, captureBeyondViewport: true });
  writeFileSync(`${OUT}/${name}.png`, Buffer.from(r.data, "base64"));
  console.log("wrote", `${OUT}/${name}.png`, `${Math.round(clip.width)}x${Math.round(clip.height)}`);
}
const at = async (f, select) => {
  await js(`(async () => { const v = document.getElementById('video'); v.pause(); v.currentTime = (${f} + 0.5) / 29.97002997; await new Promise(r => setTimeout(r, 400)); })()`);
  if (select) { await js(select); await sleep(300); }
};

await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 2, mobile: false });
await send("Page.navigate", { url: URL_ });
for (let i = 0; i < 80; i++) { await sleep(250); if (await js(`document.getElementById('loading').hidden`)) break; }
await sleep(1500);                                            // web fonts

await at(120, `document.querySelector('#court [data-ball]').dispatchEvent(new MouseEvent('click', {bubbles: true}))`);
await shot("court_ball_estimate", [".ws-court", "#panel"]);
await at(44, `(() => { const i = [...document.querySelectorAll('#court [data-ev]')].map(g => +g.dataset.ev); document.querySelector('#court [data-ev="' + i[0] + '"]').dispatchEvent(new MouseEvent('click', {bubbles: true})); })()`);
await shot("court_measured_bounce", [".ws-court", "#panel"]);
await at(5, `document.querySelector('#court [data-pid="1"]').dispatchEvent(new MouseEvent('click', {bubbles: true}))`);
await js(`document.dispatchEvent(new KeyboardEvent('keydown', {code: 'Escape'}))`); await sleep(300);
await at(100, `document.querySelector('[data-go=p2]')?.click()`);
await at(5);
await shot("court_player_missing", [".ws-court", "#panel"]);
await js(`document.dispatchEvent(new KeyboardEvent('keydown', {code: 'Escape'}))`); await at(106);
await shot("timeline", [".timeline", ".tl-key"]);
await js(`document.getElementById('movement').scrollIntoView()`); await sleep(400);
await shot("movement", ["#movement .sec-head", "#chartDistance", "#chartDepth"]);
chrome.kill();
process.exit(0);
