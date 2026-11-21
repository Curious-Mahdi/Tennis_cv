// End-to-end browser test of the Tennis Vision viewer via the Chrome DevTools Protocol.
// Real mouse/keyboard input is dispatched through CDP, so the actual UI handlers are exercised.
//   python web/serve.py &  then:  node tests/web/web_test.mjs [http://localhost:8000]
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL_ = process.argv[2] || "http://localhost:8000";
const OUT = process.env.SHOTS_DIR || "output/web_shots";
mkdirSync(OUT, { recursive: true });
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const chrome = spawn(process.env.CHROME || "C:/Program Files/Google/Chrome/Application/chrome.exe", [
  "--headless=new", `--remote-debugging-port=${PORT}`, "--autoplay-policy=no-user-gesture-required",
  "--hide-scrollbars", "--no-first-run", "--mute-audio", `--user-data-dir=${mkdtempSync(join(tmpdir(), "tvdemo-"))}`, "about:blank",
], { stdio: "ignore" });

let ws, id = 0;
const pending = new Map(), errors = [];
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
async function js(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
async function shot(name, full = false) {
  const p = { format: "png" };
  if (full) { const m = await send("Page.getLayoutMetrics"); p.clip = { x: 0, y: 0, width: m.cssContentSize.width, height: m.cssContentSize.height, scale: 1 }; p.captureBeyondViewport = true; }
  const r = await send("Page.captureScreenshot", p);
  writeFileSync(`${OUT}/${name}.png`, Buffer.from(r.data, "base64"));
}
async function click(x, y) {
  for (const type of ["mousePressed", "mouseReleased"]) await send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1 });
}
async function key(code, keyName) {
  await send("Input.dispatchKeyEvent", { type: "keyDown", code, key: keyName, windowsVirtualKeyCode: code === "ArrowRight" ? 39 : code === "ArrowLeft" ? 37 : 32 });
  await send("Input.dispatchKeyEvent", { type: "keyUp", code, key: keyName });
}
const rect = (sel) => js(`(() => { const r=document.querySelector(${JSON.stringify(sel)}).getBoundingClientRect(); return {x:r.left,y:r.top,w:r.width,h:r.height}; })()`);
const scrollTo = (sel) => js(`document.querySelector(${JSON.stringify(sel)}).scrollIntoView({block:'start'})`);

const PROBE = `(async () => {
  const v = document.getElementById('video');
  if (!window.__D) window.__D = await (await fetch('data/demo_frames.json')).json();
  const D = window.__D, fps = D.demo.fps, N = D.frames.length;
  const shown = +document.getElementById('frameReadout').textContent.match(/frame (\\d+)/)[1];
  const fromClock = Math.min(N - 1, Math.floor(v.currentTime * fps + 1e-3));
  const rec = D.frames[shown];
  const marks = [...document.querySelectorAll('#court [data-layer=players] [data-pid]')].map(g => {
    const c = g.querySelector('circle[r="0.42"]'); return { pid: g.dataset.pid, x: +c.getAttribute('cx'), y: +c.getAttribute('cy') }; });
  const exp = ['1','2'].filter(p => rec.players[p].court && rec.players[p].court.status === 'measured').map(p => ({ pid: p, xy: rec.players[p].court.xy_m_smoothed }));
  const mism = exp.filter(e => { const m = marks.find(k => k.pid === e.pid); return !m || Math.abs(m.x - e.xy[0]) > 1e-6 || Math.abs(m.y - e.xy[1]) > 1e-6; }).length + marks.filter(m => !exp.find(e => e.pid === m.pid)).length;
  const bm = document.querySelector('#court [data-layer=ball] [data-ball]');
  const fl = rec.ball.flight;
  const ballOk = fl ? !!bm && Math.abs(+bm.getAttribute('cx') - fl.xy_m[0]) < 1e-6 && Math.abs(+bm.getAttribute('cy') - fl.xy_m[1]) < 1e-6 : !bm;
  const fe = D.ball_flight.events.find((e) => e.frame === shown);
  const presented = v.dataset.presentedFrame == null ? null : +v.dataset.presentedFrame;
  return { t: +v.currentTime.toFixed(4), paused: v.paused, shown, fromClock, presented, players: marks.length, mism,
           courtBall: !!bm, flight: !!fl, ballOk, solid: bm ? bm.getAttribute('r') === '0.3' : null,
           event: fe ? fe.type + ':' + fe.status : null, cleanStatus: rec.ball.clean.status,
           panel: document.querySelector('#panel h3')?.textContent || '' };
})()`;

const results = [];
const check = (name, ok, detail) => { results.push({ name, ok, detail }); console.log(ok ? "PASS" : "FAIL", name, JSON.stringify(detail).slice(0, 260)); };

try {
  for (let i = 0; i < 40; i++) { try { await fetch(`http://127.0.0.1:${PORT}/json/version`); break; } catch { await sleep(250); } }
  const target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(URL_)}`, { method: "PUT" })).json();
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) { const p = pending.get(msg.id); pending.delete(msg.id); msg.error ? p.rej(new Error(msg.error.message)) : p.res(msg.result); }
    if (msg.method === "Runtime.exceptionThrown") errors.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") errors.push(msg.params.args.map((a) => a.value || a.description).join(" "));
  };
  await send("Runtime.enable"); await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: URL_ });
  let ready = false;
  for (let i = 0; i < 80 && !ready; i++) { await sleep(250); ready = await js(`document.getElementById('video').readyState >= 2 && document.getElementById('loading').hidden`).catch(() => false); }
  check("page, video and data load", ready, {});
  await sleep(1200);   // web fonts
  await shot("01_landing");
  await scrollTo("#analysis");
  await sleep(300);
  let p = await js(PROBE);
  check("initial: rendered frame == video clock", p.shown === p.fromClock && p.mism === 0, p);
  check("initial panel is the clip overview", /seconds/.test(p.panel), p);

  // playback sync
  await js(`document.getElementById('video').play()`);
  const samples = [];
  for (let i = 0; i < 14; i++) { await sleep(230); samples.push(await js(PROBE)); }
  await js(`document.getElementById('video').pause()`);
  await sleep(300);
  // Strict: the court shows exactly the frame the video is presenting (requestVideoFrameCallback).
  check("playback: rendered frame == frame on screen, every sample", samples.every((s) => s.presented === s.shown), { d: samples.map((s) => s.shown - s.presented) });
  // Loose: currentTime runs ahead of the presented frame while frames are decoded.
  check("playback: within 2 frames of video.currentTime", samples.every((s) => Math.abs(s.fromClock - s.shown) <= 2), { lags: samples.map((s) => s.fromClock - s.shown) });
  check("playback: frames advance", samples.at(-1).shown > samples[0].shown, { f: samples.map((s) => s.shown) });
  check("playback: player markers equal JSON every sample", samples.every((s) => s.mism === 0), {});
  check("playback: court ball drawn exactly where the JSON has a flight estimate, at its position",
        samples.every((s) => s.ballOk), { s: samples.map((s) => [s.shown, s.flight, s.courtBall]) });
  check("playback: the ball moves on the court (several frames with an estimate)", samples.filter((s) => s.flight).length >= 5, {});

  // timeline click-to-seek (real mouse)
  const tl = await rect("#timeline");
  for (const f of [30, 120]) {
    await click(tl.x + ((f + 0.5) / 150) * tl.w, tl.y + 20);
    await sleep(350);
    p = await js(PROBE);
    check(`timeline click seeks to frame ${f}`, p.shown === f && p.fromClock === f && p.mism === 0, p);
  }
  // frame stepping (real keys)
  await key("ArrowRight", "ArrowRight"); await sleep(250);
  p = await js(PROBE);
  check("→ steps one frame (120 → 121)", p.shown === 121 && p.fromClock === 121, p);
  await key("ArrowLeft", "ArrowLeft"); await key("ArrowLeft", "ArrowLeft"); await sleep(250);
  p = await js(PROBE);
  check("← ← steps back two frames (121 → 119)", p.shown === 119 && p.fromClock === 119, p);

  // measured bounce (frame 44) is drawn solid
  await click(tl.x + ((44 + 0.5) / 150) * tl.w, tl.y + 20);
  await sleep(350);
  p = await js(PROBE);
  check("frame 44: measured bounce, solid ball marker at the JSON position", p.event === "bounce:measured" && p.solid === true && p.ballOk, p);
  // event selection by clicking the event marker on the timeline (estimated bounce at 145)
  await click(tl.x + ((145 + 0.5) / 150) * tl.w, tl.y + 78);
  await sleep(400);
  p = await js(PROBE);
  check("event marker click selects the estimated bounce at 145", p.shown === 145 && p.panel === "Bounce" && p.solid === false && p.ballOk, p);
  await shot("02_bounce_selected");
  // previous event via panel button
  await js(`[...document.querySelectorAll('#panel [data-ev-step]')][0].click()`);
  await sleep(400);
  p = await js(PROBE);
  check("panel ‘previous event’ → contact at 106", p.shown === 106 && /^Contact/.test(p.panel), p);
  await shot("03_contact_selected");

  // player selection by clicking the court marker (real mouse)
  const m = await js(`(() => { const c=document.querySelector('#court [data-pid="2"] circle[r="0.42"]').getBoundingClientRect(); return {x:c.left+c.width/2,y:c.top+c.height/2}; })()`);
  await click(m.x, m.y);
  await sleep(300);
  p = await js(PROBE);
  check("court marker click selects Player 2", p.panel === "Player 2", p);
  await shot("04_player2_selected");
  // missing detection is explained, not drawn
  await click(tl.x + ((5 + 0.5) / 150) * tl.w, tl.y + 20);
  await sleep(400);
  p = await js(PROBE);
  const callout = await js(`document.querySelector('#panel .callout')?.textContent || ''`);
  check("frame 5: P2 missing → no marker, panel explains", p.players === 1 && /No reliable position/.test(callout), { p, callout: callout.slice(0, 80) });
  await shot("05_player2_missing");

  // the former 'teleport' frames
  for (const f of [74, 75]) {
    await click(tl.x + ((f + 0.5) / 150) * tl.w, tl.y + 20);
    await sleep(350);
    p = await js(PROBE);
    check(`frame ${f}: video ball ${f === 74 ? "rejected" : "unavailable"}; court ball only from the flight estimate`, p.ballOk && p.cleanStatus === (f === 74 ? "rejected" : "unavailable"), p);
  }
  await shot("06_frame74_rejected");
  // frames contradicted by the video (just after P1's contact) show no court ball
  await click(tl.x + ((57 + 0.5) / 150) * tl.w, tl.y + 20);
  await sleep(350);
  p = await js(PROBE);
  check("frame 57: estimate hidden where the video contradicts it", !p.flight && !p.courtBall, p);
  // click the ball on the court -> ball panel
  await click(tl.x + ((120 + 0.5) / 150) * tl.w, tl.y + 20);
  await sleep(350);
  const bpos = await js(`(() => { const c=document.querySelector('#court [data-ball]').getBoundingClientRect(); return {x:c.left+c.width/2,y:c.top+c.height/2}; })()`);
  await click(bpos.x, bpos.y);
  await sleep(300);
  p = await js(PROBE);
  const bpanel = await js(`document.getElementById('panel').textContent`);
  check("court ball click opens the ball panel, labelled as an estimate", p.panel === "Ball" && /Estimated/.test(bpanel) && /Uncertainty/.test(bpanel), { panel: p.panel });
  await shot("06b_ball_selected");

  // Escape returns to overview; heatmap toggle draws cells
  await key("Escape", "Escape"); await sleep(200);
  p = await js(PROBE);
  check("Esc returns to clip overview", /seconds/.test(p.panel), p);
  await js(`document.getElementById('heatToggle').click()`); await sleep(200);
  const cells = await js(`document.querySelectorAll('#court [data-layer=heat] rect').length`);
  check("occupancy layer draws cells from analytics", cells > 0, { cells });
  await shot("07_heatmap");
  await js(`document.getElementById('heatToggle').click()`);

  // charts: playhead synced, click seeks
  await scrollTo("#movement"); await sleep(300);
  const ph = await js(`document.querySelector('#chartDistance .ph').getAttribute('x1')`);
  const hit = await rect("#chartDistance .hit");
  await click(hit.x + hit.w * 0.5, hit.y + hit.h / 2);
  await sleep(500);
  p = await js(PROBE);
  const ph2 = await js(`document.querySelector('#chartDistance .ph').getAttribute('x1')`);
  check("chart click seeks the video and moves the playhead", Math.abs(p.shown - 74) <= 1 && ph !== ph2, { shown: p.shown, ph, ph2 });
  await scrollTo("#movement"); await sleep(200);
  await shot("08_movement");
  await scrollTo("#method"); await sleep(200);
  await shot("09_method");

  // loop
  await js(`(() => { const v=document.getElementById('video'); v.currentTime=v.duration-0.05; return v.play(); })()`);
  await sleep(900);
  p = await js(PROBE);
  check("loops to start", p.shown < 40 && !p.paused, p);
  await js(`document.getElementById('video').pause()`);

  // mobile
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
  await sleep(600);
  const ov = await js(`document.documentElement.scrollWidth - document.documentElement.clientWidth`);
  check("mobile 390 px: no horizontal scroll", ov <= 0, { ov });
  await js(`window.scrollTo(0,0)`); await sleep(200); await shot("10_mobile_top");
  await scrollTo("#analysis"); await sleep(300); await shot("11_mobile_analysis");
  check("no console errors", errors.length === 0, errors);
} catch (e) {
  check("harness", false, String(e.stack || e));
} finally {
  writeFileSync(`${OUT}/results.json`, JSON.stringify(results, null, 2));
  const failed = results.filter((c) => !c.ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed`);
  try { ws?.close(); } catch {}
  chrome.kill();
  process.exit(failed ? 1 : 0);
}
