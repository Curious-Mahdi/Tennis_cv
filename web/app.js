/*
 * Tennis Vision — analysis viewer.
 *
 * Sync rule: everything is drawn for the frame the VIDEO is presenting. The frame index
 * comes from requestVideoFrameCallback's mediaTime (or video.currentTime as a fallback);
 * there is no independent timer, so play, pause, seek, rate changes and looping cannot drift.
 *
 * Honesty rule: only values in data/demo_frames.json are drawn.
 *   - Players: shown only on frames where their feet were mapped; trails break at gaps.
 *   - Ball on video: the cleaned track (analytics/ball_track.py) - detected, or
 *     interpolated across <= 2 missing frames inside one consistent tracklet. Rejected
 *     detections are marked as rejected, never as the ball.
 *   - Ball on court: ONLY on the frame of a detected bounce (measured) or contact
 *     (approximate: hitter's feet). An airborne ball is not on the court plane, so it is
 *     never projected or interpolated across the court.
 */
(() => {
  "use strict";

  const W_M = 10.97, L_M = 23.76, ALLEY = 1.37, SVC = 5.48, NET = L_M / 2;
  const TRAIL = 60;        // player trail, frames
  const BALL_TRAIL = 14;   // ball trail (court and video), frames
  const C = {                       // validated palettes (dataviz validator)
    navy: { 1: "#ea5530", 2: "#2f9be0" },
    paper: { 1: "#e0502c", 2: "#1f7fc4" },
    ball: "#dff24a", rej: "#ff6d5a",
  };
  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
  const court = $("court"), tl = $("timeline"), tctx = tl.getContext("2d");

  let D, F, fps, N, srcW, srcH, EV;
  let cur = -1;
  let sel = { type: "overview" };
  const opt = { overlay: true, trails: true, heat: false, smooth: true };
  const L = {};
  const charts = [];

  // ── helpers ────────────────────────────────────────────────────────────────
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const frameFromTime = (t) => clamp(Math.floor(t * fps + 1e-3), 0, N - 1);
  const f2 = (v) => (v == null ? "—" : Number(v).toFixed(2));
  const tStr = (f) => `${(f / fps).toFixed(3)} s`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function measured(rec, pid) {
    const c = rec.players[pid] && rec.players[pid].court;
    return c && c.status === "measured" && c.xy_m ? c : null;
  }
  function pxy(rec, pid) {
    const c = measured(rec, pid);
    return c ? (opt.smooth && c.xy_m_smoothed ? c.xy_m_smoothed : c.xy_m) : null;
  }
  const side = (pid) => (D.players.side || {})[pid] || (pid === "1" ? "near" : "far");
  function depthNote(y, s) {
    const d = s === "near" ? L_M - y : y;
    if (d < 0) return `${(-d).toFixed(1)} m behind own baseline`;
    return `${d.toFixed(1)} m inside own baseline`;
  }
  function lastMeasured(pid, f) {
    for (let k = f; k >= 0; k--) if (measured(F[k], pid)) return k;
    return null;
  }
  function singlesMargin(x, y) {      // + inside the singles court, - outside (metres)
    const dx = Math.min(x - ALLEY, W_M - ALLEY - x), dy = Math.min(y, L_M - y);
    return Math.min(dx, dy);
  }

  // ── court ──────────────────────────────────────────────────────────────────
  function drawCourt() {
    const g = el("g", {}, court);
    el("rect", { x: -4.4, y: -4.0, width: 19.77, height: 32.0, fill: "#0f2551" }, g);
    el("rect", { x: 0, y: 0, width: W_M, height: L_M, fill: "#1b3a78" }, g);
    const line = (x1, y1, x2, y2, w = 0.07) => el("line", { x1, y1, x2, y2, stroke: "#f2f5fa", "stroke-width": w }, g);
    line(0, 0, W_M, 0, 0.1); line(0, L_M, W_M, L_M, 0.1); line(0, 0, 0, L_M); line(W_M, 0, W_M, L_M);
    line(ALLEY, 0, ALLEY, L_M); line(W_M - ALLEY, 0, W_M - ALLEY, L_M);
    line(ALLEY, SVC, W_M - ALLEY, SVC); line(ALLEY, L_M - SVC, W_M - ALLEY, L_M - SVC);
    line(W_M / 2, SVC, W_M / 2, L_M - SVC); line(W_M / 2, 0, W_M / 2, 0.15); line(W_M / 2, L_M - 0.15, W_M / 2, L_M);
    el("line", { x1: -0.914, y1: NET, x2: W_M + 0.914, y2: NET, stroke: "#f2f5fa", "stroke-width": 0.05, "stroke-dasharray": "0.1 0.1", opacity: 0.8 }, g);
    const lab = (t, y) => { const e = el("text", { x: W_M / 2, y, "text-anchor": "middle", fill: "#7f93bb", "font-size": 0.55, "letter-spacing": 0.12, "font-family": "Inter, sans-serif" }, g); e.textContent = t; };
    lab("FAR END", -3.4);
    lab("NEAR END · CAMERA", L_M + 3.85);
    for (const k of ["heat", "marks", "trails", "ball", "players", "sel"]) L[k] = el("g", { "data-layer": k }, court);
    court.addEventListener("click", (e) => {
      const t = e.target.closest("[data-pid],[data-ev],[data-ball]");
      if (!t) return;
      if (t.dataset.ball) select({ type: "ball" });
      else if (t.dataset.pid) select({ type: "player", pid: t.dataset.pid });
      else selectEvent(+t.dataset.ev);
    });
  }

  function heatLayer() {
    if (!opt.heat) return "";
    let s = "";
    for (const pid of ["1", "2"]) {
      const h = D.analytics[pid].heatmap;
      const max = Math.max(...h.share.flat()) || 1;
      h.share.forEach((row, j) => row.forEach((v, i) => {
        if (v > 0) s += `<rect x="${h.x0 + i * h.cell_m}" y="${h.y0 + j * h.cell_m}" width="${h.cell_m}" height="${h.cell_m}" fill="${C.navy[pid]}" fill-opacity="${(0.15 + 0.7 * v / max).toFixed(3)}"/>`;
      }));
    }
    return s;
  }

  function renderCourt(f) {
    L.heat.innerHTML = heatLayer();

    // trails: consecutive measured frames only; a gap stays a gap
    let tr = "";
    if (opt.trails) {
      for (const pid of ["1", "2"]) {
        const focus = sel.type === "player" && sel.pid === pid;
        const span = focus ? f : TRAIL;
        for (let k = Math.max(1, f - span + 1); k <= f; k++) {
          const a = pxy(F[k - 1], pid), b = pxy(F[k], pid);
          if (!a || !b) continue;
          const op = focus ? 0.9 : 0.85 * (1 - (f - k) / TRAIL);
          const dim = sel.type === "player" && !focus ? 0.35 : 1;
          tr += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${C.navy[pid]}" stroke-width="${focus ? 0.14 : 0.11}" stroke-linecap="round" stroke-opacity="${(op * dim).toFixed(3)}"/>`;
        }
      }
    }
    L.trails.innerHTML = tr;

    // event marks: where bounces / contacts happened, up to now.
    // Measured = solid; estimated = dashed and translucent, with its uncertainty radius.
    let mk = "";
    EV.forEach((e, i) => {
      if (e.frame > f || !e.xy_m) return;
      const [x, y] = e.xy_m, op = f - e.frame < 30 ? 1 : 0.5, est = e.status !== "measured";
      const ring = est && e.uncertainty_m > 0.35
        ? `<circle cx="${x}" cy="${y}" r="${e.uncertainty_m}" fill="${C.ball}" fill-opacity="0.06" stroke="${C.ball}" stroke-opacity="0.35" stroke-width="0.04" stroke-dasharray="0.15 0.15"/>` : "";
      if (e.type === "bounce") {
        const r = 0.3;
        mk += `<g data-ev="${i}" opacity="${op}" style="cursor:pointer">${ring}<circle cx="${x}" cy="${y}" r="0.6" fill="transparent"/><path d="M${x - r} ${y - r}L${x + r} ${y + r}M${x - r} ${y + r}L${x + r} ${y - r}" stroke="${C.ball}" stroke-width="0.09" ${est ? 'stroke-dasharray="0.12 0.08" stroke-opacity="0.8"' : ""}/></g>`;
      } else {
        const r = 0.34, col = C.navy[String(e.hitter)] || "#fff";
        mk += `<g data-ev="${i}" opacity="${op * 0.9}" style="cursor:pointer"><path d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z" fill="none" stroke="${col}" stroke-width="0.07" stroke-dasharray="0.12 0.08"/></g>`;
      }
    });
    L.marks.innerHTML = mk;

    // live ball on court: the estimated flight (analytics/ball_flight.py), never a
    // projection of the airborne ball. Trail = previous frames of the flight, consecutive
    // only, so a gated or unmodelled frame breaks it and a bounce keeps its corner.
    const rec = F[f], fl = rec.ball.flight;
    let b = "";
    if (fl) {
      for (let k = Math.max(1, f - BALL_TRAIL + 1); k <= f; k++) {
        const p = F[k - 1].ball.flight, q = F[k].ball.flight;
        if (!p || !q) continue;
        b += `<line x1="${p.xy_m[0]}" y1="${p.xy_m[1]}" x2="${q.xy_m[0]}" y2="${q.xy_m[1]}" stroke="${C.ball}" stroke-width="0.07" stroke-dasharray="0.16 0.12" stroke-opacity="${(0.75 * (1 - (f - k) / BALL_TRAIL)).toFixed(3)}"/>`;
      }
      const [x, y] = fl.xy_m;
      const measured = rec.ball.flightEvent && rec.ball.flightEvent.status === "measured";
      if (measured) {
        b += `<circle cx="${x}" cy="${y}" r="0.7" fill="${C.ball}" opacity="0.25"/><circle cx="${x}" cy="${y}" r="0.3" fill="${C.ball}" data-ball="1" style="cursor:pointer"/>`;
      } else {
        if (fl.uncertainty_m > 0.35) b += `<circle cx="${x}" cy="${y}" r="${fl.uncertainty_m}" fill="${C.ball}" fill-opacity="0.07" stroke="${C.ball}" stroke-opacity="0.4" stroke-width="0.04" stroke-dasharray="0.15 0.15"/>`;
        b += `<circle cx="${x}" cy="${y}" r="0.27" fill="${C.ball}" fill-opacity="0.45" stroke="${C.ball}" stroke-width="0.07" stroke-dasharray="0.12 0.08" data-ball="1" style="cursor:pointer"/>`;
      }
    }
    L.ball.innerHTML = b;

    // players
    let pl = "";
    for (const pid of ["1", "2"]) {
      const xy = pxy(rec, pid);
      if (!xy) continue;
      const focus = sel.type === "player" && sel.pid === pid;
      pl += `<g data-pid="${pid}" style="cursor:pointer">
        ${focus ? `<circle cx="${xy[0]}" cy="${xy[1]}" r="0.85" fill="none" stroke="#fff" stroke-width="0.06"/>` : ""}
        <circle cx="${xy[0]}" cy="${xy[1]}" r="0.42" fill="${C.navy[pid]}" stroke="#0c1f44" stroke-width="0.08"/>
        <text x="${xy[0]}" y="${xy[1] + 0.17}" text-anchor="middle" font-size="0.46" font-weight="600" fill="#fff" font-family="Inter, sans-serif">${pid}</text></g>`;
    }
    L.players.innerHTML = pl;

    // selected event highlight (shown even before it happens, so it can be found)
    let s = "";
    if (sel.type === "event") {
      const e = EV[sel.idx];
      if (e.xy_m) {
        const [x, y] = e.xy_m;
        const label = e.type === "bounce" ? "Bounce" : `Contact P${e.hitter}`;
        const tx = x > W_M / 2 ? x - 1.1 : x + 1.1, anchor = x > W_M / 2 ? "end" : "start";
        s = `<circle cx="${x}" cy="${y}" r="0.95" fill="none" stroke="#fff" stroke-width="0.06" stroke-dasharray="0.16 0.12"/>
             <text x="${tx}" y="${y + 0.2}" text-anchor="${anchor}" font-size="0.55" fill="#fff" font-family="Inter, sans-serif">${label} · ${(e.frame / fps).toFixed(2)} s</text>`;
      }
    }
    L.sel.innerHTML = s;
  }

  // ── video overlay ──────────────────────────────────────────────────────────
  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
  }
  function renderOverlay(f) {
    const ctx = octx, dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!opt.overlay) return;
    const rec = F[f], s = overlay.width / srcW;
    ctx.font = `600 ${11 * dpr}px Inter, sans-serif`;
    for (const pid of ["1", "2"]) {
      const p = rec.players[pid];
      if (!p || p.status !== "detected") continue;
      const [x1, y1, x2, y2] = p.bbox.map((v) => v * s);
      const focus = sel.type === "player" && sel.pid === pid;
      ctx.strokeStyle = C.navy[pid];
      ctx.lineWidth = (focus ? 3 : 1.5) * dpr;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const lab = `P${pid}`, w = ctx.measureText(lab).width + 8 * dpr;
      ctx.fillStyle = C.navy[pid];
      ctx.fillRect(x1, y1 - 15 * dpr, w, 15 * dpr);
      ctx.fillStyle = "#fff";
      ctx.fillText(lab, x1 + 4 * dpr, y1 - 4 * dpr);
    }
    const b = rec.ball.clean;
    // Trail of the ball's actual image positions: consecutive frames of the SAME cleaned
    // tracklet only, so it never bridges a gap or passes through a rejected detection.
    ctx.lineCap = "round";
    for (let k = Math.max(1, f - BALL_TRAIL + 1); k <= f; k++) {
      const p = F[k - 1].ball.clean, q = F[k].ball.clean;
      if (!p.xy || !q.xy || p.tracklet !== q.tracklet) continue;
      ctx.strokeStyle = `rgba(223,242,74,${(0.85 * (1 - (f - k) / BALL_TRAIL)).toFixed(3)})`;
      ctx.lineWidth = 2.5 * dpr;
      ctx.setLineDash(q.status === "interpolated" ? [3 * dpr, 3 * dpr] : []);
      ctx.beginPath();
      ctx.moveTo(p.xy[0] * s, p.xy[1] * s);
      ctx.lineTo(q.xy[0] * s, q.xy[1] * s);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (b.xy) {
      ctx.strokeStyle = C.ball;
      ctx.lineWidth = 2 * dpr;
      ctx.setLineDash(b.status === "detected" ? [] : [3 * dpr, 3 * dpr]);
      ctx.beginPath();
      ctx.arc(b.xy[0] * s, b.xy[1] * s, 10 * dpr, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (b.status === "rejected" && rec.ball.image.xy) {
      const [x, y] = [rec.ball.image.xy[0] * s, rec.ball.image.xy[1] * s], r = 6 * dpr;
      ctx.strokeStyle = C.rej;
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x - r, y + r); ctx.lineTo(x + r, y - r);
      ctx.stroke();
      ctx.fillStyle = C.rej;
      ctx.fillText("rejected detection", x + 10 * dpr, y + 4 * dpr);
    }
  }
  function overlayHit(ev) {
    const r = overlay.getBoundingClientRect(), s = r.width / srcW;
    const x = (ev.clientX - r.left) / s, y = (ev.clientY - r.top) / s;
    const rec = F[cur];
    for (const pid of ["1", "2"]) {
      const p = rec.players[pid];
      if (p && p.status === "detected") {
        const [x1, y1, x2, y2] = p.bbox;
        if (x >= x1 - 8 && x <= x2 + 8 && y >= y1 - 20 && y <= y2 + 8) return pid;
      }
    }
    return null;
  }

  // ── timeline: detection quality per frame ──────────────────────────────────
  const ROWS = { ball: [4, 12], court: [20, 12], p1: [36, 12], p2: [52, 12], ev: [68, 20] };
  const TL_H = 92;
  function renderTimeline(f) {
    const dpr = window.devicePixelRatio || 1, W = tl.width, H = tl.height, cw = W / N;
    const ctx = tctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.scale(1, dpr);
    const row = (key, colorAt) => {
      const [y, h] = ROWS[key];
      ctx.fillStyle = "#1a3466";
      ctx.fillRect(0, y, W, h);
      for (let k = 0; k < N; k++) {
        const c = colorAt(F[k]);
        if (!c) continue;
        ctx.fillStyle = c;
        if (c === C.rej) ctx.fillRect(k * cw + cw / 2 - 1.5 * dpr, y - 2, 3 * dpr, h + 4);
        else ctx.fillRect(k * cw, y, Math.ceil(cw), h);
      }
    };
    row("ball", (r) => ({ detected: C.ball, interpolated: "rgba(223,242,74,.38)", rejected: C.rej }[r.ball.clean.status]));
    row("court", (r) => {
      const fl = r.ball.flight;
      if (!fl) return null;
      if (r.ball.flightEvent && r.ball.flightEvent.status === "measured") return C.ball;
      return fl.uncertainty_m > 0.75 ? "rgba(223,242,74,.22)" : "rgba(223,242,74,.5)";
    });
    row("p1", (r) => (measured(r, "1") ? C.navy[1] : null));
    row("p2", (r) => (measured(r, "2") ? C.navy[2] : null));
    const [ey, eh] = ROWS.ev;
    EV.forEach((e, i) => {
      const x = (e.frame + 0.5) * cw, cy = ey + eh / 2, on = sel.type === "event" && sel.idx === i;
      const col = e.type === "bounce" ? C.ball : (C.navy[String(e.hitter)] || "#fff");
      ctx.beginPath();
      if (e.type === "bounce") ctx.arc(x, cy, on ? 6 : 4.5, 0, Math.PI * 2);
      else { const r = on ? 7 : 5.5; ctx.moveTo(x, cy - r); ctx.lineTo(x + r, cy); ctx.lineTo(x, cy + r); ctx.lineTo(x - r, cy); ctx.closePath(); }
      if (e.status === "measured" || e.type === "contact") { ctx.fillStyle = col; ctx.fill(); }
      else { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke(); }   // estimated / unavailable bounce: hollow
      if (on) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
    const px = (f + 0.5) * cw;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(px - 1, 0, 2, TL_H);
  }
  function tlFrame(ev) {
    const r = tl.getBoundingClientRect();
    return clamp(Math.floor(((ev.clientX - r.left) / r.width) * N), 0, N - 1);
  }
  function tlEventAt(ev) {
    const r = tl.getBoundingClientRect(), y = ev.clientY - r.top;
    if (y < ROWS.ev[0] - 2) return -1;
    const x = ev.clientX - r.left;
    let best = -1, bd = 9;
    EV.forEach((e, i) => { const d = Math.abs(((e.frame + 0.5) / N) * r.width - x); if (d < bd) { bd = d; best = i; } });
    return best;
  }
  function bindTimeline() {
    let drag = false;
    tl.addEventListener("pointerdown", (ev) => {
      const i = tlEventAt(ev);
      if (i >= 0) { selectEvent(i); return; }
      drag = true; tl.setPointerCapture(ev.pointerId); video.pause(); seekFrame(tlFrame(ev));
    });
    tl.addEventListener("pointermove", (ev) => {
      const f = tlFrame(ev), tip = $("tlTip"), r = tl.getBoundingClientRect(), rec = F[f];
      const st = [`f${f}`, `${(f / fps).toFixed(2)} s`, `ball ${rec.ball.clean.status}`,
                  rec.ball.flight ? `court ±${rec.ball.flight.uncertainty_m.toFixed(1)} m` : "court –",
                  `P1 ${measured(rec, "1") ? "✓" : "–"}`, `P2 ${measured(rec, "2") ? "✓" : "–"}`];
      if (rec.ball.flightEvent) st.push(`${rec.ball.flightEvent.type} (${rec.ball.flightEvent.status})`);
      tip.textContent = st.join("  ·  ");
      tip.style.left = `${50 + ((f + 0.5) / N) * r.width}px`;
      tip.hidden = false;
      if (drag) seekFrame(f);
    });
    tl.addEventListener("pointerup", () => (drag = false));
    tl.addEventListener("pointerleave", () => ($("tlTip").hidden = true));
  }

  // ── context panel ──────────────────────────────────────────────────────────
  function crumbs(parts) {
    return `<div class="crumbs">${parts.map((p, i) => (i < parts.length - 1 ? `<button data-go="overview">${p}</button> /` : `<span>${p}</span>`)).join(" ")}</div>`;
  }
  function stateRow(sw, name, value, miss, go) {
    const n = go ? `<button data-go="${go}">${name}</button>` : `<span>${name}</span>`;
    return `<div class="state"><i class="sw ${sw}"></i>${n}<span class="v${miss ? " miss" : ""}">${value}</span></div>`;
  }
  function ballText(rec) {
    const fe = rec.ball.flightEvent, fl = rec.ball.flight;
    if (fe && fe.type === "bounce" && fe.status === "measured") return "bounce · measured";
    if (fl) return `court est. ±${fl.uncertainty_m.toFixed(1)} m · ${fl.height_m.toFixed(1)} m up`;
    return { detected: "video only · no court est.", interpolated: "video (interp.) · no court est.",
             rejected: "detection rejected", unavailable: "not available" }[rec.ball.clean.status];
  }
  const segOf = (fl) => (fl && D.ball_flight ? D.ball_flight.segments[fl.segment] : null);
  function panelBall(f) {
    const rec = F[f], fl = rec.ball.flight, v = rec.ball.clean;
    const video = { detected: "detected in this frame", interpolated: "interpolated across a ≤ 2-frame gap",
                    rejected: "a detection here was rejected as a false positive", unavailable: "not detected in this frame" }[v.status];
    let body;
    if (fl) {
      const s = segOf(fl);
      const measuredBounce = rec.ball.flightEvent && rec.ball.flightEvent.status === "measured";
      body = `<dl><dt>Court x</dt><dd>${f2(fl.xy_m[0])} m</dd><dt>Court y</dt><dd>${f2(fl.xy_m[1])} m</dd>
          <dt>Height</dt><dd>${f2(fl.height_m)} m</dd><dt>Uncertainty</dt><dd>±${f2(fl.uncertainty_m)} m</dd>
          ${fl.residual_px != null ? `<dt>Off the detected ball</dt><dd>${fl.residual_px.toFixed(1)} px</dd>` : ""}</dl>
        ${measuredBounce ? `<p><b>Measured.</b> The video shows the impact here, so the ball is on the ground and its pixel maps to the court directly.</p>`
          : `<div class="callout note"><b>Estimated.</b> Flight ${s.from} → ${s.to}, frames ${s.start}–${s.end}: a gravity arc with a straight ground track, fitted to ${s.detections} ball detections (median ${s.median_residual_px} px off).
             ${s.status === "shown_uncertain" ? "Its landing point is only loosely fixed by this camera view, hence the uncertainty ring." : ""}</div>`}`;
    } else {
      body = `<div class="callout">No court position at frame ${f}. ${v.status === "detected" ? "The ball is visible in the video, but " : ""}this part of the rally is not modelled: a flight is only estimated between two detected events, and only where the estimate agrees with the video.</div>`;
    }
    return crumbs(["Clip", "Ball"]) + `<h3>Ball</h3><p class="sub">Frame ${f} · ${tStr(f)} · video: ${video}</p>${body}
      <p class="hint">An airborne ball is never projected through the court homography: that would place it metres from where it is.</p>`;
  }
  function panelOverview(f) {
    const rec = F[f], a = D.analytics, bc = D.ball_clean, cal = D.court.calibration;
    const bar = (name, n, d) => `<div class="avail"><span>${name}</span><span>${n}/${d}</span><div class="bar"><i style="width:${(100 * n / d).toFixed(1)}%"></i></div></div>`;
    const p = (pid) => {
      const c = measured(rec, pid);
      return c ? `x ${f2(c.xy_m[0])} · y ${f2(c.xy_m[1])} m` : "no position";
    };
    return crumbs(["Clip"]) +
      `<h3>First ${D.demo.duration_s.toFixed(1)} seconds</h3>
       <p class="sub">${N} frames at ${fps.toFixed(2)} fps · ${D.events.length} ball events detected</p>
       <p class="label">Tracking availability</p>
       ${bar("Player 1 · near", a["1"].frames_measured, N)}
       ${bar("Player 2 · far", a["2"].frames_measured, N)}
       ${bar("Ball · retained detections", bc.retained, N)}
       ${bar("Court gate · sampled frames on painted lines", cal.passed, cal.sampled)}
       <div class="rule"></div>
       <p class="label">Frame ${f} · ${tStr(f)}</p>
       ${stateRow("p1", "Player 1", p("1"), !measured(rec, "1"), "p1")}
       ${stateRow("p2", "Player 2", p("2"), !measured(rec, "2"), "p2")}
       ${stateRow("ball", "Ball", ballText(rec), !rec.ball.flight && rec.ball.clean.status !== "detected" && rec.ball.clean.status !== "interpolated", "ball")}
       ${stateRow("court", "Court", rec.court.homography ? `calibrated · ${rec.court.line_support}` : "no homography", !rec.court.homography)}
       <p class="hint" style="margin-top:14px">Select a player on the court or video, or an event on the timeline.</p>`;
  }
  function panelPlayer(f, pid) {
    const rec = F[f], c = measured(rec, pid), a = D.analytics[pid], s = side(pid);
    const cum = a.series.cumulative_m[f];
    let body = "";
    if (c) {
      body = `<dl>
        <dt>Court x</dt><dd>${f2(c.xy_m[0])} m</dd>
        <dt>Court y</dt><dd>${f2(c.xy_m[1])} m</dd>
        <dt>Depth</dt><dd>${depthNote(c.xy_m[1], s)}</dd>
        <dt>Path so far</dt><dd>${f2(cum)} m</dd></dl>`;
    } else {
      const lm = lastMeasured(pid, f);
      body = `<div class="callout">No reliable position at frame ${f}. The tracker returned no box for this player here, so nothing is drawn.${lm != null ? ` Last measured at frame ${lm} (${tStr(lm)}).` : " Not yet measured in this clip."}</div>`;
    }
    return crumbs(["Clip", `Player ${pid}`]) +
      `<h3>Player ${pid}</h3><p class="sub">${s === "near" ? "Near side, camera end" : "Far side"} · frame ${f}</p>${body}
       <div class="rule"></div>
       <p class="label">Across the clip</p>
       <dl>
        <dt>Frames measured</dt><dd>${a.frames_measured} / ${N}</dd>
        <dt>Path (smoothed)</dt><dd>${f2(a.path_m_smoothed)} m</dd>
        <dt>Tracking gaps</dt><dd>${a.gaps_not_bridged}</dd>
        <dt>Behind own baseline</dt><dd>${a.share_behind_baseline == null ? "—" : Math.round(100 * a.share_behind_baseline) + "% of frames"}</dd>
       </dl>
       <div class="callout note">${s === "far"
         ? "Far baseline: one image pixel spans roughly 0.1 m of depth, so positions here carry about ±1 m uncertainty."
         : "Feet are the bottom-centre of the detector box mapped through this frame's homography."}</div>`;
  }
  function panelEvent(f, idx) {
    const e = EV[idx], n = EV.length;
    const prev = EV[idx - 1], next = EV[idx + 1];
    let title, sub, body;
    const segs = (D.ball_flight ? D.ball_flight.segments : []).filter((s) => s.start === e.frame || s.end === e.frame);
    const check = segs.length ? `<p class="label">Check against the video</p><dl>${segs.map((s) =>
      `<dt>Flight ${s.start}–${s.end}</dt><dd>${s.status === "withheld" ? "withheld" : `${s.detections} det. · ${s.median_residual_px} px`}</dd>`).join("")}</dl>` : "";
    const moved = e.pipeline_frame != null && e.pipeline_frame !== e.frame ? ` (pipeline frame ${e.pipeline_frame})` : "";
    const [x, y] = e.xy_m || [null, null];
    if (e.type === "bounce") {
      title = "Bounce";
      const m = x == null ? null : singlesMargin(x, y), u = e.uncertainty_m || 0;
      const line = m == null ? "" : Math.abs(m) <= Math.max(u, 0.3)
        ? "too close to call" : `${Math.abs(m).toFixed(2)} m ${m >= 0 ? "inside" : "outside"}`;
      body = x == null
        ? `<div class="callout">No court position. The impact is not visible in the video, and no flight into or out of this bounce could be fitted reliably.</div>`
        : `<dl><dt>Court x</dt><dd>${f2(x)} m</dd><dt>Court y</dt><dd>${f2(y)} m</dd>
            ${e.status === "estimated" ? `<dt>Uncertainty</dt><dd>±${f2(u)} m</dd>` : ""}
            <dt>Half</dt><dd>${y > NET ? "near" : "far"}</dd><dt>Singles lines</dt><dd>${line}</dd>
            ${prev && prev.type === "contact" ? `<dt>Since contact (P${prev.hitter})</dt><dd>${((e.frame - prev.frame) / fps).toFixed(2)} s</dd>` : ""}</dl>
          ${e.status === "measured"
            ? `<p><b>Measured.</b> The impact is visible: the ball's downward image speed drops by ${e.vy_drop_px} px/frame here, so the ball is on the ground and its pixel maps straight to the court.</p>`
            : `<div class="callout note"><b>Estimated.</b> The ball was not detected through the impact. The landing point is where the fitted flights on either side meet the ground; equally good fits place it within ±${f2(u)} m.</div>`}`;
    } else {
      title = `Contact · P${e.hitter ?? "?"}`;
      body = x == null
        ? `<div class="callout">No position: the flights around this contact could not be fitted reliably.</div>`
        : `<dl><dt>Ball ground point x</dt><dd>${f2(x)} m</dd><dt>Ball ground point y</dt><dd>${f2(y)} m</dd>
            <dt>Strike height</dt><dd>${f2(e.height_m)} m</dd>
            ${e.distance_to_hitter_m != null ? `<dt>From hitter's feet</dt><dd>${f2(e.distance_to_hitter_m)} m</dd>` : ""}
            ${next && next.type === "bounce" ? `<dt>Flight to next bounce</dt><dd>${((next.frame - e.frame) / fps).toFixed(2)} s</dd>` : ""}</dl>
          <div class="callout note"><b>Estimated.</b> The ball is in the air at contact. Its height is fitted so that the flights either side match the detected ball, under the constraint that it is within reach (≤ 1.5 m) of the hitter; the ground point is where the camera ray through the ball reaches that height.</div>`;
    }
    body += check;
    sub = `Event ${idx + 1} of ${n} · frame ${e.frame}${moved} · ${tStr(e.frame)}${e.frame !== f ? ` · now at frame ${f}` : ""}`;
    return crumbs(["Clip", "Events", title]) + `<h3>${title}</h3><p class="sub">${sub}</p>${body}
      <div class="nav-ev"><button data-ev-step="-1" ${prev ? "" : "disabled"}>← Previous event</button><button data-ev-step="1" ${next ? "" : "disabled"}>Next event →</button></div>`;
  }
  function renderPanel(f) {
    const p = $("panel");
    p.innerHTML = sel.type === "player" ? panelPlayer(f, sel.pid) : sel.type === "event" ? panelEvent(f, sel.idx)
      : sel.type === "ball" ? panelBall(f) : panelOverview(f);
  }
  function bindPanel() {
    $("panel").addEventListener("click", (e) => {
      const go = e.target.closest("[data-go]"), st = e.target.closest("[data-ev-step]");
      if (go) {
        const g = go.dataset.go;
        if (g === "overview") select({ type: "overview" });
        else if (g === "p1" || g === "p2") select({ type: "player", pid: g.slice(1) });
        else if (g === "ball") select({ type: "ball" });
      } else if (st && !st.disabled) selectEvent(sel.idx + +st.dataset.evStep);
    });
  }

  // ── movement charts (SVG, one y-axis each, playhead synced to video) ───────
  function buildChart(host, { title, unit, series, yFmt, zeroLabel, caption }) {
    const W = 560, H = 250, m = { l: 44, r: 64, t: 10, b: 30 };
    const vals = series.flatMap((s) => s.values.filter((v) => v != null));
    let lo = Math.min(0, ...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08 || 1; lo = lo < 0 ? lo - pad : 0; hi += pad;
    const X = (f) => m.l + (f / (N - 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const ticks = []; const step = niceStep((hi - lo) / 4);
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(+v.toFixed(6));
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
    for (const v of ticks) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="${v === 0 ? "#0c1f44" : "#e6e9ef"}" stroke-width="${v === 0 ? 1 : 1}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" font-size="11" fill="#7d8aa3" font-family="IBM Plex Mono, monospace">${yFmt(v)}</text>`;
    if (zeroLabel && lo < 0) svg += `<text x="${W - m.r - 4}" y="${Y(0) - 6}" text-anchor="end" font-size="11" fill="#4a5a78">${zeroLabel}</text>`;
    for (let s = 0; s <= Math.floor((N - 1) / fps); s++) { const f = s * fps; svg += `<text x="${X(f)}" y="${H - 8}" text-anchor="middle" font-size="11" fill="#7d8aa3" font-family="IBM Plex Mono, monospace">${s} s</text>`; }
    for (const s of series) {
      let d = "", pen = false, lastF = null;
      s.values.forEach((v, f) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(f).toFixed(1)} ${Y(v).toFixed(1)}`; pen = true; lastF = f; });
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
      if (lastF != null) svg += `<text x="${X(lastF) + 8}" y="${Y(s.values[lastF]) + 4}" font-size="12" fill="#0c1f44">${s.name}</text>`;
    }
    svg += `<line class="ph" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}" stroke="#0c1f44" stroke-width="1" stroke-dasharray="3 3"/>`;
    svg += `<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="#7d8aa3" stroke-width="1" visibility="hidden"/>`;
    svg += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent" style="cursor:crosshair"/></svg>`;
    const rows = [];
    for (let f = 0; f < N; f += 15) rows.push(`<tr><td>${(f / fps).toFixed(2)}</td>${series.map((s) => `<td>${s.values[f] == null ? "—" : yFmt(s.values[f], 2)}</td>`).join("")}</tr>`);
    host.innerHTML = `<h4>${title}</h4><p class="unit">${unit}</p>
      <div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>
      ${svg}<div class="tip" hidden></div><figcaption>${caption}</figcaption>
      <details><summary>Data table</summary><table><tr><th>t (s)</th>${series.map((s) => `<th>${s.name}</th>`).join("")}</tr>${rows.join("")}</table></details>`;
    const root = host.querySelector("svg"), ph = root.querySelector(".ph"), xh = root.querySelector(".xh"), tip = host.querySelector(".tip");
    const toF = (ev) => { const r = root.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W; return clamp(Math.round(((x - m.l) / (W - m.l - m.r)) * (N - 1)), 0, N - 1); };
    const hit = root.querySelector(".hit");
    hit.addEventListener("pointermove", (ev) => {
      const f = toF(ev), r = root.getBoundingClientRect(), k = r.width / W;
      xh.setAttribute("x1", X(f)); xh.setAttribute("x2", X(f)); xh.setAttribute("visibility", "visible");
      tip.innerHTML = `${(f / fps).toFixed(2)} s · ` + series.map((s) => `${s.name} ${s.values[f] == null ? "—" : yFmt(s.values[f], 2) + " m"}`).join(" · ");
      tip.style.left = `${X(f) * k}px`; tip.style.top = `${root.offsetTop + m.t * k}px`; tip.hidden = false;
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; xh.setAttribute("visibility", "hidden"); });
    hit.addEventListener("click", (ev) => { video.pause(); seekFrame(toF(ev)); document.getElementById("analysis").scrollIntoView({ behavior: "smooth" }); });
    charts.push((f) => { ph.setAttribute("x1", X(f)); ph.setAttribute("x2", X(f)); });
  }
  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
  }
  function buildMovement() {
    const a = D.analytics;
    $("moveLede").textContent = `Computed only from frames where a player's feet were mapped through that frame's homography. ` +
      `Distance is never accumulated across a tracking gap. ${D.demo.duration_s.toFixed(1)} seconds is a short sample: ` +
      `these numbers describe this clip, not a player.`;
    buildChart($("chartDistance"), {
      title: "Distance covered", unit: "Cumulative path length, metres, against clip time",
      series: [{ name: "P1", color: C.paper[1], values: a["1"].series.cumulative_m }, { name: "P2", color: C.paper[2], values: a["2"].series.cumulative_m }],
      yFmt: (v, d = 0) => v.toFixed(d),
      caption: "P2's line starts late: the far player's first 17 frames were lost to a track-ID switch, and those frames are not counted.",
    });
    buildChart($("chartDepth"), {
      title: "Court depth", unit: "Metres inside own baseline (negative = behind it), against clip time",
      series: [{ name: "P1", color: C.paper[1], values: a["1"].series.depth_m }, { name: "P2", color: C.paper[2], values: a["2"].series.depth_m }],
      yFmt: (v, d = 0) => v.toFixed(d), zeroLabel: "baseline",
      caption: "Breaks in a line are frames with no measured position. Far-side depth carries about ±1 m uncertainty.",
    });
    const row = (pid) => { const m = a[pid]; return `<tr><td><i class="sw p${pid}"></i>&nbsp; Player ${pid}</td><td>${m.side}</td>
      <td class="num">${m.frames_measured} / ${N}</td><td class="num">${m.tracked_seconds.toFixed(2)} s</td>
      <td class="num">${m.path_m_smoothed.toFixed(1)} m</td><td class="num">${m.path_m_raw.toFixed(1)} m</td>
      <td class="num">${m.gaps_not_bridged}</td><td class="num">${Math.round(100 * m.share_behind_baseline)}%</td></tr>`; };
    $("moveTable").innerHTML = `<table><thead><tr><th>Player</th><th>Side</th><th class="num">Frames measured</th><th class="num">Time tracked</th>
      <th class="num">Path, smoothed</th><th class="num">Path, raw</th><th class="num">Gaps</th><th class="num">Behind baseline</th></tr></thead>
      <tbody>${row("1")}${row("2")}</tbody></table>
      <p class="note">Smoothed = the pipeline's Kalman filter. It changes path length by under 0.5% here, so the raw/smoothed gap is not a jitter estimate.
      Detector-box jitter at broadcast resolution does add some path that was never run; no ground truth exists to size it.</p>`;
  }

  function buildMethod() {
    const b = D.ball_clean, m = D.analysis.models;
    $("methodGrid").innerHTML = `
      <div><h3>Players</h3><p>${esc(m.players)}. The two players are chosen once for the whole clip by court position and persistence, and must stand on opposite sides of the net.</p>
        <ul><li>Position = bottom-centre of the box (the feet), mapped through that frame's homography.</li>
        <li>A frame without a tracked box has no position; nothing is held or invented.</li></ul></div>
      <div><h3>Ball</h3><p>${esc(m.ball)}. Raw detections are linked into tracklets with a speed bound; single unconfirmed detections are rejected.</p>
        <table><tr><td>Raw detections</td><td>${b.raw_detections}</td></tr><tr><td>Retained</td><td>${b.retained}</td></tr>
        <tr><td>Rejected (frames ${b.rejected_frames.join(", ")})</td><td>${b.rejected}</td></tr>
        <tr><td>Interpolated (≤ 2-frame gaps)</td><td>${b.interpolated}</td></tr><tr><td>No position</td><td>${b.unavailable}</td></tr></table>
        <p style="margin-top:12px"><b>On the court</b> the ball is never projected through the homography while airborne. A camera is recovered from the court keypoints (${D.ball_flight.camera.focal_px.toFixed(0)} px focal length, keypoints reproject within ${D.ball_flight.camera.keypoint_reprojection_median_px} px), and each flight between two events is fitted as a gravity arc over a straight ground track to the detected ball. A flight is drawn only where it agrees with the video; its uncertainty is the spread of equally good fits.</p>
        <table>${D.ball_flight.segments.map((s) => `<tr><td>Flight ${s.start}–${s.end} (${s.from} → ${s.to})</td><td>${s.status === "withheld" ? "withheld" : `${s.median_residual_px} px · ±${Math.max(...s.uncertainty_m).toFixed(1)} m`}</td></tr>`).join("")}</table>
        <p style="margin-top:12px"><b>Checked by eye:</b> frame 74 is a false positive at the edge of the picture; the serve toss (frames 11–33) and the ball at frames 76–83 (camera panning with it) are real. The serve contact (~frame 31) is not detected, so the serve is not modelled.</p></div>
      <div><h3>Court</h3><p>${esc(m.court)}. ${esc(m.court_mapping)}.</p>
        <ul><li>Validity gate: predicted lines must lie on painted pixels — ${D.court.calibration.passed} of ${D.court.calibration.sampled} sampled frames pass (median support ${D.court.calibration.median_line_support}).</li>
        <li>Court frame: metres, origin at the far-left doubles corner, y towards the camera.</li></ul></div>`;
  }

  function buildChrome() {
    const cal = D.court.calibration;
    $("wsMeta").textContent = `US Open broadcast · frames 0–${N - 1} · ${fps.toFixed(2)} fps · court gate ${cal.valid ? "passed" : "failed"} (support ${cal.median_line_support})`;
    const still = D.demo.stills && D.demo.stills[0];
    if (still) $("heroStill").src = still;
    $("heroCaption").innerHTML = `<span>Frame 106 · 3.54 s — Player 2 strikes at the far baseline.</span><span>Still from the analysed clip</span>`;
    $("foot").innerHTML = `<p><b>Built on</b> <a href="https://github.com/HarshTomar1234/Tennis-Vision">HarshTomar1234/Tennis-Vision</a> (MIT): detection, tracking, court model, homography and event pipeline.
      This fork adds the streaming exporter, the ball-track clean-up, the movement analytics and this viewer.</p>
      <p><b>Models:</b> Ultralytics YOLOv8x (AGPL-3.0), TrackNet weights by yastrebksv (research use), court keypoint fine-tune by the upstream author.
      Broadcast footage belongs to its rights holder and is not redistributed with the code.</p>`;
  }

  // ── master render ──────────────────────────────────────────────────────────
  function render(f, force = false) {
    if (!F || !(f >= 0 && f < N)) return;      // e.g. a resize before the first frame
    if (f === cur && !force) return;
    cur = f;
    renderCourt(f);
    renderOverlay(f);
    renderTimeline(f);
    renderPanel(f);
    for (const c of charts) c(f);
    $("timeReadout").textContent = tStr(f);
    $("frameReadout").textContent = `frame ${f} / ${N - 1}`;
  }
  const syncToVideo = () => render(frameFromTime(video.currentTime));
  let noVideo = false;   // source clip not generated locally (it is not distributed)
  function seekFrame(f) {
    if (noVideo) { render(clamp(f, 0, N - 1)); return; }
    video.currentTime = (clamp(f, 0, N - 1) + 0.5) / fps;
    syncToVideo();
  }
  function select(s) { sel = s; render(cur, true); }
  function selectEvent(i) {
    if (i < 0 || i >= EV.length) return;
    sel = { type: "event", idx: i };
    video.pause();
    if (cur === EV[i].frame) render(cur, true); else seekFrame(EV[i].frame);
  }

  function startClock() {
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const onFrame = (_n, meta) => {
        const f = frameFromTime(meta.mediaTime);
        video.dataset.presentedFrame = f;      // the frame actually on screen (used by the browser test)
        render(f);
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
    } else {
      const tick = () => { if (!video.paused) syncToVideo(); requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }
    for (const ev of ["seeked", "seeking", "pause", "loadeddata"]) video.addEventListener(ev, syncToVideo);
  }

  function bindControls() {
    const icon = () => { $("iconPlay").hidden = !video.paused; $("iconPause").hidden = video.paused; $("btnPlay").setAttribute("aria-label", video.paused ? "Play" : "Pause"); };
    video.addEventListener("play", icon);
    video.addEventListener("pause", icon);
    const toggle = () => (video.paused ? video.play() : video.pause());
    $("btnPlay").addEventListener("click", toggle);
    $("btnPrev").addEventListener("click", () => { video.pause(); seekFrame(cur - 1); });
    $("btnNext").addEventListener("click", () => { video.pause(); seekFrame(cur + 1); });
    $("rate").addEventListener("change", (e) => (video.playbackRate = +e.target.value));
    const bindOpt = (id, key) => $(id).addEventListener("change", (e) => { opt[key] = e.target.checked; render(cur, true); });
    bindOpt("overlayToggle", "overlay"); bindOpt("trailToggle", "trails"); bindOpt("heatToggle", "heat"); bindOpt("smoothToggle", "smooth");
    overlay.addEventListener("click", (ev) => { const pid = overlayHit(ev); if (pid) select({ type: "player", pid }); else toggle(); });
    document.addEventListener("keydown", (e) => {
      if (["SELECT", "INPUT"].includes(e.target.tagName) && e.target.type !== "checkbox") return;
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      else if (e.code === "ArrowLeft") { e.preventDefault(); video.pause(); seekFrame(cur - 1); }
      else if (e.code === "ArrowRight") { e.preventDefault(); video.pause(); seekFrame(cur + 1); }
      else if (e.code === "Escape") select({ type: "overview" });
    });
    const ro = new ResizeObserver(() => { sizeCanvas(overlay); sizeCanvas(tl); render(cur, true); });
    ro.observe(overlay); ro.observe(tl);
  }

  async function boot() {
    D = await (await fetch("data/demo_frames.json", { cache: "no-store" })).json();
    F = D.frames; fps = D.demo.fps; N = F.length; srcW = D.source.width; srcH = D.source.height;
    // Events as refined by the flight model (measured / estimated / unavailable positions);
    // the pipeline's own event frame is kept on each as pipeline_frame.
    EV = (D.ball_flight ? D.ball_flight.events : D.events).slice().sort((a, b) => a.frame - b.frame);
    for (const e of EV) if (F[e.frame]) F[e.frame].ball.flightEvent = e;
    drawCourt(); buildChrome(); buildMovement(); buildMethod();
    bindControls(); bindTimeline(); bindPanel();
    const startAt = new URLSearchParams(location.search).get("frame");
    try {
      const resp = await fetch(D.demo.video);
      if (!resp.ok) throw new Error(String(resp.status));
      video.src = URL.createObjectURL(await resp.blob());
    } catch (_) {
      // Broadcast footage is not redistributed with the code. Without it the court,
      // timeline and panel still work; frames are stepped instead of played.
      noVideo = true;
      $("loading").innerHTML = `<div style="max-width:46ch;text-align:center;line-height:1.6">Source video not found.
        Generate it locally with <code>scripts/prepare_web_demo.py</code> (see README).<br>
        The court, timeline and panel still work: step with ← / → or click the timeline.</div>`;
      $("btnPlay").disabled = true;
      sizeCanvas(overlay); sizeCanvas(tl);
      render(startAt !== null ? clamp(parseInt(startAt, 10) || 0, 0, N - 1) : 0, true);
      return;
    }
    video.loop = true;
    video.addEventListener("loadeddata", () => {
      $("loading").hidden = true; sizeCanvas(overlay); sizeCanvas(tl);
      if (startAt !== null) seekFrame(parseInt(startAt, 10) || 0); else syncToVideo();
    }, { once: true });
    startClock();
    render(0, true);
  }
  boot().catch((err) => { $("loading").textContent = `Could not load demo data: ${err.message}. Start it with: python web/serve.py`; console.error(err); });
})();
