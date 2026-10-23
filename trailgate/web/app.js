"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const STATIC = window.TRAILGATE_STATIC === true;
const thumbOf = (n) => (STATIC ? "thumbs/" : "/thumb/") + n;
const cards = new Map();
const S = {
  status: null, mode: "replay", reel: null, frames: [], rows: new Map(), es: null, running: false,
  pinned: null, current: null, queue: [], phoneBusy: false, total: {},
};
const STAMP = { skip: "Skipped", small: "Nova Lite", large: "Nova Pro", escalate: "Alert" };
const BADGE = { small: "Lite", large: "Pro", escalate: "Alert" };
const usd = (v) => "$" + (v < 1 ? v.toFixed(4) : v.toFixed(2));
const int = (v) => Math.round(v).toLocaleString("en-US");
const pct = (v) => (v == null ? "" : v.toFixed(2));
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const title = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const replaying = () => STATIC || S.mode === "replay";

async function boot() {
  S.status = await (await fetch(STATIC ? "data/status.json" : "/api/status")).json();
  S.mode = STATIC ? "replay" : params.get("mode") || S.status.mode;
  if (S.mode === "replay" && !S.status.replay.frames) S.mode = "live";
  renderReels();
  renderModes();
  await loadReel(params.get("reel") || S.status.reels[0].id);
  document.addEventListener("keydown", onKey);
  $("run").addEventListener("click", () => (S.running ? stop() : start()));
  $("digest-x").addEventListener("click", () => ($("digest").hidden = true));
  $("ph-acts").addEventListener("click", onAct);
  $("pin").addEventListener("click", () => S.pinned != null && pin(S.pinned));
}

function renderReels() {
  const box = $("reels");
  box.innerHTML = "";
  for (const r of S.status.reels) {
    const b = el("button", "", esc(r.name));
    b.type = "button";
    b.dataset.id = r.id;
    b.title = `${r.size} frames`;
    b.addEventListener("click", () => !S.running && loadReel(r.id));
    box.append(b);
  }
}

function renderModes() {
  const st = S.status;
  const modes = $("modes");
  modes.hidden = STATIC;
  for (const b of modes.querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.mode === S.mode));
    if (b.dataset.mode === "replay") b.disabled = !st.replay.frames;
    b.onclick = () => { if (!S.running && S.mode !== b.dataset.mode) { S.mode = b.dataset.mode; renderModes(); loadReel(S.reel.id); } };
  }
  const chip = $("chip");
  const live = S.mode === "live";
  const down = live && !st.decider.reachable;
  chip.className = "chip " + (down ? "down" : live ? "live" : "replay");
  $("chip-text").textContent = down ? "Decider offline" : live ? "Live" : "Replay";
  chip.title = live ? (st.decider.url || "") + (st.record ? ` · recording to ${st.record}` : "") : "Recorded Decider answers";
  const hint = st.setup || (down ? "Start the Decider or switch to Replay." : "");
  $("hint").textContent = hint;
  $("hint").hidden = !hint;
  $("fine").textContent = replaying()
    ? "Replay of recorded Decider answers. Bedrock costs are estimates."
    : "Live Decider calls. Costs marked est. are estimates.";
}

async function loadReel(id) {
  const data = STATIC ? await staticCard(id) : await (await fetch(`/api/reel?id=${encodeURIComponent(id)}&mode=${S.mode}`)).json();
  S.reel = S.status.reels.find((r) => r.id === id) || S.status.reels[0];
  S.frames = data.frames;
  for (const b of $("reels").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === S.reel.id));
  reset();
  const grid = $("grid");
  grid.innerHTML = "";
  grid.style.setProperty("--cols", S.frames.length > 63 ? 10 : 9);
  for (const f of S.frames) {
    const t = el("div", "tile");
    t.dataset.state = "pending";
    t.dataset.i = f.index;
    t.tabIndex = 0;
    t.setAttribute("aria-label", `Camera ${f.location}, ${f.clock}`);
    t.innerHTML = `<div class="face"><img src="${f.thumb}" alt="" loading="eager"><span class="clock">${esc(f.clock)}</span><span class="tag"><span class="sp"></span><span class="md"></span></span></div>`;
    t.addEventListener("mouseenter", () => inspect(f.index));
    t.addEventListener("mouseleave", () => inspect(S.pinned ?? S.current));
    t.addEventListener("click", () => pin(f.index));
    t.addEventListener("keydown", (e) => { if (e.key === "Enter") pin(f.index); });
    grid.append(t);
  }
  $("c-total").textContent = S.frames.length;
  inspect(S.frames.length ? 0 : null);
}

async function staticCard(id) {
  if (!cards.has(id)) cards.set(id, await (await fetch(`data/${encodeURIComponent(id)}.json`)).json());
  return cards.get(id);
}

async function playStatic() {
  const run = {};
  S.play = run;
  const card = await staticCard(S.reel.id);
  const pace = Math.max(0.05, Math.min(parseFloat(params.get("pace")) || 1, 5));
  const wait = (s) => new Promise((ok) => setTimeout(ok, s * 1000));
  for (const row of card.rows) {
    if (S.play !== run) return;
    onDeciding({ index: row.index });
    await wait(Math.max(0.05, row.delay_s * pace));
    if (S.play !== run) return;
    onFrame(row);
  }
  if (S.play !== run) return;
  stop();
  showDigest(card.digest);
}

function reset() {
  S.rows.clear();
  S.queue = [];
  S.current = null;
  S.pinned = null;
  S.total = { seen: 0, skip: 0, small: 0, large: 0, escalate: 0, spent: 0, est: 0, base: 0, lite: 0, frames: 0 };
  clearInterval(S.digestTimer);
  $("digest").hidden = true;
  $("pin").hidden = true;
  hidePhone(true);
  updateLedger();
  for (const t of document.querySelectorAll(".tile")) { t.dataset.state = "pending"; t.classList.remove("selected"); t.removeAttribute("title"); }
}

function start() {
  if (!S.frames.length || S.status.setup) return;
  reset();
  if (STATIC) { setRunning(true); playStatic(); return; }
  const q = new URLSearchParams({ id: S.reel.id, mode: S.mode });
  if (params.get("pace")) q.set("pace", params.get("pace"));
  const es = new EventSource("/api/run?" + q);
  S.es = es;
  setRunning(true);
  es.addEventListener("deciding", (e) => onDeciding(JSON.parse(e.data)));
  es.addEventListener("frame", (e) => onFrame(JSON.parse(e.data)));
  es.addEventListener("failed", (e) => onFailed(JSON.parse(e.data)));
  es.addEventListener("done", (e) => { stop(); showDigest(JSON.parse(e.data)); });
  es.onerror = () => { if (S.es === es) stop(); };
}

function stop() {
  S.play = null;
  if (S.es) S.es.close();
  S.es = null;
  setRunning(false);
  for (const t of document.querySelectorAll('.tile[data-state="deciding"]')) t.dataset.state = "pending";
  inspect(S.pinned ?? S.current);
}

function setRunning(on) {
  S.running = on;
  $("run").classList.toggle("going", on);
  $("run-label").textContent = on ? "Stop" : S.rows.size ? "Run again" : "Run";
  for (const b of document.querySelectorAll("#reels button, #modes button")) b.classList.toggle("locked", on);
}

function tile(i) { return document.querySelector(`.tile[data-i="${i}"]`); }

function onDeciding({ index }) {
  S.current = index;
  tile(index).dataset.state = "deciding";
  if (S.pinned == null) inspect(index);
}

function onFrame(row) {
  const { index, result, costing } = row;
  S.rows.set(index, row);
  const t = tile(index);
  t.classList.remove("flip");
  void t.offsetWidth;
  t.classList.add("flip");
  setTimeout(() => paintTile(t, row), 240);

  const T = S.total;
  T.seen += 1;
  T[result.tier] = (T[result.tier] || 0) + 1;
  T.spent += costing.actual_usd;
  T.est += costing.estimated_usd;
  T.base += costing.baseline_usd;
  T.lite += costing.baseline_lite_usd;
  T.frames += costing.frames;
  updateLedger(result.tier);

  if (S.pinned == null) inspect(index);
  if (result.tier === "escalate") queueAlert(row);
}

function onFailed({ index, error }) {
  const t = tile(index);
  t.dataset.state = "failed";
  t.querySelector(".md").textContent = "Error";
  t.title = error;
  S.total.seen += 1;
  updateLedger();
}

function paintTile(t, row) {
  const r = row.result;
  t.dataset.state = r.tier;
  t.querySelector(".sp").textContent = r.tier !== "escalate" && r.species ? title(r.species) : "";
  t.querySelector(".md").textContent = BADGE[r.tier] || "";
}

function bump(id) {
  const n = $(id).parentElement;
  n.classList.remove("bump");
  void n.offsetWidth;
  n.classList.add("bump");
}

function updateLedger(tier) {
  const T = S.total;
  $("c-seen").textContent = T.seen;
  $("c-empty").textContent = T.skip || 0;
  $("c-lite").textContent = T.small || 0;
  $("c-pro").textContent = T.large || 0;
  $("c-alert").textContent = T.escalate || 0;
  if (tier) bump({ skip: "c-empty", small: "c-lite", large: "c-pro", escalate: "c-alert" }[tier]);
  const spent = T.spent + T.est;
  const est = !replaying() && T.est > 0 ? " est." : "";
  $("c-spent").textContent = usd(spent) + est;
  $("c-pro-all").textContent = usd(T.base);
  $("c-lite-all").textContent = usd(T.lite);
  const full = (T.base / (T.frames || 1)) * Math.max(S.frames.length, T.frames) || 1;
  $("bar-spent").style.width = Math.min(100, (spent / full) * 100) + "%";
  $("bar-pro").style.width = Math.min(100, (T.base / full) * 100) + "%";
  $("bar-lite").style.width = Math.min(100, (T.lite / full) * 100) + "%";
}

function pin(i) {
  S.pinned = S.pinned === i ? null : i;
  for (const t of document.querySelectorAll(".tile.selected")) t.classList.remove("selected");
  if (S.pinned != null) tile(i).classList.add("selected");
  $("pin").hidden = S.pinned == null;
  inspect(S.pinned ?? S.current);
}

function inspect(i) {
  const f = i == null ? null : S.frames[i];
  const row = i == null ? null : S.rows.get(i);
  const lens = $("lens");
  const img = $("lens-img");
  lens.className = "lens";
  if (!f) {
    img.removeAttribute("src");
    for (const id of ["lens-cam", "lens-clock", "lens-truth"]) $(id).textContent = "";
    renderQs(null);
    renderBurst(null);
    setVerdict(null);
    return;
  }
  if (img.getAttribute("src") !== f.full) img.src = f.full;
  $("lens-cam").textContent = `Cam ${f.location}`;
  $("lens-clock").textContent = f.clock;
  $("lens-truth").textContent = `Truth: ${f.audit ? f.audit + " (audited)" : f.label}`;
  const deciding = !row && tile(i)?.dataset.state === "deciding";
  if (deciding) lens.classList.add("busy");
  if (row) lens.classList.add({ escalate: "alert", large: "pro", skip: "skip" }[row.result.tier] || "lite");
  renderQs(row ? row.result : null, deciding);
  renderBurst(row ? row.result : null, f);
  setVerdict(row, f, deciding);
}

function renderBurst(r, f) {
  const box = $("burst");
  if (!f || f.burst.length < 2) { box.hidden = true; box.innerHTML = ""; return; }
  const items = (r && r.burst) || f.burst.map((n) => ({ file_name: n, self: n === f.file_name, thumb: thumbOf(n) }));
  const th = S.status.thresholds;
  box.hidden = false;
  box.innerHTML = `<span class="k">Burst</span>` + items.map((b) => {
    const hot = b.escalate != null && b.escalate >= th.escalate_above;
    return `<span class="bf${b.self ? " self" : ""}${hot ? " hot" : ""}" title="Animal ${pct(b.present)}"><img src="${b.thumb}" alt=""><b>${b.present != null ? pct(b.present) : ""}</b></span>`;
  }).join("");
}

function answerOf(r, key) {
  const a = (r && r.answers) || {};
  return a[key] || {};
}

function meter(v, ticks) {
  return `<span class="meter"><span style="width:${(v || 0) * 100}%"></span>${ticks.map((t) => `<i style="left:${t * 100}%"></i>`).join("")}</span>`;
}

function renderQs(r, deciding) {
  const box = $("qs");
  if (!r && !deciding) { box.hidden = true; return; }
  const q = S.status.questions;
  const th = S.status.thresholds;
  const present = r ? r.present : null;
  const escl = r ? r.escalate : null;
  const diff = r ? r.difficulty : null;
  const dconf = r ? (r.difficulty_confidence ?? answerOf(r, "difficulty").confidence) : null;
  const wait = `<span class="qv wait">…</span>`;
  const hot = escl != null && escl >= th.escalate_above;
  box.hidden = false;
  box.innerHTML = `
    <div class="q" title="${esc(q.present.instructions)} Skip below ${th.skip_below}, Nova Lite from ${th.small_above}."><span class="ql">Animal</span>${meter(present, [th.skip_below, th.small_above])}${present == null ? wait : `<span class="qv">${pct(present)}</span>`}</div>
    <div class="q esc${hot ? " hot" : ""}" title="${esc(q.escalate.instructions)} Alert from ${th.escalate_above}."><span class="ql">Person or vehicle</span>${meter(escl, [th.escalate_above])}${escl == null ? wait : `<span class="qv">${pct(escl)}</span>`}</div>
    <div class="q diff" title="${esc(q.difficulty.instructions)} ${esc(title(S.status.difficulty_note))}."><span class="ql">Difficulty</span><span class="dv">${diff ? esc(title(diff)) + (dconf != null ? ` <small>${pct(dconf)}</small>` : "") : ""}</span><span class="qv">not used</span></div>`;
}

function reasonOf(r) {
  const th = S.status.thresholds;
  const raw = r.reason || "";
  if (raw.startsWith("burst:")) return "Another frame in the burst looks like an animal.";
  if (r.tier === "skip" && r.present != null) return `Animal ${pct(r.present)}, under ${th.skip_below}. No model called.`;
  if (r.tier === "small" && r.present != null) return `Animal ${pct(r.present)}. Decider is sure.`;
  if (r.tier === "large" && r.present != null) return `Animal ${pct(r.present)}. Decider is unsure.`;
  if (r.tier === "escalate" && r.escalate != null) return `Person or vehicle ${pct(r.escalate)}. Sent to the ranger.`;
  return title(raw);
}

function setVerdict(row, f, deciding) {
  const stamp = $("stamp");
  const calls = $("callrow");
  calls.innerHTML = "";
  if (!row) {
    stamp.hidden = !deciding;
    stamp.className = "stamp";
    stamp.textContent = "Deciding";
    $("reason").textContent = deciding ? "" : !f ? "" : S.running ? "Waiting" : S.rows.size ? "Not decided" : "Press Run";
    return;
  }
  const r = row.result;
  stamp.hidden = false;
  if (stamp.dataset.k !== `${row.index}`) { stamp.classList.remove("land"); void stamp.offsetWidth; }
  stamp.dataset.k = row.index;
  stamp.className = "stamp land " + r.tier;
  stamp.textContent = STAMP[r.tier] || r.tier;
  $("reason").textContent = reasonOf(r);

  calls.append(el("div", "call", `<span>Decider 2B</span><span class="num">${int(r.decider_latency_ms || 0)} ms</span>`));
  const c = row.costing;
  const groups = new Map();
  for (const l of c.lines) {
    const key = `${l.model}|${l.kind}|${l.estimated}`;
    const g = groups.get(key) || { ...l, n: 0, usd: 0, latency: 0 };
    g.n += 1;
    g.usd += l.usd;
    g.latency = Math.max(g.latency, l.latency_ms || 0);
    groups.set(key, g);
  }
  for (const g of groups.values()) {
    const what = g.kind === "alert" ? "alert" : r.species && g.n === 1 ? esc(title(r.species)) : "label";
    const est = g.estimated && !replaying() ? " est." : "";
    const ms = g.estimated ? "" : ` · ${int(g.latency)} ms`;
    const n = el("div", "call" + (g.model === "Nova Pro" ? " pro" : " lite"), `<span>${esc(g.model)} ${what}${g.n > 1 ? ` ×${g.n}` : ""}</span><span class="num">${usd(g.usd)}${est}${ms}</span>`);
    if (g.error) n.title = g.error;
    calls.append(n);
  }
}

function queueAlert(row) {
  S.queue.push(row);
  $("ph-queue").textContent = S.queue.length > 1 ? `${S.queue.length - 1} more waiting` : "";
  if (!S.phoneBusy) nextAlert();
}

function nextAlert() {
  const row = S.queue[0];
  if (!row) { hidePhone(); return; }
  S.phoneBusy = true;
  const f = row.meta;
  const r = row.result;
  $("ph-time").textContent = f.clock || "--:--";
  $("ph-title").textContent = `Person or vehicle at camera ${f.location}`;
  $("ph-img").src = f.full;
  $("ph-text").textContent = r.alert || `Decider is ${Math.round((r.escalate || 0) * 100)}% sure. ${f.lighting === "night_ir" ? "Night IR" : "Daylight"}, decided in ${int(r.decider_latency_ms || 0)} ms.`;
  $("ph-done").hidden = true;
  $("ph-acts").hidden = false;
  for (const b of $("ph-acts").querySelectorAll("button")) { b.disabled = false; b.classList.remove("tapped"); }
  $("ph-queue").textContent = S.queue.length > 1 ? `${S.queue.length - 1} more waiting` : "";
  const p = $("phone");
  p.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => p.classList.add("in")));
}

function hidePhone(now) {
  const p = $("phone");
  p.classList.remove("in");
  S.phoneBusy = false;
  if (now) p.hidden = true;
  else setTimeout(() => { if (!S.phoneBusy) p.hidden = true; }, 650);
}

async function onAct(e) {
  const b = e.target.closest("button[data-act]");
  const row = S.queue[0];
  if (!b || !row || b.disabled) return;
  for (const x of $("ph-acts").querySelectorAll("button")) x.disabled = true;
  b.classList.add("tapped");
  let ok = true;
  if (STATIC) recordLocal(b.dataset.act, row);
  else try {
    const res = await fetch("/api/action", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: b.dataset.act, frame: row.meta.file_name, alert: row.result.alert, escalate: row.result.escalate, mode: S.mode }),
    });
    ok = res.ok;
  } catch { ok = false; }
  const done = $("ph-done");
  done.textContent = (b.dataset.act === "dispatch_patrol" ? "Patrol dispatched to camera " + row.meta.location : "Marked as known staff") + (ok ? "" : ". Not logged.");
  done.hidden = false;
  $("ph-acts").hidden = true;
  setTimeout(() => {
    S.queue.shift();
    hidePhone();
    setTimeout(() => { if (S.queue.length) nextAlert(); }, 700);
  }, 2200);
}

function recordLocal(act, row) {
  try {
    const key = "trailgate.actions";
    const log = JSON.parse(localStorage.getItem(key) || "[]");
    log.push({ at: new Date().toISOString(), action: act, frame: row.meta.file_name, camera: row.meta.location, escalate: row.result.escalate });
    localStorage.setItem(key, JSON.stringify(log.slice(-200)));
  } catch { return; }
}

function showDigest(d) {
  $("dg-line").innerHTML = `${d.frames} frames. <span class="n-empty">${d.empty} skipped</span>, <span class="n-lab">${d.labelled} labelled</span>, <span class="n-alert">${d.alerts} for you</span>.`;
  const sp = Object.entries(d.species).sort((a, b) => b[1] - a[1]);
  $("dg-species").innerHTML = sp.length
    ? sp.map(([k, v]) => `<li>${esc(title(k))}<b>${v}</b></li>`).join("")
    : `<li class="lite">Nova Lite<b>${d.lite}</b></li><li class="pro">Nova Pro<b>${d.pro}</b></li>`;
  $("dg-alerts").innerHTML = d.alert_frames.map((f) => `<figure><img src="${f.thumb}" alt=""><figcaption>Cam ${esc(f.location)}, ${esc(f.clock)}</figcaption></figure>`).join("") || `<p class="dg-note">Nothing tonight.</p>`;
  const saved = d.baseline_usd ? Math.round((1 - d.spent_usd / d.baseline_usd) * 100) : 0;
  $("dg-cost").innerHTML = `<b>${usd(d.spent_usd)}</b><span class="vs">vs ${usd(d.baseline_usd)} with Nova Pro on every frame</span>`;
  $("dg-note").textContent = `${saved}% less. Decider p50 ${d.decider_p50_ms ? int(d.decider_p50_ms) + " ms" : "n/a"}.`;
  S.digestTimer = setTimeout(waitDigest, 900);
}

function waitDigest() {
  const ready = () => !S.phoneBusy && !S.queue.length && !S.running;
  clearInterval(S.digestTimer);
  if (ready()) { $("digest").hidden = false; return; }
  S.digestTimer = setInterval(() => { if (ready()) { clearInterval(S.digestTimer); $("digest").hidden = false; } }, 500);
}

function onKey(e) {
  if (e.target.closest("input, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); S.running ? stop() : start(); }
  if (e.key === "Escape") { $("digest").hidden = true; if (S.pinned != null) pin(S.pinned); }
}

boot();
