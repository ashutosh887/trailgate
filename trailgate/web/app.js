"use strict";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const S = {
  status: null, mode: "replay", reel: null, frames: [], rows: new Map(), es: null, running: false,
  pinned: null, current: null, queue: [], phoneBusy: false, bedrock: null, total: {},
};
const ACTION = { skip: "Empty", small: "Label", large: "Label", escalate: "Alert" };
const MODEL = { small: "Nova Lite", large: "Nova Pro", escalate: "Nova Lite" };
const usd = (v) => "$" + (v < 1 ? v.toFixed(4) : v.toFixed(2));
const int = (v) => Math.round(v).toLocaleString("en-US");
const pct = (v) => (v == null ? "—" : v.toFixed(2));
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const title = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

async function boot() {
  S.status = await (await fetch("/api/status")).json();
  S.mode = params.get("mode") || S.status.mode;
  if (S.mode === "replay" && !S.status.replay.frames) S.mode = "live";
  renderReels();
  renderModes();
  await loadReel(params.get("reel") || S.status.reels[0].id);
  document.addEventListener("keydown", onKey);
  $("run").addEventListener("click", () => (S.running ? stop() : start()));
  $("digest-x").addEventListener("click", () => ($("digest").hidden = true));
  $("ph-acts").addEventListener("click", onAct);
}

function renderReels() {
  const box = $("reels");
  box.innerHTML = "";
  for (const r of S.status.reels) {
    const b = el("button", "", `${esc(r.name)}<small>${r.size}</small>`);
    b.type = "button";
    b.dataset.id = r.id;
    b.addEventListener("click", () => !S.running && loadReel(r.id));
    box.append(b);
  }
}

function renderModes() {
  for (const b of $("modes").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.mode === S.mode));
    if (b.dataset.mode === "replay") b.disabled = !S.status.replay.frames;
    b.onclick = () => { if (!S.running && S.mode !== b.dataset.mode) { S.mode = b.dataset.mode; renderModes(); loadReel(S.reel.id); } };
  }
  const st = S.status;
  const dec = $("pill-decider");
  if (S.mode === "live") {
    dec.className = "pill " + (st.decider.reachable ? "live" : "down");
    dec.querySelector("em").textContent = st.decider.reachable ? "live · " + st.decider.url.replace(/^https?:\/\//, "") : "offline";
    $("hint").textContent = st.decider.reachable
      ? `Live: one Decider call per frame, one frame at a time${st.pipeline.available ? "" : " · router-only adapter"}`
      : "Decider is not reachable; start it or switch to Replay";
    setPill("pill-source", "", st.record ? "recording → " + st.record : "live run, not recorded");
  } else {
    dec.className = "pill replay";
    dec.querySelector("em").textContent = "replay";
    $("hint").textContent = "Replay · real Decider answers, v2 burst policy · sped up for viewing, latencies are real";
    setPill("pill-source", "", (st.replay.path || "—").split("/").pop());
  }
  setBedrock(null);
}

function setPill(id, cls, text) {
  const p = $(id);
  p.className = "pill " + (cls || "quiet");
  p.querySelector("em").textContent = text;
}

function setBedrock(state) {
  S.bedrock = state;
  if (state === "live") setPill("pill-bedrock", "live", "live");
  else if (state === "queued") setPill("pill-bedrock", "warn", "unavailable · calls queued");
  else setPill("pill-bedrock", "warn", S.mode === "replay" ? "from cache" : "waiting");
}

async function loadReel(id) {
  const data = await (await fetch(`/api/reel?id=${encodeURIComponent(id)}&mode=${S.mode}`)).json();
  S.reel = S.status.reels.find((r) => r.id === id) || S.status.reels[0];
  S.frames = data.frames;
  for (const b of $("reels").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.id === S.reel.id));
  reset();
  const cams = new Set(S.frames.map((f) => f.location)).size;
  const night = S.frames.filter((f) => f.lighting === "night_ir").length;
  $("card-title").textContent = `${S.reel.name} · last night`;
  $("card-meta").textContent = `${S.frames.length} frames · ${cams} cameras · ${night} night IR · Caltech Camera Traps`;
  const grid = $("grid");
  grid.innerHTML = "";
  grid.style.setProperty("--cols", S.frames.length > 63 ? 10 : 9);
  for (const f of S.frames) {
    const t = el("div", "tile");
    t.dataset.state = "pending";
    t.dataset.i = f.index;
    t.tabIndex = 0;
    t.innerHTML = `<div class="face"><img src="${f.thumb}" alt="" loading="eager"><span class="idx">${String(f.index + 1).padStart(2, "0")} · ${esc(f.clock)}</span><span class="tag"><span class="sp"></span><span class="md"></span></span></div>`;
    t.addEventListener("mouseenter", () => inspect(f.index));
    t.addEventListener("mouseleave", () => inspect(S.pinned ?? S.current));
    t.addEventListener("click", () => pin(f.index));
    grid.append(t);
  }
  $("c-total").textContent = S.frames.length;
  inspect(S.frames.length ? 0 : null);
}

function reset() {
  S.rows.clear();
  S.queue = [];
  S.current = null;
  S.pinned = null;
  S.total = { seen: 0, skip: 0, small: 0, large: 0, escalate: 0, spent: 0, est: 0, base: 0, lite: 0, tokens: 0, tokEst: 0, btok: 0 };
  $("digest").hidden = true;
  hidePhone(true);
  updateLedger();
  $("pin").textContent = "following";
  $("pin").className = "pin";
  for (const t of document.querySelectorAll(".tile")) { t.dataset.state = "pending"; t.classList.remove("selected"); t.removeAttribute("title"); }
}

function start() {
  if (!S.frames.length) return;
  reset();
  for (const t of document.querySelectorAll(".tile")) t.dataset.state = "pending";
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
  if (S.es) S.es.close();
  S.es = null;
  setRunning(false);
  for (const t of document.querySelectorAll('.tile[data-state="deciding"]')) t.dataset.state = "pending";
  $("lens").classList.remove("busy");
}

function setRunning(on) {
  S.running = on;
  $("run").classList.toggle("going", on);
  $("run-label").textContent = on ? "Stop" : S.rows.size ? "Run again" : "Run the card";
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
  T.tokens += costing.tokens;
  T.tokEst += costing.tokens_estimated;
  T.btok += costing.baseline_tokens;
  T.perFrame = costing.baseline_usd;
  updateLedger(result.tier);

  const calls = result.calls || [];
  if (calls.some((c) => !c.error)) setBedrock("live");
  else if (MODEL[result.tier] && S.bedrock !== "live") setBedrock("queued");

  if (S.pinned == null) inspect(index);
  if (result.tier === "escalate") queueAlert(row);
}

function onFailed({ index, error }) {
  const t = tile(index);
  t.dataset.state = "failed";
  t.querySelector(".sp").textContent = "decider error";
  t.title = error;
  S.total.seen += 1;
  updateLedger();
}

function modelCall(result) {
  const calls = result.calls || [];
  const ok = calls.find((c) => !c.error);
  return { ok, failed: calls.find((c) => c.error), queued: !ok };
}

function paintTile(t, row) {
  const r = row.result;
  t.dataset.state = r.tier;
  const sp = t.querySelector(".sp");
  const md = t.querySelector(".md");
  md.className = "md";
  if (r.tier === "skip") { sp.textContent = "$0 · no model"; md.textContent = ""; return; }
  if (r.tier === "escalate") { sp.textContent = "Alert"; md.textContent = "ranger"; return; }
  const { ok } = modelCall(r);
  md.textContent = r.tier === "small" ? "Lite" : "Pro";
  if (ok && r.species) sp.textContent = title(r.species);
  else { sp.textContent = "queued"; md.classList.add("queued"); t.title = `${MODEL[r.tier]} — queued (Bedrock unavailable)`; }
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
  const est = T.est > 0 ? " est." : "";
  $("c-spent").textContent = usd(spent) + est;
  $("c-lite-all").textContent = usd(T.lite) + " est.";
  $("c-pro-all").textContent = usd(T.base) + " est.";
  $("c-tokens").textContent = `${int(T.tokens + T.tokEst)} tokens · Pro on every frame ${int(T.btok)}`;
  $("c-ratio").textContent = T.base > 0
    ? `${Math.round((1 - spent / T.base) * 100)}% under Pro · ${(spent / (T.lite || 1)).toFixed(1)}× Lite`
    : "Bedrock costs estimated";
  const full = (T.perFrame || 0) * S.frames.length || 1;
  $("bar-spent").style.width = Math.min(100, (spent / full) * 100) + "%";
  $("bar-lite").style.width = Math.min(100, (T.lite / full) * 100) + "%";
  $("bar-pro").style.width = Math.min(100, (T.base / full) * 100) + "%";
}

function pin(i) {
  S.pinned = S.pinned === i ? null : i;
  for (const t of document.querySelectorAll(".tile.selected")) t.classList.remove("selected");
  if (S.pinned != null) tile(i).classList.add("selected");
  $("pin").textContent = S.pinned != null ? "pinned · click again to follow" : "following";
  $("pin").className = "pin" + (S.pinned != null ? " on" : "");
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
    $("lens-cam").textContent = "cam —";
    $("lens-clock").textContent = "--:--";
    $("lens-light").textContent = "";
    renderQs(null);
    setVerdict(null);
    return;
  }
  if (img.getAttribute("src") !== f.full) img.src = f.full;
  $("lens-cam").textContent = `cam ${f.location}`;
  $("lens-clock").textContent = f.clock;
  $("lens-light").textContent = `truth: ${f.label}${f.audit ? " (audit: " + f.audit + ")" : ""} · ${f.lighting === "night_ir" ? "night IR" : "day"}`;
  const deciding = !row && tile(i)?.dataset.state === "deciding";
  if (deciding) lens.classList.add("busy");
  if (row) lens.classList.add({ escalate: "alert", large: "pro", skip: "skip" }[row.result.tier] || "x");
  renderQs(row ? row.result : null, deciding);
  renderBurst(row ? row.result : null, f);
  setVerdict(row, f, deciding);
}

function renderBurst(r, f) {
  const box = $("burst");
  const frames = (r && r.burst) || null;
  if (!f || f.burst.length < 2) { box.innerHTML = `<span class="k">burst</span><span class="b-none">single frame</span>`; return; }
  const items = frames || f.burst.map((n) => ({ file_name: n, self: n === f.file_name, thumb: `/thumb/${n}` }));
  const th = S.status.thresholds;
  box.innerHTML = `<span class="k">burst · ${items.length}</span>` + items.map((b) => {
    const hot = b.escalate != null && b.escalate >= th.escalate_above;
    const p = b.present != null ? pct(b.present) : "…";
    return `<span class="bf${b.self ? " self" : ""}${hot ? " hot" : ""}"><img src="${b.thumb}" alt=""><b>${p}</b></span>`;
  }).join("");
}

function answerOf(r, key) {
  const a = (r && r.answers) || {};
  return a[key] || {};
}

function renderQs(r, deciding) {
  const q = S.status.questions;
  const th = S.status.thresholds;
  const box = $("qs");
  const present = r ? r.present : null;
  const escl = r ? r.escalate : null;
  const diff = r ? r.difficulty : null;
  const dconf = r ? (r.difficulty_confidence ?? answerOf(r, "difficulty").confidence) : null;
  const probs = answerOf(r, "difficulty").probabilities || {};
  const wait = deciding ? "…" : "—";
  box.innerHTML = `
    <div class="q"><span class="qt"><small>present · noul · P(animal)</small>${esc(q.present.instructions)}</span>
      <span class="qv">${present == null ? wait : pct(present)}</span>
      <span class="meter"><span style="width:${(present || 0) * 100}%"></span><i style="left:${th.skip_below * 100}%" data-t="skip <${th.skip_below}"></i><i style="left:${th.small_above * 100}%" data-t="Lite ≥${th.small_above}"></i></span></div>
    <div class="q esc"><span class="qt"><small>escalate · noul · P(person/vehicle)</small>${esc(q.escalate.instructions)}</span>
      <span class="qv ${escl != null && escl >= th.escalate_above ? "hot" : ""}">${escl == null ? wait : pct(escl)}</span>
      <span class="meter"><span style="width:${(escl || 0) * 100}%"></span><i style="left:${th.escalate_above * 100}%" data-t="alert ≥${th.escalate_above}"></i></span></div>
    <div class="q diff"><span class="qt"><small>difficulty · choice</small>${esc(title(S.status.difficulty_note))}</span>
      <span class="qv">${dconf == null ? "" : "conf " + pct(dconf)}</span>
      <div class="opts">${Object.keys(q.difficulty.criteria).map((k) => `<span class="opt ${k} ${diff === k ? "on" : ""}"><em style="width:${(probs[k] || 0) * 100}%"></em><span>${k}</span><b>${probs[k] != null ? pct(probs[k]) : ""}</b></span>`).join("")}</div></div>`;
}

function setVerdict(row, f, deciding) {
  const stamp = $("stamp");
  const calls = $("callrow");
  calls.innerHTML = "";
  if (!row) {
    stamp.className = "stamp";
    stamp.textContent = deciding ? "Deciding" : "Pending";
    $("route").textContent = deciding ? "One Decider call, three typed questions, one frame." : "Press Run. Each frame gets one Decider call with three typed questions.";
    $("reason").textContent = f ? `CCT ground truth: ${f.label}` : "";
    return;
  }
  const r = row.result;
  if (stamp.dataset.k !== `${row.index}`) { stamp.classList.remove("land"); void stamp.offsetWidth; stamp.classList.add("land"); }
  stamp.dataset.k = row.index;
  stamp.className = "stamp land " + r.tier;
  stamp.textContent = ACTION[r.tier] || r.tier;
  $("route").textContent = {
    skip: "Archived. No model was called.",
    small: "Nova Lite · Decider confident there is an animal.",
    large: "Nova Pro · Decider unsure, so the stronger model looks.",
    escalate: "Person or vehicle → straight to the ranger's phone.",
  }[r.tier];
  $("reason").textContent = r.reason;

  calls.append(el("div", "call", `<span class="who dec">Decider 2B</span><span>3 questions · <b>$0</b> local</span><span class="num">${int(r.decider_latency_ms || 0)} ms</span>`));
  const want = MODEL[r.tier];
  if (want) {
    const { ok, failed } = modelCall(r);
    const cls = (ok ? ok.model : want) === "Nova Pro" ? "pro" : "lite";
    if (ok) {
      const what = r.tier === "escalate" ? "alert written" : `<b>${esc(title(r.species || "?"))}</b>${r.species_confidence ? " · " + esc(r.species_confidence) : ""}`;
      calls.append(el("div", "call", `<span class="who ${cls}">${esc(ok.model)}</span><span>${what}</span><span class="num">${usd(ok.cost_usd)} · ${int(ok.latency_ms)} ms</span>`));
    } else {
      const n = el("div", "call queued", `<span class="who off">${esc(want)}</span><span>queued · Bedrock unavailable</span><span class="num"></span>`);
      n.title = `${want} — queued (Bedrock unavailable). Estimated ${usd(row.costing.estimated_usd)}.` + (failed ? ` ${failed.error}` : "");
      calls.append(n);
    }
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
  $("ph-title").textContent = `Person or vehicle at camera ${f.location}, ${f.clock}`;
  $("ph-img").src = f.full;
  const { ok } = modelCall(r);
  $("ph-text").textContent = ok && r.alert ? r.alert : `Decider is ${Math.round((r.escalate || 0) * 100)}% sure a person or vehicle is in frame. Alert sent with the frame attached; the written note from Nova Lite is queued (Bedrock unavailable).`;
  $("ph-meta").textContent = `${f.captured} · ${f.lighting === "night_ir" ? "night IR" : "daylight"} · decided in ${int(r.decider_latency_ms || 0)} ms`;
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
  let note = "";
  try {
    const res = await fetch("/api/action", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: b.dataset.act, frame: row.meta.file_name, alert: row.result.alert, escalate: row.result.escalate, mode: S.mode }),
    });
    note = res.ok ? " · logged to outbox/actions.jsonl" : " · not logged";
  } catch { note = " · not logged"; }
  const done = $("ph-done");
  done.textContent = (b.dataset.act === "dispatch_patrol" ? "Patrol dispatched to camera " + row.meta.location : "Marked as known staff") + note;
  done.hidden = false;
  $("ph-acts").hidden = true;
  setTimeout(() => {
    S.queue.shift();
    hidePhone();
    setTimeout(() => { if (S.queue.length) nextAlert(); }, 700);
  }, 2200);
}

function showDigest(d) {
  $("dg-line").innerHTML = `${d.frames} frames: <span class="n-empty">${d.empty} empty skipped</span>, <span class="n-lab">${d.labelled} labelled</span>, <span class="n-alert">${d.alerts} for you</span>.`;
  const sp = Object.entries(d.species).sort((a, b) => b[1] - a[1]);
  $("dg-species").innerHTML = sp.length
    ? sp.map(([k, v]) => `<li>${esc(title(k))}<b>${v}</b></li>`).join("")
    : `<li class="none">${d.lite} to Nova Lite, ${d.pro} to Nova Pro; species labels queued while Bedrock is unavailable.</li>`;
  $("dg-alerts").innerHTML = d.alert_frames.map((f) => `<figure><img src="${f.thumb}" alt=""><figcaption><b>cam ${esc(f.location)}</b> · ${esc(f.clock)}</figcaption></figure>`).join("") || `<p class="dg-note mono">Nothing for you tonight.</p>`;
  const saved = d.baseline_usd ? Math.round((1 - d.spent_usd / d.baseline_usd) * 100) : 0;
  $("dg-cost").innerHTML = `<b>${usd(d.spent_usd)}</b> <span class="vs">vs Pro ${usd(d.baseline_usd)} · Lite ${usd(d.baseline_lite_usd)}</span>`;
  $("dg-note").textContent = `${saved}% less than Nova Pro on every frame; ${(d.spent_usd / (d.baseline_lite_usd || 1)).toFixed(1)}× Nova Lite on every frame (estimates). ${d.empty} frames never reached a model. Decider p50 ${d.decider_p50_ms ? int(d.decider_p50_ms) + " ms" : "—"}, run locally.` + (S.total.est > 0 ? " Bedrock spend is estimated from token counts; calls are queued." : "");
  setTimeout(() => { if (!S.phoneBusy) $("digest").hidden = false; else waitDigest(); }, 900);
}

function waitDigest() {
  const iv = setInterval(() => { if (!S.phoneBusy && !S.queue.length) { clearInterval(iv); $("digest").hidden = false; } }, 500);
}

function onKey(e) {
  if (e.target.closest("input, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); S.running ? stop() : start(); }
  if (e.key === "Escape") { $("digest").hidden = true; if (S.pinned != null) pin(S.pinned); }
}

boot();
