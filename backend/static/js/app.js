/* ═══════════════════════════════════════════════════════════
   app.js - 设备智能服务助手演示页
   职责：登录（JWT）+ SSE 聊天（POST /api/v1/chat/stream）
   SSE 事件：routing_decision / progress / token / guidance /
             pipeline_plan / meta / done / error
   ═══════════════════════════════════════════════════════════ */
"use strict";

/* ── 全局状态 ─────────────────────────────── */
const state = {
  token: localStorage.getItem("eqcs_token") || "",
  user: null,
  sessionId: newSessionId(),
  sending: false,
  abortCtrl: null,
  attachImage: null,      // 待发送图片（dataURL）
  micUnsupported: false,  // 浏览器不支持语音识别
  pendingMessage: null,   // 登录墙拦截下来的待发消息
};

function newSessionId() {
  return (crypto.randomUUID && crypto.randomUUID()) ||
    "s-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
}

/* ── DOM 快捷引用 ─────────────────────────── */
const $ = (id) => document.getElementById(id);
const chatWindow = $("chatWindow");
const msgInput = $("msgInput");
const sendBtn = $("sendBtn");
const stopBtn = $("stopBtn");
const micBtn = $("micBtn");
const attachBtn = $("attachBtn");
const fileInput = $("fileInput");
const attachPreview = $("attachPreview");
const attachThumb = $("attachThumb");

const BOT_SVG =
  '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="8" ' +
  'width="16" height="12" rx="3"/><path d="M12 8V4M8 4h8"/><circle cx="9" cy="14" ' +
  'r="0.6"/><circle cx="15" cy="14" r="0.6"/></svg>';

/* ═══════════════════════════════════════════
   一、鉴权与视图切换（登录页 ⇄ 主界面）
   ═══════════════════════════════════════════ */

const loginView = $("loginView");
const appView = $("appView");
const loginError = $("loginError");
const loginSubmitBtn = $("loginSubmitBtn");

async function login(username, password) {
  const resp = await fetch("/api/v1/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `登录失败（HTTP ${resp.status}）`);
  }
  const data = await resp.json();
  state.token = data.access_token;
  localStorage.setItem("eqcs_token", state.token);
  await fetchMe();
}

/** 校验本地 token；返回是否有效（无效时清除本地态）。 */
async function fetchMe() {
  if (!state.token) return false;
  const resp = await fetch("/api/v1/me", {
    headers: { Authorization: "Bearer " + state.token },
  });
  if (resp.ok) {
    state.user = await resp.json();
    renderUserBox();
    return true;
  }
  state.token = "";
  state.user = null;
  localStorage.removeItem("eqcs_token");
  return false;
}

function renderUserBox() {
  if (!state.user) return;
  const name = state.user.user_id || "-";
  const role = state.user.role || "-";
  $("userName").textContent = role === "engineer" ? "张工程师" : name;
  $("userRole").textContent = role;
  $("userAvatar").textContent = role === "engineer" ? "工" : role.charAt(0).toUpperCase();
}

/* ── 视图切换 ─────────────────────────────── */
function showApp() {
  loginView.classList.add("hidden");
  appView.classList.remove("hidden");
  msgInput.focus();
}

function showLogin(message) {
  appView.classList.add("hidden");
  loginView.classList.remove("hidden");
  loginError.textContent = message || "";
  loginError.classList.toggle("hidden", !message);
  $("loginUsername").focus();
}

/** 会话中途鉴权失效：清态回登录页 */
function gotoLogin(message) {
  state.token = "";
  state.user = null;
  localStorage.removeItem("eqcs_token");
  showLogin(message);
}

/* 页面加载后的鉴权分流：有 token 且有效 → 直接进主界面 */
async function initAuth() {
  if (state.token && (await fetchMe())) {
    showApp();
  } else {
    showLogin();
  }
}

/* ═══════════════════════════════════════════
   二、消息渲染
   ═══════════════════════════════════════════ */

function el(tag, cls, html, isHtml) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) {
    if (isHtml) node.innerHTML = html;
    else node.textContent = html;
  }
  return node;
}

function scrollBottom() { chatWindow.scrollTop = chatWindow.scrollHeight; }

function addWelcomeGone() {
  const w = chatWindow.querySelector(".welcome");
  if (w) w.remove();
}

/** 简版 markdown：**粗体**、`代码`，转义后输出 HTML */
function renderMd(text) {
  const escaped = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, '<span class="md-code">$1</span>');
}

function nowHM() {
  const d = new Date();
  return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
}

function addUserMsg(text) {
  addWelcomeGone();
  const wrap = el("div", "msg-user");
  const col = el("div", "user-col");
  const bubble = el("div", "bubble", text);
  col.appendChild(bubble);
  col.appendChild(el("span", "msg-time", nowHM()));
  const avatar = el("span", "avatar avatar-user", "我");
  wrap.appendChild(col);
  wrap.appendChild(avatar);
  chatWindow.appendChild(wrap);
  scrollBottom();
}

/** 开启一个新的助手回合：左侧机器人头像 + 右侧内容列（流式时头像呼吸） */
function openTurn() {
  const turn = el("div", "turn streaming");
  const avatar = el("span", "avatar avatar-bot", BOT_SVG, true);
  const body = el("div", "turn-body");
  turn.appendChild(avatar);
  turn.appendChild(body);
  chatWindow.appendChild(turn);
  scrollBottom();
  return body;
}

const _AGENT_BADGE = {
  knowledge:  ["rb-knowledge", "产品知识问答"],
  diagnosis:  ["rb-diagnosis", "故障诊断"],
  ticket:     ["rb-ticket", "工单管理"],
  after_sale: ["rb-after_sale", "售后协调"],
};

function routeCard(body, data) {
  const agentKey = data.agent_type || "knowledge";
  const [badgeCls, defaultName] = _AGENT_BADGE[agentKey] || _AGENT_BADGE.knowledge;
  const card = el("div", "route-card");

  const badge = el("span", "route-badge " + badgeCls);
  badge.appendChild(el("i", "dot"));
  badge.appendChild(el("span", null, data.agent_display || defaultName));
  card.appendChild(badge);

  const conf = Math.round((data.confidence || 0) * 100);
  const confWrap = el("span", "route-conf");
  const confBar = el("span", "conf-bar");
  confBar.appendChild(el("i"));
  confWrap.appendChild(confBar);
  confWrap.appendChild(el("span", null, conf + "%"));
  card.appendChild(confWrap);

  requestAnimationFrame(() => { confBar.firstChild.style.width = conf + "%"; });

  const modeText = data.execution_mode === "pipeline" ? "多 Agent 协同"
    : data.execution_mode === "clarify" ? "意图澄清" : "单 Agent";
  card.appendChild(el("span", "route-mode", modeText));

  if (data.reason) card.appendChild(el("span", "route-reason", data.reason));
  body.appendChild(card);
  scrollBottom();
}

/** 进度行：打字三点 + 阶段文案，同回合内复用 */
function progressLine(body, stage) {
  let line = body.querySelector(".progress-line");
  if (!line) {
    line = el("div", "progress-line");
    const dots = el("span", "typing-dots");
    dots.appendChild(el("i")); dots.appendChild(el("i")); dots.appendChild(el("i"));
    const txt = el("span", "progress-text");
    line.appendChild(dots);
    line.appendChild(txt);
    body.appendChild(line);
  }
  line.querySelector(".progress-text").textContent = stage;
  scrollBottom();
}

function tokenChunk(body, content) {
  let bubble = body.querySelector(".msg-assistant");
  if (!bubble) {
    bubble = el("div", "msg-assistant streaming-cursor");
    body.appendChild(bubble);
  }
  bubble.dataset.raw = (bubble.dataset.raw || "") + content;
  bubble.innerHTML = renderMd(bubble.dataset.raw);
  scrollBottom();
}

function guidanceBox(body, data) {
  body.appendChild(el("div", "guidance-box", data.message || ""));
  scrollBottom();
}

function errorBox(body, message) {
  body.appendChild(el("div", "error-box", "⚠ " + (message || "服务异常")));
  scrollBottom();
}

function pipelineCard(body, plan) {
  const card = el("div", "pipeline-card");
  card.appendChild(el("h4", null, "🚑 " + (plan.title || "多 Agent 协同计划")));
  card.appendChild(el("div", "intro", plan.intro || ""));
  const steps = el("div", "pipeline-steps");
  (plan.steps || []).forEach((s) => {
    const row = el("div", "pstep");
    row.appendChild(el("span", "no", String(s.step)));
    const meta = el("div", "pstep-meta");
    meta.appendChild(el("span", "lb", s.label || ""));
    meta.appendChild(el("span", "ds", s.desc || ""));
    row.appendChild(meta);
    steps.appendChild(row);
  });
  card.appendChild(steps);
  body.appendChild(card);
  scrollBottom();
}

/** meta 脚注：回答模式 / 置信度 / 来源 / 工单号 / Agent 链 */
function metaLine(body, data) {
  const line = el("div", "meta-line");
  const chip = (label, value) => {
    if (value === undefined || value === null || value === "") return;
    const c = el("span", "meta-chip");
    c.appendChild(document.createTextNode(label));
    c.appendChild(el("b", null, String(value)));
    line.appendChild(c);
  };
  const conf = data.confidence;
  chip("模式", data.answer_mode);
  chip("置信度", typeof conf === "number" ? Math.round(conf * 100) + "%" : conf);
  chip("工单号", data.ticket_id);
  if (Array.isArray(data.agent_chain) && data.agent_chain.length) {
    chip("Agent 链", data.agent_chain.join(" → "));
  }
  if (Array.isArray(data.sources) && data.sources.length) {
    const names = data.sources.map((s) =>
      typeof s === "string" ? s : (s.source || s.doc || "")).filter(Boolean);
    if (names.length) chip("来源", names.slice(0, 3).join("、"));
  }
  if (line.children.length) { body.appendChild(line); scrollBottom(); }
}

/** 完成回合：移除进度行与流式光标，附加时间戳，停止头像呼吸 */
function finishTurn(body) {
  const line = body && body.querySelector(".progress-line");
  if (line) line.remove();
  if (body && body.querySelector(".msg-assistant")) {
    body.appendChild(el("span", "msg-time assistant-time", nowHM()));
  }
  const turn = body && body.parentElement;
  if (turn) turn.classList.remove("streaming");
  if (body && !body.childNodes.length) body.parentElement.remove();
  const bubble = body && body.querySelector(".msg-assistant");
  if (bubble) bubble.classList.remove("streaming-cursor");
}

/* ═══════════════════════════════════════════
   三、SSE 聊天（POST + ReadableStream 解析）
   ═══════════════════════════════════════════ */

async function sendMessage(text) {
  const trimmed = (text || "").trim();
  if (state.sending || (!trimmed && !state.attachImage)) return;

  /* 登录墙：未登录先到登录页，登录成功后自动续发 */
  if (!state.token) {
    state.pendingMessage = trimmed;
    gotoLogin("请先登录后再发送消息");
    return;
  }

  state.sending = true;
  setSendingUI(true);

  const outText = trimmed || "请结合图片分析设备故障";
  addUserMsg(trimmed || "📷 [故障图片]");
  const attach = state.attachImage;
  clearAttachment();

  const body = openTurn();
  progressLine(body, "正在连接服务…");
  let gotEvent = false;

  state.abortCtrl = new AbortController();
  const payload = {
    session_id: state.sessionId,
    message: outText,
    customer_id: "",
    device_model: $("deviceModel").value.trim(),
    device_sn: $("deviceSn").value.trim(),
    image: attach || "",
  };

  try {
    const resp = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + state.token,
      },
      body: JSON.stringify(payload),
      signal: state.abortCtrl.signal,
    });
    if (resp.status === 401) {
      /* Token 失效：清除并回登录页 */
      gotoLogin("登录已过期，请重新登录");
      return;
    }
    if (!resp.ok || !resp.body) {
      throw new Error(`服务返回 HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    /* SSE 帧：data: <json>\r\n\r\n（sse-starlette），归一化后按空行分帧 */
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          let ev;
          try { ev = JSON.parse(raw); } catch { continue; }
          gotEvent = true;
          handleEvent(body, ev);
        }
      }
    }
    if (!gotEvent) errorBox(body, "服务无响应内容，请检查后端日志");
  } catch (err) {
    if (err.name === "AbortError") {
      tokenChunk(body, "\n\n（已手动停止本次回答）");
    } else {
      errorBox(body, err.message || "网络异常，请确认后端已启动");
    }
  } finally {
    finishTurn(body);
    state.sending = false;
    state.abortCtrl = null;
    setSendingUI(false);
    msgInput.focus();
  }
}

function handleEvent(body, ev) {
  switch (ev.type) {
    case "routing_decision": routeCard(body, ev); break;
    case "progress":         progressLine(body, ev.stage); break;
    case "token":            tokenChunk(body, ev.content || ""); break;
    case "guidance":         guidanceBox(body, ev); break;
    case "pipeline_plan":    pipelineCard(body, ev); break;
    case "meta":             metaLine(body, ev); break;
    case "error":            errorBox(body, ev.message); break;
    case "done":             break; /* finishTurn 统一处理 */
    default:                 break;
  }
}

function setSendingUI(sending) {
  sendBtn.disabled = sending;
  sendBtn.classList.toggle("hidden", sending);
  stopBtn.classList.toggle("hidden", !sending);
  msgInput.disabled = sending;
  attachBtn.disabled = sending;
  if (!state.micUnsupported) micBtn.disabled = sending;
  if (sending && listening) stopListening();
}

/* ═══════════════════════════════════════════
   三·五、语音输入（Web Speech API）与图片上传
   ═══════════════════════════════════════════ */

/* ── 语音输入：浏览器内置识别（Edge/Chrome），结果填入输入框 ── */
const SRClass = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let listening = false;

if (!SRClass) {
  state.micUnsupported = true;
  micBtn.disabled = true;
  micBtn.title = "当前浏览器不支持语音识别，请使用 Edge / Chrome";
}

micBtn.addEventListener("click", () => {
  if (listening) { stopListening(); return; }
  startListening();
});

function startListening() {
  recognition = new SRClass();
  recognition.lang = "zh-CN";
  recognition.interimResults = true;   // 实时上屏
  recognition.continuous = false;      // 单句结束自动停止
  const base = msgInput.value ? msgInput.value.replace(/\s+$/, "") + " " : "";

  recognition.onresult = (e) => {
    let text = "";
    for (const res of e.results) text += res[0].transcript;
    msgInput.value = base + text;
    autoGrow();
  };
  recognition.onend = () => { setListening(false); msgInput.focus(); };
  recognition.onerror = (e) => {
    setListening(false);
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      micBtn.title = "麦克风权限被拒绝，请在浏览器地址栏设置中允许";
    }
  };

  try {
    recognition.start();
    setListening(true);
  } catch {
    setListening(false);
  }
}

function stopListening() {
  if (recognition) { try { recognition.stop(); } catch { /* 忽略 */ } }
  setListening(false);
}

function setListening(on) {
  listening = on;
  micBtn.classList.toggle("listening", on);
  micBtn.title = on ? "停止录音" : "语音输入";
}

/* ── 图片上传：选图 → canvas 压缩（≤1024px JPEG）→ 预览 ── */
attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files && fileInput.files[0];
  fileInput.value = "";
  if (!file || !file.type.startsWith("image/")) return;
  try {
    state.attachImage = await fileToDataUrl(file);
    attachThumb.src = state.attachImage;
    attachPreview.classList.remove("hidden");
  } catch {
    clearAttachment();
  }
});

$("attachRemoveBtn").addEventListener("click", clearAttachment);

function clearAttachment() {
  state.attachImage = null;
  attachThumb.removeAttribute("src");
  attachPreview.classList.add("hidden");
}

/** 读文件并压缩：最长边 ≤1024px，JPEG q0.85（控制 base64 体积） */
async function fileToDataUrl(file) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = dataUrl;
  });
  const MAX = 1024;
  let { width, height } = img;
  if (Math.max(width, height) > MAX) {
    const scale = MAX / Math.max(width, height);
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(img, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", 0.85);
}

/* ═══════════════════════════════════════════
   四、会话与交互绑定
   ═══════════════════════════════════════════ */

/* ── 欢迎屏模板（含预设问题，初始载入与重置会话共用同一份标记）── */
const PRESET_QUESTIONS = [
  ["📚", "CNC-1000 的主轴怎么保养？", "CNC-1000 的主轴应该怎么保养？", ""],
  ["🔧", "报 E001，电机不转还发热", "设备报E001，电机不转还发热，过载报警一直响", ""],
  ["🔍", "主轴异响，精度变差了", "主轴运行时有异响，加工精度也变差了，帮我看看什么问题", ""],
  ["🚑", "E002 报修全流程", "设备报E002错误，变频器过流报警，启动就跳闸，需要报修并处理", "q-accent"],
  ["🛡", "SN-AC-0001 在保修期吗？", "帮我查一下 SN-AC-0001 这台设备还在保修期内吗？", ""],
  ["📦", "轴承 BRG-6204 有货吗？", "主轴轴承 BRG-6204 还有库存吗？", ""],
];

function welcomeHtml(title, sub) {
  const chips =
    '<span class="chip"><i class="dot d-k"></i>RAG 混合检索</span>' +
    '<span class="chip"><i class="dot d-d"></i>245 节点诊断树</span>' +
    '<span class="chip"><i class="dot d-t"></i>工单真实落库</span>' +
    '<span class="chip"><i class="dot d-a"></i>多轮追问</span>';
  const presets = PRESET_QUESTIONS.map(([emoji, label, msg, extra]) =>
    '<button class="q-btn ' + extra + '" data-msg="' + msg.replace(/"/g, "&quot;") + '">' +
    '<span class="q-emoji">' + emoji + "</span>" + label + "</button>"
  ).join("");
  return (
    '<div class="welcome">' +
    '  <div class="welcome-logo">' + BOT_SVG_LOGO + "</div>" +
    "  <h2>" + title + "</h2>" +
    '  <p class="welcome-sub">' + sub + "</p>" +
    '  <div class="welcome-chips">' + chips + "</div>" +
    '  <div class="preset-title">试着问问看</div>' +
    '  <div class="preset-grid">' + presets + "</div>" +
    '  <p class="hint">点击问题立即体验，或在下方输入框描述您的问题</p>' +
    "</div>"
  );
}

function resetSession() {
  if (state.sending) state.abortCtrl.abort();
  state.sessionId = newSessionId();
  chatWindow.innerHTML = welcomeHtml("设备智能服务助手", "知识问答 · 故障诊断 · 工单管理 · 售后协调 —— 一句话直达全链路");
  $("sessionIdText").textContent = state.sessionId.slice(0, 8);
}

/* 欢迎屏 logo 的内联 SVG（复用首屏样式）*/
const BOT_SVG_LOGO =
  '<svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" ' +
  'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" ' +
  'cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 ' +
  '2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 ' +
  '0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-' +
  '2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.' +
  '09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83' +
  'l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1' +
  ".65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-." +
  '06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09' +
  'a1.65 1.65 0 0 0-1.51 1z"/></svg>';

sendBtn.addEventListener("click", () => {
  const text = msgInput.value;
  msgInput.value = "";
  autoGrow();
  sendMessage(text);
});

msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = msgInput.value;
    msgInput.value = "";
    autoGrow();
    sendMessage(text);
  }
});

msgInput.addEventListener("input", autoGrow);
function autoGrow() {
  msgInput.style.height = "auto";
  msgInput.style.height = Math.min(msgInput.scrollHeight, 130) + "px";
}

stopBtn.addEventListener("click", () => state.abortCtrl && state.abortCtrl.abort());

$("newSessionBtn").addEventListener("click", resetSession);

/* 快捷场景 + 预设问题：事件委托（覆盖重置会话后重建的 DOM） */
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".scene-btn, .q-btn");
  if (btn && btn.dataset.msg) sendMessage(btn.dataset.msg);
});

/* ═══════════════════════════════════════════
   五、登录页交互
   ═══════════════════════════════════════════ */

$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = loginSubmitBtn;
  const btnText = btn.querySelector(".login-btn-text");
  const btnLoading = btn.querySelector(".login-loading");
  btn.disabled = true;
  btnText.classList.add("hidden");
  btnLoading.classList.remove("hidden");
  try {
    await login($("loginUsername").value.trim(), $("loginPassword").value);
    loginError.classList.add("hidden");
    showApp();
    /* 登录墙拦截的消息自动续发 */
    const pending = state.pendingMessage;
    state.pendingMessage = null;
    if (pending) sendMessage(pending);
  } catch (err) {
    loginError.textContent = err.message;
    loginError.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btnText.classList.remove("hidden");
    btnLoading.classList.add("hidden");
  }
});

/* 演示账号一键填入 */
document.querySelectorAll(".demo-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("loginUsername").value = chip.dataset.u;
    $("loginPassword").value = chip.dataset.p;
    loginError.classList.add("hidden");
    $("loginPassword").focus();
  });
});

/* 密码可见切换 */
$("pwdEyeBtn").addEventListener("click", () => {
  const input = $("loginPassword");
  input.type = input.type === "password" ? "text" : "password";
});

/* 退出登录 → 回登录页 */
$("logoutBtn").addEventListener("click", () => {
  if (state.sending && state.abortCtrl) state.abortCtrl.abort();
  gotoLogin("已退出登录");
});

/* ── 初始化 ─────────────────────────────── */
$("sessionIdText").textContent = state.sessionId.slice(0, 8);
initAuth();
