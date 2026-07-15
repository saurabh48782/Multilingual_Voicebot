// Multilingual Voicebot frontend
(() => {
  const SESSION_KEY = "voicebot_session_id";

  const $ = (id) => document.getElementById(id);

  const elements = {
    micBtn: $("mic-btn"),
    micIcon: $("mic-icon"),
    micStatus: $("mic-status"),
    fileInput: $("file-input"),
    textForm: $("text-form"),
    textInput: $("text-input"),
    langSelect: $("lang-select"),
    sessionId: $("session-id"),
    errorBanner: $("error-banner"),
    loading: $("loading"),
    loadingMsg: $("loading-msg"),
    // conversation
    chatLog: $("chat-log"),
    chatEmpty: $("chat-empty"),
    clearChatBtn: $("clear-chat-btn"),
    // sidebar
    newChatBtn: $("new-chat-btn"),
    sessionList: $("session-list"),
    sessionEmpty: $("session-empty"),
    // details
    detailsCard: $("details-card"),
    detectedLang: $("detected-lang"),
    responseTextEn: $("response-text-en"),
    fallbackNotice: $("fallback-notice"),
    fallbackReason: $("fallback-reason"),
    confidencePill: $("confidence-pill"),
    contextCount: $("context-count"),
    contextList: $("context-list"),
  };

  // --- session id (mutable; one active chat at a time) ---
  function newId() {
    return (
      (crypto.randomUUID && crypto.randomUUID()) ||
      `s-${Date.now()}-${Math.random().toString(36).slice(2)}`
    );
  }
  function getActiveSessionId() {
    let id = localStorage.getItem(SESSION_KEY);
    if (!id) {
      id = newId();
      localStorage.setItem(SESSION_KEY, id);
    }
    return id;
  }
  function setActiveSessionId(id) {
    localStorage.setItem(SESSION_KEY, id);
    sessionId = id;
    elements.sessionId.textContent = id;
  }
  let sessionId = getActiveSessionId();
  elements.sessionId.textContent = sessionId;

  // --- helpers ---
  function showError(msg) {
    elements.errorBanner.textContent = msg;
    elements.errorBanner.classList.remove("d-none");
    setTimeout(() => elements.errorBanner.classList.add("d-none"), 6000);
  }

  function setLoading(on, msg) {
    elements.loading.classList.toggle("d-none", !on);
    elements.loadingMsg.textContent = msg || "Processing...";
  }

  // --- conversation transcript ---
  function clearChatLog() {
    elements.chatLog.querySelectorAll(".msg").forEach((n) => n.remove());
    elements.chatEmpty.classList.remove("d-none");
    elements.detailsCard.classList.add("d-none");
  }

  function appendBubble(role, text, audioUrl) {
    elements.chatEmpty.classList.add("d-none");
    const wrap = document.createElement("div");
    wrap.className = `msg msg-${role}`;

    const label = document.createElement("div");
    label.className = "msg-role";
    label.textContent = role === "user" ? "You" : "Bot";

    const body = document.createElement("div");
    body.className = "msg-body";
    body.setAttribute("dir", "auto");
    body.textContent = text || (role === "user" ? "(no transcript)" : "");

    wrap.appendChild(label);
    wrap.appendChild(body);

    if (audioUrl) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = audioUrl;
      audio.className = "w-100 mt-2";
      wrap.appendChild(audio);
    }

    elements.chatLog.appendChild(wrap);
    elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  }

  function renderDetails(data) {
    elements.detailsCard.classList.remove("d-none");
    elements.detectedLang.textContent = data.detected_language || "?";
    elements.responseTextEn.textContent = data.response_text_english || "";

    if (data.fallback_triggered) {
      elements.fallbackNotice.classList.remove("d-none");
      elements.fallbackReason.textContent = data.fallback_reason || "low confidence";
    } else {
      elements.fallbackNotice.classList.add("d-none");
    }

    const overall = data.confidence?.overall || "red";
    elements.confidencePill.className = `badge ${overall}`;
    elements.confidencePill.textContent = overall.toUpperCase();

    const chunks = data.retrieved_context || [];
    elements.contextCount.textContent = chunks.length;
    elements.contextList.innerHTML = "";
    for (const c of chunks) {
      const div = document.createElement("div");
      div.className = "context-chunk";
      const score = typeof c.score === "number" ? c.score.toFixed(3) : "?";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `${c.source || "?"} · page ${c.page_num ?? "?"} · score ${score}`;
      const body = document.createElement("div");
      body.textContent = c.text || "";
      div.appendChild(meta);
      div.appendChild(body);
      elements.contextList.appendChild(div);
    }
  }

  function renderResponse(data) {
    appendBubble("user", data.transcript);
    appendBubble("assistant", data.response_text, data.audio_url);
    renderDetails(data);
    loadSessions(); // title / ordering may have changed
  }

  // --- API key (only needed when the server sets VOICEBOT_API_KEY) ---
  // The static page is served open, but /api/* requires X-API-Key when the
  // server is locked down. We keep the user-supplied key in localStorage and
  // attach it to every /api call; on a 401 we prompt for it and retry once.
  const API_KEY_KEY = "voicebot_api_key";
  const getApiKey = () => localStorage.getItem(API_KEY_KEY) || "";
  const setApiKey = (k) =>
    k ? localStorage.setItem(API_KEY_KEY, k) : localStorage.removeItem(API_KEY_KEY);

  function _rawApiFetch(url, options = {}) {
    const opts = { ...options };
    const headers = new Headers(options.headers || {});
    const key = getApiKey();
    if (key) headers.set("X-API-Key", key);
    opts.headers = headers;
    return fetch(url, opts);
  }

  async function apiFetch(url, options = {}) {
    let res = await _rawApiFetch(url, options);
    if (res.status === 401) {
      const entered = window.prompt(
        "This voicebot requires an API key. Enter your X-API-Key:",
        getApiKey()
      );
      if (entered !== null) {
        setApiKey(entered.trim());
        res = await _rawApiFetch(url, options);
      }
    }
    return res;
  }

  async function getJSON(url) {
    const res = await apiFetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function postJSON(url, body) {
    const res = await apiFetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
    }
    return res.json();
  }

  async function postForm(url, formData) {
    const res = await apiFetch(url, { method: "POST", body: formData });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${txt || res.statusText}`);
    }
    return res.json();
  }

  // --- session list / resume ---
  function fmtWhen(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    const now = new Date();
    const sameDay = d.toDateString() === now.toDateString();
    return sameDay
      ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString();
  }

  async function loadSessions() {
    let sessions = [];
    try {
      const data = await getJSON("/api/sessions");
      sessions = data.sessions || [];
    } catch {
      // metadata unavailable (e.g. no Postgres) - sidebar stays empty
    }
    elements.sessionList.querySelectorAll(".session-item").forEach((n) => n.remove());
    elements.sessionEmpty.classList.toggle("d-none", sessions.length > 0);

    for (const s of sessions) {
      const item = document.createElement("div");
      item.className = "session-item";
      if (s.session_id === sessionId) item.classList.add("active");

      const main = document.createElement("button");
      main.type = "button";
      main.className = "session-open";
      const title = document.createElement("div");
      title.className = "session-title";
      title.textContent = s.title || "Untitled chat";
      const when = document.createElement("div");
      when.className = "session-when";
      when.textContent = fmtWhen(s.last_active);
      main.appendChild(title);
      main.appendChild(when);
      main.addEventListener("click", () => selectSession(s.session_id));

      const del = document.createElement("button");
      del.type = "button";
      del.className = "session-del";
      del.title = "Delete chat";
      del.textContent = "×";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(s.session_id, s.title);
      });

      item.appendChild(main);
      item.appendChild(del);
      elements.sessionList.appendChild(item);
    }
  }

  async function loadHistory(id) {
    clearChatLog();
    let messages = [];
    try {
      // An unknown/new session_id returns 200 with an empty messages list -
      // any thrown error here is a genuine failure (network/server), not an
      // empty session, so it must surface rather than be swallowed silently.
      const data = await getJSON(`/api/sessions/${encodeURIComponent(id)}`);
      messages = data.messages || [];
    } catch (err) {
      showError(`Could not load chat history: ${err.message}`);
      return;
    }
    for (const m of messages) {
      appendBubble(m.role === "assistant" ? "assistant" : "user", m.content);
    }
  }

  async function selectSession(id) {
    if (id !== sessionId) setActiveSessionId(id);
    await loadHistory(id);
    loadSessions();
  }

  function startNewChat() {
    setActiveSessionId(newId());
    clearChatLog();
    loadSessions();
    elements.textInput?.focus();
  }

  async function deleteSession(id, title) {
    if (!confirm(`Delete "${title || "this chat"}"? This cannot be undone.`)) return;
    try {
      const res = await apiFetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      showError(`Delete failed: ${err.message}`);
      return;
    }
    if (id === sessionId) {
      startNewChat();
    } else {
      loadSessions();
    }
  }

  async function clearCurrentChat() {
    if (!confirm("Clear this conversation's history?")) return;
    try {
      const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/reset`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      showError(`Reset failed: ${err.message}`);
      return;
    }
    clearChatLog();
    loadSessions();
  }

  // --- voice (MediaRecorder) ---
  let mediaRecorder = null;
  let recChunks = [];

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Microphone API not available in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
      recChunks = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recChunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recChunks, { type: mime });
        await uploadAudio(blob, "recording.webm");
      };
      mediaRecorder.start();
      elements.micBtn.classList.add("recording");
      elements.micIcon.textContent = "⏹";
      elements.micStatus.textContent = "Recording... tap to stop";
    } catch (err) {
      showError(`Mic error: ${err.message}`);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    elements.micBtn.classList.remove("recording");
    elements.micIcon.textContent = "🎤";
    elements.micStatus.textContent = "Tap mic to start recording";
  }

  async function uploadAudio(blob, filename) {
    setLoading(true, "Transcribing & answering...");
    try {
      const fd = new FormData();
      fd.append("audio", blob, filename);
      fd.append("session_id", sessionId);
      const lang = elements.langSelect.value;
      if (lang) fd.append("language", lang);
      const data = await postForm("/api/voice", fd);
      renderResponse(data);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  }

  elements.micBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  });

  elements.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadAudio(file, file.name);
    e.target.value = "";
  });

  elements.textForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = elements.textInput.value.trim();
    if (!text) return;
    setLoading(true, "Answering...");
    try {
      const body = { text, session_id: sessionId };
      const lang = elements.langSelect.value;
      if (lang) body.language = lang;
      const data = await postJSON("/api/chat", body);
      renderResponse(data);
      elements.textInput.value = "";
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  });

  // --- document upload ---
  const docFileInput = $("doc-file-input");
  const docUploadBtn = $("doc-upload-btn");
  const docStatus = $("doc-status");

  function setDocStatus(type, html) {
    docStatus.className = `mt-3 alert alert-${type}`;
    docStatus.innerHTML = html;
    docStatus.classList.remove("d-none");
  }

  docFileInput.addEventListener("change", () => {
    docUploadBtn.disabled = !docFileInput.files?.length;
    docStatus.classList.add("d-none");
  });

  docUploadBtn.addEventListener("click", async () => {
    const file = docFileInput.files?.[0];
    if (!file) return;
    docUploadBtn.disabled = true;
    setDocStatus("secondary", `<span class="spinner-border spinner-border-sm me-2"></span>Indexing <strong>${file.name}</strong>…`);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const res = await apiFetch("/api/documents", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setDocStatus("success", `<strong>${data.filename}</strong> indexed - ${data.chunks_added} chunks ready.`);
      docFileInput.value = "";
    } catch (err) {
      setDocStatus("danger", `Upload failed: ${err.message}`);
    } finally {
      docUploadBtn.disabled = false;
    }
  });

  elements.newChatBtn.addEventListener("click", startNewChat);
  elements.clearChatBtn.addEventListener("click", clearCurrentChat);

  // --- boot: resume the active chat + populate the sidebar ---
  (async () => {
    await loadHistory(sessionId);
    await loadSessions();
  })();
})();
