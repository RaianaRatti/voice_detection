const resetButton = document.getElementById("reset-btn");
const connectButton = document.getElementById("connect-btn");
const speakersDiv   = document.getElementById("speakers");
const statusDot     = document.getElementById("status-dot");
const statusText    = document.getElementById("status-text");

const SPEAKER_COLORS = [
  { border: "#ff6b6b", glow: "rgba(255,107,107,0.3)" },
  { border: "#4dabf7", glow: "rgba(77,171,247,0.3)"  },
  { border: "#51cf66", glow: "rgba(81,207,102,0.3)"  },
  { border: "#ffd43b", glow: "rgba(255,212,59,0.3)"  },
  { border: "#b197fc", glow: "rgba(177,151,252,0.3)" },
  { border: "#ffa94d", glow: "rgba(255,169,77,0.3)"  },
];

let socket = null;
let running = false;

function formatTime(seconds) {
  const ms   = Math.floor((seconds % 1) * 10); // tenths of a second
  const s    = Math.floor(seconds);
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return String(mins).padStart(2, "0") + ":" + String(secs).padStart(2, "0") + "." + ms;
}

function getOrCreateCard(speakerId) {
  const domId = `speaker-${speakerId}`;
  let card = document.getElementById(domId);

  if (!card) {
    const isSilent = speakerId === "silent";
    const idx      = isSilent ? 0 : Number(speakerId) % SPEAKER_COLORS.length;
    const color    = isSilent
      ? { border: "#6b7280", glow: "rgba(107,114,128,0.3)" }
      : SPEAKER_COLORS[idx];
    const avatar   = isSilent ? "—" : Number(speakerId) + 1;
    const label    = isSilent ? "Silent" : `Speaker ${Number(speakerId) + 1}`;

    card = document.createElement("div");
    card.className = "speaker-card";
    card.id = domId;
    card.style.setProperty("--speaker-color", color.border);
    card.style.setProperty("--speaker-glow",  color.glow);

    card.innerHTML = `
      <div class="speaker-avatar">${avatar}</div>
      <div class="speaker-info">
        <div class="speaker-label">${label}</div>
        <div class="speaker-time">00:00</div>
        <div class="speaker-bar-track"><div class="speaker-bar"></div></div>
      </div>
      <div class="speaker-active-badge">LIVE</div>
    `;

    speakersDiv.appendChild(card);

    // staggered entrance
    card.style.opacity   = "0";
    card.style.transform = "translateY(16px)";
    requestAnimationFrame(() => {
      card.style.transition = "opacity 0.4s ease, transform 0.4s ease";
      card.style.opacity    = "1";
      card.style.transform  = "translateY(0)";
    });
  }

  return card;
}

function applyState(data) {
  const current = data.current ?? null;
  const times   = data.times || {};
  const silence = data.silence ?? 0;
  const nowSilent = current === null;

  const allTimes = { ...times, silent: silence };
  const totalTime = Object.values(allTimes).reduce((a, b) => a + b, 0) || 1;

  for (const speakerId in allTimes) {
    if (allTimes[speakerId] <= 0 && speakerId !== "silent") continue;
    const card   = getOrCreateCard(speakerId);
    const timeEl = card.querySelector(".speaker-time");
    const barEl  = card.querySelector(".speaker-bar");
    const pct    = Math.round((allTimes[speakerId] / totalTime) * 100);

    timeEl.textContent = formatTime(allTimes[speakerId]);
    barEl.style.width  = pct + "%";

    // Only Silent ever gets the LIVE badge; speakers are never marked active
    const isActive = speakerId === "silent" && nowSilent;
    card.classList.toggle("active", isActive);
  }
}

function setStatus(state) {
  // state: "disconnected" | "connecting" | "live"
  statusDot.className  = "status-dot" + (state === "live" ? " live" : "");
  statusText.textContent =
    state === "live"        ? "Live — detecting speakers" :
    state === "connecting"  ? "Connecting…"               :
                              "Disconnected";
}

function disconnect(reason) {
  if (socket) {
    socket.close();
    socket = null;
  }
  running = false;
  connectButton.textContent = "Connect";
  connectButton.classList.remove("connected");
  setStatus("disconnected");

  // de-activate all cards but leave times visible
  document.querySelectorAll(".speaker-card").forEach(c => c.classList.remove("active"));

  if (reason) console.warn("WebSocket closed:", reason);
}

function connect() {
  setStatus("connecting");
  socket = new WebSocket("ws://localhost:5050/ws");

  socket.onopen = () => {
    running = true;
    connectButton.textContent = "Disconnect";
    connectButton.classList.add("connected");
    setStatus("live");
    console.log("WebSocket connected.");
  };

  socket.onmessage = (event) => {
    console.log("RAW:", event.data);
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error("Bad JSON from server:", event.data);
      return;
    }
    applyState(data);
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };

  socket.onclose = (event) => {
    disconnect(event.reason || `code ${event.code}`);
  };
}

connectButton.addEventListener("click", () => {
  if (running) {
    disconnect("user requested");
  } else {
    connect();
  }
});

resetButton.addEventListener("click", () => {
  speakersDiv.innerHTML = "";
  fetch("/reset", { method: "POST" });
});