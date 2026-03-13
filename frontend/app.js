const chatLog = document.getElementById("chat-log");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const sampleBtn = document.getElementById("sample-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const messageTemplate = document.getElementById("message-template");

const sampleQuestion =
  "Plan a 4-day Europe city break for Rome with food spots, transport tips, and a mid-range budget.";

function addMessage(role, text) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  const roleEl = node.querySelector(".message-role");
  const contentEl = node.querySelector(".message-content");

  node.classList.add(role);
  roleEl.textContent = role === "user" ? "Traveler" : "Advisor";
  contentEl.textContent = text;

  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setStatus(mode, text) {
  statusDot.classList.remove("ready", "error");
  if (mode === "ready") statusDot.classList.add("ready");
  if (mode === "error") statusDot.classList.add("error");
  statusText.textContent = text;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health check failed");

    const data = await res.json();
    if (data.ready) {
      setStatus("ready", "Backend ready");
    } else {
      setStatus("", "Backend initializing...");
    }
  } catch {
    setStatus("error", "Backend unreachable");
  }
}

async function askQuestion(question) {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || "Request failed");
  }

  return data.answer;
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Thinking...";

  try {
    const answer = await askQuestion(question);
    addMessage("assistant", answer || "No answer generated.");
  } catch (error) {
    addMessage("assistant", `Error: ${error.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Ask Advisor";
    questionInput.focus();
  }
});

sampleBtn.addEventListener("click", () => {
  questionInput.value = sampleQuestion;
  questionInput.focus();
});

addMessage(
  "assistant",
  "Welcome. Ask a travel question and I will answer using the loaded Wikivoyage global travel knowledge base."
);

checkHealth();
setInterval(checkHealth, 15000);
