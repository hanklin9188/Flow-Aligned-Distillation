"use strict";

const state = {
  catalog: null,
  health: null,
  selectedId: null,
  runToken: 0,
  busy: new Set(),
  results: {},
  drawing: false,
  poolDataset: "all",
};

const $ = (id) => document.getElementById(id);
const elements = {
  health: $("health"), healthLabel: $("health-label"), runtimeDetail: $("runtime-detail"),
  questionPicker: $("question-picker"), datasetFilter: $("dataset-filter"),
  drawFive: $("draw-five"), drawStatus: $("draw-status"),
  position: $("question-position"), dataset: $("dataset-badge"), sample: $("sample-badge"),
  categoryBadge: $("category-badge"), title: $("question-title"), choices: $("choices"),
  historicalNote: $("historical-note"), runBoth: $("run-both"), clear: $("clear-results"),
  runStudent: $("run-student"), runTeacher: $("run-teacher"),
  studentCard: $("student-card"), teacherCard: $("teacher-card"),
  studentAnswer: $("student-answer"), teacherAnswer: $("teacher-answer"),
  studentMetrics: $("student-metrics"), teacherMetrics: $("teacher-metrics"),
  comparison: $("comparison"), comparisonTitle: $("comparison-title"),
  comparisonDetail: $("comparison-detail"), goldAnswer: $("gold-answer"),
  studentBar: $("student-bar"), teacherBar: $("teacher-bar"),
  studentBarLabel: $("student-bar-label"), teacherBarLabel: $("teacher-bar-label"),
  localSpeedup: $("local-speedup"), localProtocol: $("local-protocol"),
  localStat: document.querySelector(".local-stat"),
  studentVram: $("student-vram"), teacherVram: $("teacher-vram"),
  sharedVram: $("shared-vram"), sharedVramPeak: $("shared-vram-peak"),
  sharedVramPercent: $("shared-vram-percent"), sharedVramBar: $("shared-vram-bar"),
  sharedVramTensors: $("shared-vram-tensors"), sharedVramBaseline: $("shared-vram-baseline"),
  sharedVramSaving: $("shared-vram-saving"),
};

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "—";
  const gib = value / (1024 ** 3);
  if (gib >= 1) return `${gib.toFixed(2)} GiB`;
  return `${(value / (1024 ** 2)).toFixed(0)} MiB`;
}

function currentQuestion() {
  return state.catalog?.questions.find((question) => question.id === state.selectedId) || null;
}

function choiceLetter(label, question = currentQuestion()) {
  const index = question?.choices.findIndex((choice) => choice.label === label) ?? -1;
  return index >= 0 ? String.fromCharCode(65 + index) : label.toUpperCase();
}

function setHealth(health, failed = false) {
  state.health = health;
  elements.health.className = `health ${failed ? "health-error" : health?.ready ? "health-ready" : "health-loading"}`;
  if (failed) {
    elements.healthLabel.textContent = "本機服務未連線";
    elements.runtimeDetail.textContent = "請確認本機 PowerShell 的 Demo server 是否仍在執行";
  } else if (health?.ready) {
    const mode = health.mode === "mock" ? "預覽模式" : "GPU Ready";
    elements.healthLabel.textContent = `${mode} · ${health.gpu}`;
    const totalVram = Number(health.vram?.device_total_bytes || 0);
    elements.runtimeDetail.textContent = `${health.gpu}${totalVram ? ` · ${formatBytes(totalVram)} VRAM` : ""} · torch ${health.torch}`;
  } else {
    elements.healthLabel.textContent = `模型準備中 · ${health?.loading_stage || "starting"}`;
  }
  updateButtons();
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    setHealth(await response.json());
  } catch (error) {
    setHealth(null, true);
  }
}

function buildDatasetFilter() {
  elements.datasetFilter.replaceChildren();
  const all = document.createElement("option");
  all.value = "all";
  all.textContent = `全部資料集 · ${Number(state.catalog.question_count).toLocaleString()} 題`;
  elements.datasetFilter.append(all);
  for (const dataset of state.catalog.datasets || []) {
    const option = document.createElement("option");
    option.value = dataset.id;
    option.textContent = `${dataset.id} · ${Number(dataset.count).toLocaleString()} 題`;
    elements.datasetFilter.append(option);
  }
  elements.datasetFilter.value = state.poolDataset;
}

function buildPicker() {
  elements.questionPicker.replaceChildren();
  for (const question of state.catalog?.questions || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "question-pill";
    button.setAttribute("aria-current", String(question.id === state.selectedId));
    button.setAttribute("aria-label", `第 ${question.draw_rank} 題，${question.dataset} sample ${question.sample_id}`);
    button.textContent = String(question.draw_rank);
    button.addEventListener("click", () => selectQuestion(question.id));
    elements.questionPicker.append(button);
  }
}

function selectQuestion(id) {
  if (state.busy.size || state.selectedId === id) return;
  state.runToken += 1;
  state.selectedId = id;
  state.results = {};
  buildPicker();
  renderQuestion();
  renderAllResults();
}

async function drawRandomFive() {
  if (state.busy.size || state.drawing || !state.catalog) return;
  state.runToken += 1;
  state.drawing = true;
  state.poolDataset = elements.datasetFilter.value || "all";
  state.selectedId = null;
  state.results = {};
  elements.drawFive.classList.add("is-rolling");
  elements.drawStatus.textContent = "正在擲骰子並抽取五題…";
  renderAllResults();
  updateButtons();
  try {
    const query = new URLSearchParams({ count: "5", dataset: state.poolDataset });
    const response = await fetch(`/api/sample?${query}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.catalog.questions = payload.questions;
    state.selectedId = payload.questions[0]?.id || null;
    const poolName = payload.pool === "all" ? "完整測試集" : payload.pool;
    elements.drawStatus.textContent = `${poolName} ${Number(payload.pool_size).toLocaleString()} 題中，均勻且不重複抽出 5 題`;
    buildPicker();
    renderQuestion();
    renderAllResults();
  } catch (error) {
    state.catalog.questions = [];
    elements.drawStatus.textContent = `抽題失敗：${error.message}`;
    elements.title.textContent = "無法取得隨機題目，請確認本機服務仍在執行。";
  } finally {
    state.drawing = false;
    elements.drawFive.classList.remove("is-rolling");
    updateButtons();
  }
}

function renderQuestion() {
  const question = currentQuestion();
  if (!question) return;
  elements.position.textContent = question.draw_rank;
  elements.dataset.textContent = question.dataset;
  elements.sample.textContent = `Sample ${question.sample_id}`;
  elements.categoryBadge.textContent = "均勻隨機 · 未篩選正誤";
  elements.title.textContent = question.display_stem;
  elements.choices.replaceChildren();
  question.choices.forEach((choice, index) => {
    const item = document.createElement("li");
    item.className = "choice";
    const letter = document.createElement("span");
    letter.className = "choice-letter";
    letter.textContent = String.fromCharCode(65 + index);
    const text = document.createElement("span");
    text.textContent = choice.text;
    item.append(letter, text);
    elements.choices.append(item);
  });
  elements.historicalNote.textContent = "本題從完整 paired test set 隨機抽出；按下公平競速後才揭曉兩模型的現場答案、正誤、速度與 VRAM。";
  elements.historicalNote.classList.remove("outlier-warning");
  updateButtons();
}

function updateButtons() {
  const ready = Boolean(state.health?.ready && state.selectedId);
  const anyBusy = state.busy.size > 0;
  elements.runBoth.disabled = !ready || anyBusy || state.drawing;
  elements.runStudent.disabled = !ready || anyBusy || state.drawing;
  elements.runTeacher.disabled = !ready || anyBusy || state.drawing;
  elements.clear.disabled = anyBusy || state.drawing;
  elements.drawFive.disabled = anyBusy || state.drawing || !state.catalog;
  elements.datasetFilter.disabled = anyBusy || state.drawing || !state.catalog;
}

function setBusy(models, busy) {
  for (const model of models) {
    if (busy) state.busy.add(model); else state.busy.delete(model);
    const card = model === "student" ? elements.studentCard : elements.teacherCard;
    card.classList.toggle("is-busy", busy);
    card.setAttribute("aria-busy", String(busy));
  }
  updateButtons();
}

function renderResult(model, result) {
  const question = currentQuestion();
  const answer = model === "student" ? elements.studentAnswer : elements.teacherAnswer;
  const metrics = model === "student" ? elements.studentMetrics : elements.teacherMetrics;
  answer.replaceChildren();
  const content = document.createElement("span");
  content.className = "answer-content";
  const label = document.createElement("span");
  label.className = "answer-label";
  label.textContent = `${choiceLetter(result.prediction, question)} · ${result.prediction}`;
  const text = document.createElement("span");
  text.className = "answer-text";
  text.textContent = result.answer_text || "模型已選擇此答案";
  content.append(label, text);
  answer.append(content);

  metrics.replaceChildren();
  const values = [
    { text: result.correct ? "✓ 答對" : "✕ 答錯", className: result.correct ? "metric-success" : "metric-error" },
    { text: result.timing ? `Median ${result.latency_ms.toFixed(2)} ms` : `本次 ${result.latency_ms.toFixed(2)} ms` },
    { text: `Exit L${result.exit_layer}` },
  ];
  if (result.timing) {
    values.push({ text: `${result.timing.repeats} 次公平實測` });
    values.push({ text: `P25–P75 ${result.timing.p25_ms.toFixed(1)}–${result.timing.p75_ms.toFixed(1)} ms` });
  }
  if (Number.isFinite(result.confidence)) values.push({ text: `信心 ${(100 * result.confidence).toFixed(1)}%` });
  if (result.source === "historical_mock") values.push({ text: "預覽資料" });
  for (const value of values) {
    const chip = document.createElement("span");
    chip.className = `metric ${value.className || ""}`;
    chip.textContent = value.text;
    metrics.append(chip);
  }
  renderVram(model, result);
}

function renderVram(model, result) {
  const panel = model === "student" ? elements.studentVram : elements.teacherVram;
  const vram = result?.vram;
  if (!vram || !Number.isFinite(Number(vram.model_tensor_bytes))) {
    panel.hidden = true;
    return;
  }
  const prefix = `${model}-vram`;
  const modelBytes = Number(vram.model_tensor_bytes);
  const total = Math.max(1, Number(vram.device_total_bytes));
  const percent = Math.min(100, Math.max(0, 100 * modelBytes / total));
  $(`${prefix}-model`).textContent = formatBytes(modelBytes);
  $(`${prefix}-percent`).textContent = `${percent.toFixed(1)}% of GPU capacity`;
  $(`${prefix}-extra`).textContent = formatBytes(vram.inference_peak_extra_bytes);
  $(`${prefix}-bar`).style.width = `${Math.max(2, percent)}%`;
  panel.hidden = false;
}

function renderSharedVram() {
  const vrams = Object.values(state.results)
    .map((result) => result?.vram)
    .filter((vram) => vram && Number.isFinite(Number(vram.process_peak_allocated_bytes)));
  if (!vrams.length) {
    elements.sharedVram.hidden = true;
    return;
  }
  const total = Math.max(1, ...vrams.map((vram) => Number(vram.device_total_bytes) || 0));
  const peak = Math.max(...vrams.map((vram) => Number(vram.process_peak_allocated_bytes) || 0));
  const baseline = Math.max(...vrams.map((vram) => Number(vram.process_baseline_bytes) || 0));
  const healthVram = state.health?.vram || {};
  const studentTensor = Number(state.results.student?.vram?.model_tensor_bytes
    ?? healthVram.student_model_tensor_bytes ?? 0);
  const teacherTensor = Number(state.results.teacher?.vram?.model_tensor_bytes
    ?? healthVram.teacher_model_tensor_bytes ?? 0);
  const tensorTotal = studentTensor + teacherTensor;
  const saving = Math.max(0, teacherTensor - studentTensor);
  const savingPercent = teacherTensor > 0 ? 100 * saving / teacherTensor : 0;
  const percent = Math.min(100, Math.max(0, 100 * peak / total));
  elements.sharedVramPeak.textContent = `${formatBytes(peak)} / ${formatBytes(total)}`;
  elements.sharedVramPercent.textContent = `${percent.toFixed(1)}% of GPU`;
  elements.sharedVramBar.style.width = `${Math.max(2, percent)}%`;
  elements.sharedVramTensors.textContent = formatBytes(tensorTotal);
  elements.sharedVramBaseline.textContent = formatBytes(baseline);
  elements.sharedVramSaving.textContent = teacherTensor > 0
    ? `${formatBytes(saving)} · ${savingPercent.toFixed(1)}%`
    : "—";
  elements.sharedVram.hidden = false;
}

function renderEmpty(model) {
  const answer = model === "student" ? elements.studentAnswer : elements.teacherAnswer;
  const metrics = model === "student" ? elements.studentMetrics : elements.teacherMetrics;
  answer.replaceChildren();
  const placeholder = document.createElement("span");
  placeholder.className = "answer-placeholder";
  placeholder.textContent = "等待作答";
  answer.append(placeholder);
  metrics.replaceChildren();
  const empty = document.createElement("span");
  empty.className = "metric-empty";
  empty.textContent = "尚未執行推論";
  metrics.append(empty);
  const panel = model === "student" ? elements.studentVram : elements.teacherVram;
  panel.hidden = true;
}

function renderComparison() {
  const student = state.results.student;
  const teacher = state.results.teacher;
  if (!student || !teacher) {
    elements.comparison.hidden = true;
    elements.localSpeedup.textContent = "等待實測";
    elements.localProtocol.textContent = "各模型獨立暖機 · 交替順序 · 5 次取 median";
    elements.localStat.classList.remove("is-faster", "is-slower");
    return;
  }
  const difference = teacher.latency_ms - student.latency_ms;
  const studentFaster = difference >= 0;
  const fasterName = studentFaster ? "Student" : "Teacher";
  const speedup = studentFaster
    ? teacher.latency_ms / student.latency_ms
    : student.latency_ms / teacher.latency_ms;
  const statisticLabel = student.timing && teacher.timing ? "median" : "本次";
  elements.comparisonTitle.textContent = `${fasterName} ${statisticLabel} 快 ${Math.abs(difference).toFixed(2)} ms`;
  elements.localSpeedup.textContent = `${fasterName} ${speedup.toFixed(2)}×`;
  elements.localProtocol.textContent = `${student.timing?.repeats || 1} 次 RTX 本機實測 · ${statisticLabel} latency`;
  elements.localStat.classList.toggle("is-faster", studentFaster);
  elements.localStat.classList.toggle("is-slower", !studentFaster);
  const computeSaved = 100 * (1 - student.exit_layer / Math.max(1, teacher.exit_layer));
  if (student.correct && !teacher.correct) {
    elements.comparisonDetail.textContent = `Student 答對、Teacher 答錯；Student 本題少執行 ${computeSaved.toFixed(1)}% decoder layers。`;
  } else if (student.correct && teacher.correct) {
    elements.comparisonDetail.textContent = `兩個模型都答對；Student 本題少執行 ${computeSaved.toFixed(1)}% decoder layers。`;
  } else {
    elements.comparisonDetail.textContent = `本次結果：Student ${student.correct ? "答對" : "答錯"}、Teacher ${teacher.correct ? "答對" : "答錯"}。`;
  }
  elements.goldAnswer.textContent = `正確答案：${choiceLetter(student.gold_answer)} · ${student.gold_answer}`;
  const maximum = Math.max(student.latency_ms, teacher.latency_ms, 0.001);
  elements.studentBar.style.width = `${Math.max(3, 100 * student.latency_ms / maximum)}%`;
  elements.teacherBar.style.width = `${Math.max(3, 100 * teacher.latency_ms / maximum)}%`;
  elements.studentBarLabel.textContent = `${student.latency_ms.toFixed(2)} ms`;
  elements.teacherBarLabel.textContent = `${teacher.latency_ms.toFixed(2)} ms`;
  elements.comparison.hidden = false;
}

function renderAllResults() {
  for (const model of ["student", "teacher"]) {
    if (state.results[model]) renderResult(model, state.results[model]); else renderEmpty(model);
  }
  renderSharedVram();
  renderComparison();
}

async function runModels(models) {
  if (!currentQuestion() || state.busy.size) return;
  const token = ++state.runToken;
  const questionId = state.selectedId;
  setBusy(models, true);
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: questionId, models, benchmark: true }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
    if (token !== state.runToken || questionId !== state.selectedId) return;
    for (const [model, result] of Object.entries(payload.results)) state.results[model] = result;
    renderAllResults();
  } catch (error) {
    if (token !== state.runToken) return;
    for (const model of models) {
      const answer = model === "student" ? elements.studentAnswer : elements.teacherAnswer;
      answer.textContent = `推論失敗：${error.message}`;
    }
  } finally {
    if (token === state.runToken) setBusy(models, false);
  }
}

function clearResults() {
  if (state.busy.size) return;
  state.runToken += 1;
  state.results = {};
  renderAllResults();
}

async function initialize() {
  // Only lightweight metadata is embedded. Question text remains on the local
  // server and each dice roll transfers exactly five randomly selected items.
  refreshHealth();
  try {
    const embedded = document.getElementById("question-catalog")?.textContent || "";
    if (embedded && !embedded.includes("__QUESTION_CATALOG_JSON__")) {
      state.catalog = JSON.parse(embedded);
    } else {
      const response = await fetch("/api/questions", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.catalog = await response.json();
    }
    state.catalog.questions = [];
    buildDatasetFilter();
    await drawRandomFive();
  } catch (error) {
    elements.title.textContent = `題庫載入失敗：${error.message}`;
  }
}

elements.runBoth.addEventListener("click", () => runModels(["student", "teacher"]));
elements.runStudent.addEventListener("click", () => runModels(["student"]));
elements.runTeacher.addEventListener("click", () => runModels(["teacher"]));
elements.clear.addEventListener("click", clearResults);
elements.drawFive.addEventListener("click", drawRandomFive);
window.addEventListener("keydown", (event) => {
  if (state.busy.size || state.drawing || event.ctrlKey || event.metaKey || event.altKey) return;
  const index = Number(event.key) - 1;
  const questions = state.catalog?.questions || [];
  if (index >= 0 && index < questions.length) selectQuestion(questions[index].id);
});

initialize();
setInterval(refreshHealth, 10000);
