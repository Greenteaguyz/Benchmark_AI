/**
 * LLM Reasoning Benchmark Suite — Client-Side Application Logic
 * Pure Vanilla JavaScript • Zero Page Reloads • Real-Time Token Streaming
 */

// --- 1. Global Application State ---
const state = {
  mode: 'test', // 'test' or 'official'
  online: false,
  installedModels: [],
  loadedInVram: [],
  models: {},
  questions: {},
  frozenSettings: {},
  selectedModel: '',
  selectedQid: 'Q01',
  activeCategory: 'All',
  questionSourceTab: 'frozen', // 'frozen' or 'custom'
  isGenerating: false,
  autoLoadVram: false,
};

// --- 2. DOM Elements Cache ---
const DOM = {
  // Navigation & Badges
  modeBtnTest: document.getElementById('modeBtnTest'),
  modeBtnOfficial: document.getElementById('modeBtnOfficial'),
  modeBanner: document.getElementById('modeBanner'),
  modeBannerText: document.getElementById('modeBannerText'),
  ollamaStatusDot: document.getElementById('ollamaStatusDot'),
  ollamaStatusText: document.getElementById('ollamaStatusText'),
  vramTelemetryText: document.getElementById('vramTelemetryText'),

  // Progress & Rubric
  progressBarFill: document.getElementById('progressBarFill'),
  progressBadge: document.getElementById('progressBadge'),
  matrixTableBody: document.getElementById('matrixTableBody'),

  // Model Controls
  modelSelect: document.getElementById('modelSelect'),
  modelStatusTag: document.getElementById('modelStatusTag'),
  modelDescriptionText: document.getElementById('modelDescriptionText'),
  autoLoadVramCheckbox: document.getElementById('autoLoadVramCheckbox'),
  loadVramBtn: document.getElementById('loadVramBtn'),
  evictVramBtn: document.getElementById('evictVramBtn'),
  pullModelBtn: document.getElementById('pullModelBtn'),

  // Question Controls
  tabFrozenQuestions: document.getElementById('tabFrozenQuestions'),
  tabCustomPrompt: document.getElementById('tabCustomPrompt'),
  frozenQuestionContainer: document.getElementById('frozenQuestionContainer'),
  customQuestionContainer: document.getElementById('customQuestionContainer'),
  categoryFilter: document.getElementById('categoryFilter'),
  questionSelect: document.getElementById('questionSelect'),
  frozenPromptText: document.getElementById('frozenPromptText'),
  customQidInput: document.getElementById('customQidInput'),
  customPromptText: document.getElementById('customPromptText'),
  questionMetaBadges: document.getElementById('questionMetaBadges'),

  // Hyperparameters
  tempSlider: document.getElementById('tempSlider'),
  tempValueLabel: document.getElementById('tempValueLabel'),
  numPredictInput: document.getElementById('numPredictInput'),
  numCtxInput: document.getElementById('numCtxInput'),
  paramModeBadge: document.getElementById('paramModeBadge'),
  officialLockedNotice: document.getElementById('officialLockedNotice'),

  // Run & Output
  runBenchmarkBtn: document.getElementById('runBenchmarkBtn'),
  outputSection: document.getElementById('outputSection'),
  outputMetaBadge: document.getElementById('outputMetaBadge'),
  metricTotalTime: document.getElementById('metricTotalTime'),
  metricTokens: document.getElementById('metricTokens'),
  metricSpeed: document.getElementById('metricSpeed'),
  metricVram: document.getElementById('metricVram'),
  metricStatus: document.getElementById('metricStatus'),
  thinkDrawer: document.getElementById('thinkDrawer'),
  thinkHeader: document.getElementById('thinkHeader'),
  thinkContent: document.getElementById('thinkContent'),
  thinkToggleIcon: document.getElementById('thinkToggleIcon'),
  finalAnswerContainer: document.getElementById('finalAnswerContainer'),
  savedArtifactBanner: document.getElementById('savedArtifactBanner'),
  savedArtifactPath: document.getElementById('savedArtifactPath'),

  // Inspector
  logsTableBody: document.getElementById('logsTableBody'),
  refreshLogsBtn: document.getElementById('refreshLogsBtn'),
  toastContainer: document.getElementById('toastContainer'),

  // Modal Pop-out Elements
  artifactModal: document.getElementById('artifactModal'),
  modalTitle: document.getElementById('modalTitle'),
  modalBadge: document.getElementById('modalBadge'),
  modalSubtitle: document.getElementById('modalSubtitle'),
  modalPromptText: document.getElementById('modalPromptText'),
  modalMetricsBar: document.getElementById('modalMetricsBar'),
  modalThinkDrawer: document.getElementById('modalThinkDrawer'),
  modalThinkHeader: document.getElementById('modalThinkHeader'),
  modalThinkContent: document.getElementById('modalThinkContent'),
  modalThinkToggleIcon: document.getElementById('modalThinkToggleIcon'),
  modalOutputContainer: document.getElementById('modalOutputContainer'),
  modalFilePath: document.getElementById('modalFilePath'),
  modalCopyBtn: document.getElementById('modalCopyBtn'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalCloseFooterBtn: document.getElementById('modalCloseFooterBtn'),
};

// --- 3. UI Toast Notification Utility ---
function showToast(message, duration = 3000) {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerText = message;
  DOM.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// --- 4. Initialization & Data Fetching ---
async function initApp() {
  try {
    // 1. Fetch metadata
    const metaRes = await fetch('/api/metadata');
    const metaData = await metaRes.json();
    state.models = metaData.models || {};
    state.questions = metaData.questions || {};
    state.frozenSettings = metaData.frozen_settings || {};

    // 2. Fetch system status
    await refreshStatus();

    // 3. Populate model and question selectors
    populateModelSelector();
    populateQuestionSelector();
    renderActiveQuestion();

    // 4. Fetch progress and logs
    await refreshProgress();
    await refreshLogs();

    // 5. Setup periodic background polling (every 5 seconds)
    setInterval(backgroundPoll, 5000);

  } catch (err) {
    console.error('Initialization error:', err);
    showToast('⚠️ Error initializing application: ' + err.message);
  }
}

async function backgroundPoll() {
  if (!state.isGenerating) {
    await refreshStatus();
    await refreshProgress();
  }
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    state.online = data.online;
    state.installedModels = data.installed_models || [];
    state.loadedInVram = data.loaded_in_vram || [];

    // Update Status Pill
    if (state.online) {
      DOM.ollamaStatusDot.className = 'status-dot online';
      DOM.ollamaStatusText.innerText = 'Ollama Online';
    } else {
      DOM.ollamaStatusDot.className = 'status-dot offline';
      DOM.ollamaStatusText.innerText = 'Ollama Offline';
    }

    // Update VRAM telemetry
    DOM.vramTelemetryText.innerText = `${Math.round(data.current_vram_mb || 0)} MB VRAM`;

    // Refresh model status buttons and labels
    updateModelControls();

  } catch (err) {
    DOM.ollamaStatusDot.className = 'status-dot offline';
    DOM.ollamaStatusText.innerText = 'Offline';
  }
}

// --- Modal Pop-out State & Controller ---
let currentModalAnswerText = '';

function openModal(data) {
  const qid = data.question_id || 'Detail';
  const modelName = data.model_alias || data.model || 'Model';
  DOM.modalTitle.innerText = `${qid} — ${modelName}`;

  const qInfo = state.questions[qid] || {};
  const category = qInfo.category || (data.model_alias ? 'Reasoning Benchmark' : 'Run Details');
  DOM.modalBadge.innerText = category;
  DOM.modalSubtitle.innerText = data.timestamp ? `Logged at ${data.timestamp} • Mode: ${state.mode.toUpperCase()}` : `Mode: ${state.mode.toUpperCase()}`;

  // 1. Input Prompt Box
  const promptText = data.prompt || qInfo.prompt || '(No prompt recorded)';
  renderRichContent(DOM.modalPromptText, promptText);

  // 2. Metrics Bar
  const metrics = data.metrics || {};
  let pills = '';
  if (metrics.total_duration_s !== undefined) {
    const genS = metrics.generation_duration_s || metrics.eval_duration_s;
    const durLabel = genS ? `${metrics.total_duration_s}s (Gen: ${genS}s)` : `${metrics.total_duration_s}s`;
    pills += `<div class="modal-metric-pill">⏱️ Total: <b>${durLabel}</b></div>`;
  }
  if (metrics.output_tokens !== undefined) {
    pills += `<div class="modal-metric-pill">🔢 Tokens: <b>${metrics.output_tokens}</b></div>`;
  }
  if (metrics.tokens_per_sec !== undefined) {
    pills += `<div class="modal-metric-pill">⚡ Speed: <b>${metrics.tokens_per_sec} tok/s</b></div>`;
  }
  const vramVal = metrics.peak_vram_mb !== undefined && metrics.peak_vram_mb !== null ? metrics.peak_vram_mb : (metrics.peak_vram_gb ? metrics.peak_vram_gb * 1024 : null);
  if (vramVal !== null && vramVal !== undefined) {
    pills += `<div class="modal-metric-pill">💾 Peak VRAM: <b>${Math.round(vramVal)} MB</b></div>`;
  }
  if (metrics.completion_status) {
    const isSuccess = metrics.completion_status === 'Completed';
    pills += `<div class="modal-metric-pill">📌 Status: <b style="color: ${isSuccess ? 'var(--accent-emerald)' : 'var(--accent-amber)'}">${metrics.completion_status}</b></div>`;
  }
  DOM.modalMetricsBar.innerHTML = pills;

  // 3. Past Thinking Process (<think>)
  if (data.thought && data.thought.trim()) {
    DOM.modalThinkDrawer.style.display = 'block';
    DOM.modalThinkContent.classList.remove('collapsed');
    DOM.modalThinkToggleIcon.innerText = '▼';
    renderRichContent(DOM.modalThinkContent, data.thought.trim());
  } else {
    DOM.modalThinkDrawer.style.display = 'none';
    DOM.modalThinkContent.innerHTML = '';
  }

  // 4. Final Output with Math & Markdown
  const answer = data.final_ans || data.raw_response || '';
  currentModalAnswerText = answer;
  renderRichContent(DOM.modalOutputContainer, answer);

  // 5. Filepath Footer
  DOM.modalFilePath.innerText = data.filepath || '';

  // 6. Display Modal Pop-out
  DOM.artifactModal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  if (DOM.artifactModal) {
    DOM.artifactModal.style.display = 'none';
    document.body.style.overflow = '';
  }
}

// Attach Modal Event Listeners
if (DOM.modalCloseBtn) {
  DOM.modalCloseBtn.addEventListener('click', closeModal);
}
if (DOM.modalCloseFooterBtn) {
  DOM.modalCloseFooterBtn.addEventListener('click', closeModal);
}
if (DOM.artifactModal) {
  DOM.artifactModal.addEventListener('click', (e) => {
    if (e.target === DOM.artifactModal) {
      closeModal();
    }
  });
}
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && DOM.artifactModal && DOM.artifactModal.style.display !== 'none') {
    closeModal();
  }
});

// Thinking Drawer Toggles
if (DOM.modalThinkHeader) {
  DOM.modalThinkHeader.addEventListener('click', () => {
    const isCollapsed = DOM.modalThinkContent.classList.toggle('collapsed');
    DOM.modalThinkToggleIcon.innerText = isCollapsed ? '▶' : '▼';
  });
}
if (DOM.thinkHeader) {
  DOM.thinkHeader.addEventListener('click', () => {
    const isCollapsed = DOM.thinkContent.classList.toggle('collapsed');
    DOM.thinkToggleIcon.innerText = isCollapsed ? '▶' : '▼';
  });
}

// Copy Answer to Clipboard
if (DOM.modalCopyBtn) {
  DOM.modalCopyBtn.addEventListener('click', async () => {
    if (!currentModalAnswerText) {
      showToast('No answer content to copy');
      return;
    }
    try {
      await navigator.clipboard.writeText(currentModalAnswerText);
      showToast('📋 Copied answer to clipboard!');
    } catch (err) {
      const ta = document.createElement('textarea');
      ta.value = currentModalAnswerText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      showToast('📋 Copied answer to clipboard!');
    }
  });
}

async function refreshProgress() {
  try {
    const res = await fetch('/api/progress');
    const data = await res.json();

    DOM.progressBadge.innerText = `${data.official_count} / ${data.total_target} (${data.official_percentage}%)`;
    DOM.progressBarFill.style.width = `${data.official_percentage}%`;

    // Render Matrix Table
    const matrix = state.mode === 'official' ? data.official_matrix : data.test_matrix;
    DOM.matrixTableBody.innerHTML = '';
    matrix.forEach(row => {
      const tr = document.createElement('tr');
      const aliases = [
        { key: 'PHI', name: 'phi4-mini-reasoning' },
        { key: 'DSR1', name: 'deepseek-r1:7b' },
        { key: 'QWEN', name: 'qwen3:8b' }
      ];

      let colsHtml = `<td class="qid">${row.qid}</td>`;
      aliases.forEach(a => {
        if (row[a.key]) {
          colsHtml += `<td class="matrix-cell-done" data-qid="${row.qid}" data-alias="${a.key}" data-model="${a.name}" title="Click to pop out past thinking & output" style="cursor: pointer; background: rgba(16, 185, 129, 0.14); transition: background 0.15s;">✅</td>`;
        } else {
          colsHtml += `<td style="color: var(--text-muted); opacity: 0.4;">❌</td>`;
        }
      });
      tr.innerHTML = colsHtml;

      // Add click listeners to done cells to pop out modal
      tr.querySelectorAll('.matrix-cell-done').forEach(td => {
        td.addEventListener('click', async (e) => {
          e.stopPropagation();
          const qid = td.getAttribute('data-qid');
          const alias = td.getAttribute('data-alias');
          const modelName = td.getAttribute('data-model');
          const fileName = `${qid}_${alias}.json`;

          try {
            const artRes = await fetch(`/api/artifact?mode=${state.mode}&file=${encodeURIComponent(fileName)}`);
            if (artRes.ok) {
              const artData = await artRes.json();
              openModal({
                question_id: qid,
                model: artData.model || modelName,
                model_alias: alias,
                prompt: artData.prompt,
                thought: parseThought(artData.response),
                final_ans: parseAnswer(artData.response),
                raw_response: artData.response,
                timestamp: artData.timestamp,
                filepath: fileName,
                metrics: artData.metrics || {
                  completion_status: 'Completed',
                }
              });
            } else {
              showToast(`⚠️ Artifact ${fileName} not found for mode ${state.mode}`);
            }
          } catch (err) {
            showToast(`⚠️ Error loading artifact: ${err.message}`);
          }
        });
      });

      DOM.matrixTableBody.appendChild(tr);
    });

  } catch (err) {
    console.warn('Progress refresh error:', err);
  }
}

async function refreshLogs() {
  try {
    const res = await fetch(`/api/logs?mode=${state.mode}`);
    const data = await res.json();
    DOM.logsTableBody.innerHTML = '';

    if (!data.rows || data.rows.length === 0) {
      DOM.logsTableBody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted);">No records logged yet.</td></tr>';
      return;
    }

    // Show newest first
    const reversed = [...data.rows].reverse();
    reversed.slice(0, 25).forEach(row => {
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.title = 'Click to pop out past thinking & output dialog';
      tr.innerHTML = `
        <td style="font-family:var(--font-mono);font-size:0.75rem;">${row.timestamp || ''}</td>
        <td><b>${row.question_id || ''}</b></td>
        <td><code>${row.model_alias || row.model || ''}</code></td>
        <td>${row.total_duration_s ? row.total_duration_s + 's' : ''}</td>
        <td>${row.output_tokens || 0}</td>
        <td>${row.tokens_per_sec || 0}</td>
        <td>${Math.round(row.peak_vram_mb || 0)} MB</td>
        <td><span class="badge ${row.completion_status === 'Completed' ? 'badge-green' : 'badge-amber'}">${row.completion_status}</span></td>
      `;

      // Click to pop out modal with past thinking, prompt, and output
      tr.addEventListener('click', async () => {
        const fileName = row.evidence_file ? row.evidence_file.split(/[\\/]/).pop() : `${row.question_id}_${row.model_alias || 'run'}.json`;
        try {
          const artRes = await fetch(`/api/artifact?mode=${state.mode}&file=${encodeURIComponent(fileName)}`);
          if (artRes.ok) {
            const artData = await artRes.json();
            const thought = parseThought(artData.response);
            const answer = parseAnswer(artData.response);

            openModal({
              question_id: row.question_id,
              model: row.model,
              model_alias: row.model_alias,
              prompt: artData.prompt,
              thought: thought,
              final_ans: answer,
              raw_response: artData.response,
              timestamp: row.timestamp || artData.timestamp,
              filepath: row.evidence_file || fileName,
              metrics: {
                total_duration_s: row.total_duration_s,
                generation_duration_s: row.generation_duration_s,
                output_tokens: row.output_tokens,
                tokens_per_sec: row.tokens_per_sec,
                peak_vram_mb: row.peak_vram_mb !== undefined ? row.peak_vram_mb : (row.peak_vram_gb ? row.peak_vram_gb * 1024 : undefined),
                completion_status: row.completion_status,
              }
            });
          } else {
            showToast(`⚠️ Could not load artifact file ${fileName}`);
          }
        } catch (e) {
          console.error('Artifact load error:', e);
          showToast(`⚠️ Error loading artifact: ${e.message}`);
        }
      });

      DOM.logsTableBody.appendChild(tr);
    });

  } catch (err) {
    console.warn('Logs refresh error:', err);
  }
}

function parseThought(rawText) {
  if (!rawText) return null;
  const match = rawText.match(/<think>([\s\S]*?)(?:<\/think>|$)/i);
  return match && match[1] ? match[1].trim() : null;
}

function parseAnswer(rawText) {
  if (!rawText) return '';
  return rawText.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, '').trim();
}

// --- 5. Model Selection & VRAM Management ---
function isModelInstalled(modelName) {
  return state.installedModels.some(tag => tag === modelName || tag.startsWith(`${modelName}:`) || modelName.startsWith(`${tag.split(':')[0]}:`));
}

function isModelInVram(modelName) {
  return state.loadedInVram.some(tag => tag === modelName || tag.startsWith(`${modelName}:`) || modelName.startsWith(`${tag.split(':')[0]}:`));
}

function populateModelSelector() {
  DOM.modelSelect.innerHTML = '';
  const modelKeys = Object.keys(state.models);

  const savedModel = localStorage.getItem('bench_selected_model');
  if (savedModel && state.models[savedModel]) {
    state.selectedModel = savedModel;
  } else if (!state.selectedModel && modelKeys.length > 0) {
    state.selectedModel = modelKeys[0];
  }

  modelKeys.forEach((mKey) => {
    const info = state.models[mKey];
    const opt = document.createElement('option');
    opt.value = mKey;
    opt.innerText = `${mKey} (${info.alias})`;
    DOM.modelSelect.appendChild(opt);
  });

  DOM.modelSelect.value = state.selectedModel;
  updateModelControls();
}

function updateModelControls() {
  const modelKey = DOM.modelSelect.value || state.selectedModel;
  if (!modelKey || !state.models[modelKey]) return;

  const info = state.models[modelKey];
  const installed = isModelInstalled(modelKey);
  const inVram = isModelInVram(modelKey);

  // Update Status Tag
  if (inVram) {
    DOM.modelStatusTag.innerHTML = '<span class="badge badge-blue">🟢 Loaded in VRAM</span>';
  } else if (installed) {
    DOM.modelStatusTag.innerHTML = '<span class="badge badge-green">✅ Installed (Idle on Disk)</span>';
  } else {
    DOM.modelStatusTag.innerHTML = `<span class="badge badge-amber">⚠️ Not Installed</span>`;
  }

  // Description
  DOM.modelDescriptionText.innerHTML = `
    <b>Model:</b> <code>${modelKey}</code> | <b>Alias:</b> <code>${info.alias}</code> | <b>Size:</b> <code>${info.size}</code><br>
    <i>${info.purpose}</i>
  `;

  // Button States
  DOM.loadVramBtn.disabled = !installed || inVram;
  DOM.loadVramBtn.innerText = inVram ? '⚡ Loaded in VRAM' : '⚡ Load into VRAM';

  DOM.evictVramBtn.disabled = !inVram;

  if (!installed) {
    DOM.pullModelBtn.style.display = 'inline-flex';
    DOM.pullModelBtn.innerText = `📥 Pull ${info.alias}`;
  } else {
    DOM.pullModelBtn.style.display = 'none';
  }
}

// Action Button Handlers
DOM.loadVramBtn.addEventListener('click', async () => {
  const model = DOM.modelSelect.value;
  DOM.loadVramBtn.disabled = true;
  DOM.loadVramBtn.innerText = 'Loading...';
  showToast(`Loading ${model} into GPU VRAM...`);

  try {
    const res = await fetch('/api/vram/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`✅ ${model} active in VRAM!`);
    } else {
      showToast(`⚠️ ${data.error || 'Failed to load'}`);
    }
    await refreshStatus();
  } catch (e) {
    showToast('Error loading into VRAM: ' + e.message);
  }
});

DOM.evictVramBtn.addEventListener('click', async () => {
  const model = DOM.modelSelect.value;
  DOM.evictVramBtn.disabled = true;
  showToast(`Evicting ${model} from VRAM...`);

  try {
    const res = await fetch('/api/vram/evict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`🧹 Memory released for ${model}!`);
    }
    await refreshStatus();
  } catch (e) {
    showToast('Error evicting from VRAM: ' + e.message);
  }
});

DOM.pullModelBtn.addEventListener('click', async () => {
  const model = DOM.modelSelect.value;
  DOM.pullModelBtn.disabled = true;
  DOM.pullModelBtn.innerText = 'Downloading...';
  showToast(`Pulling ${model} from Ollama Library (this may take several minutes)...`);

  try {
    const res = await fetch('/api/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`✅ Downloaded ${model}!`);
    } else {
      showToast(`⚠️ Download failed: ${data.error}`);
    }
    await refreshStatus();
  } catch (e) {
    showToast('Pull failed: ' + e.message);
  }
});

DOM.modelSelect.addEventListener('change', async () => {
  state.selectedModel = DOM.modelSelect.value;
  try {
    localStorage.setItem('bench_selected_model', state.selectedModel);
  } catch (e) {}
  updateModelControls();

  if (DOM.autoLoadVramCheckbox.checked && isModelInstalled(state.selectedModel) && !isModelInVram(state.selectedModel)) {
    DOM.loadVramBtn.click();
  }
});

// --- 6. Question Selector & Category Filter ---
function populateQuestionSelector() {
  DOM.questionSelect.innerHTML = '';
  const qids = Object.keys(state.questions);

  const filtered = qids.filter(qid => {
    return state.activeCategory === 'All' || state.questions[qid].category === state.activeCategory;
  });

  filtered.forEach(qid => {
    const opt = document.createElement('option');
    opt.value = qid;
    opt.innerText = `${qid} (${state.questions[qid].difficulty})`;
    DOM.questionSelect.appendChild(opt);
  });

  if (filtered.length > 0) {
    state.selectedQid = filtered[0];
    DOM.questionSelect.value = state.selectedQid;
  }
}

function renderActiveQuestion() {
  const qData = state.questions[state.selectedQid];
  if (!qData) return;

  DOM.frozenPromptText.value = qData.prompt;
  DOM.questionMetaBadges.innerHTML = `
    <span class="badge badge-blue">${qData.category}</span>
    <span class="badge badge-amber">${qData.difficulty}</span>
  `;
}

DOM.categoryFilter.addEventListener('change', () => {
  state.activeCategory = DOM.categoryFilter.value;
  populateQuestionSelector();
  renderActiveQuestion();
});

DOM.questionSelect.addEventListener('change', () => {
  state.selectedQid = DOM.questionSelect.value;
  renderActiveQuestion();
});

// Question Source Tabs (Frozen vs Custom)
DOM.tabFrozenQuestions.addEventListener('click', () => {
  state.questionSourceTab = 'frozen';
  DOM.tabFrozenQuestions.classList.add('active');
  DOM.tabCustomPrompt.classList.remove('active');
  DOM.frozenQuestionContainer.style.display = 'block';
  DOM.customQuestionContainer.style.display = 'none';
});

DOM.tabCustomPrompt.addEventListener('click', () => {
  state.questionSourceTab = 'custom';
  DOM.tabCustomPrompt.classList.add('active');
  DOM.tabFrozenQuestions.classList.remove('active');
  DOM.frozenQuestionContainer.style.display = 'none';
  DOM.customQuestionContainer.style.display = 'block';
});

// --- 7. Mode Switching (Test vs Official) ---
DOM.modeBtnTest.addEventListener('click', () => setMode('test'));
DOM.modeBtnOfficial.addEventListener('click', () => setMode('official'));

function setMode(newMode) {
  state.mode = newMode;

  if (newMode === 'official') {
    DOM.modeBtnOfficial.classList.add('active');
    DOM.modeBtnTest.classList.remove('active');

    DOM.modeBanner.className = 'mode-banner official';
    DOM.modeBannerText.innerHTML = '<b>Official Mode Active:</b> Evidence saves to <code>data/responses/</code>. Non-regeneration rule strictly enforced.';

    // Lock Hyperparameters
    DOM.paramModeBadge.className = 'badge badge-green';
    DOM.paramModeBadge.innerText = 'Locked (Official Mode)';
    DOM.officialLockedNotice.style.display = 'block';

    DOM.tempSlider.value = '0.0';
    DOM.tempSlider.disabled = true;
    DOM.tempValueLabel.innerText = '0.00';
    DOM.numPredictInput.value = '1024';
    DOM.numPredictInput.disabled = true;
    DOM.numCtxInput.value = '4096';
    DOM.numCtxInput.disabled = true;

  } else {
    DOM.modeBtnTest.classList.add('active');
    DOM.modeBtnOfficial.classList.remove('active');

    DOM.modeBanner.className = 'mode-banner test';
    DOM.modeBannerText.innerHTML = '<b>Test Mode Active:</b> Evidence saves to <code>test_runs/responses/</code>. Official dataset remains pristine.';

    // Unlock Hyperparameters
    DOM.paramModeBadge.className = 'badge badge-amber';
    DOM.paramModeBadge.innerText = 'Adjustable (Test Mode)';
    DOM.officialLockedNotice.style.display = 'none';

    DOM.tempSlider.disabled = false;
    DOM.numPredictInput.disabled = false;
    DOM.numCtxInput.disabled = false;
  }

  refreshProgress();
  refreshLogs();
}

// Temperature Slider Value Sync
DOM.tempSlider.addEventListener('input', () => {
  DOM.tempValueLabel.innerText = parseFloat(DOM.tempSlider.value).toFixed(2);
});

// Think Drawer Accordion Toggle
DOM.thinkHeader.addEventListener('click', () => {
  const isCollapsed = DOM.thinkContent.classList.toggle('collapsed');
  DOM.thinkToggleIcon.innerText = isCollapsed ? '▶' : '▼';
});

// Refresh Logs Button
DOM.refreshLogsBtn.addEventListener('click', () => {
  refreshLogs();
  showToast('Comparison logs refreshed');
});

// --- 8. Real-Time Token Streaming Inference Engine ---
DOM.runBenchmarkBtn.addEventListener('click', async () => {
  if (state.isGenerating) return;

  if (!state.online) {
    showToast('🔴 Cannot run test: Ollama is offline. Run `ollama serve`.');
    return;
  }

  // Determine active prompt and question ID
  let activePrompt = '';
  let activeQid = '';

  if (state.questionSourceTab === 'frozen') {
    activePrompt = DOM.frozenPromptText.value.trim();
    activeQid = state.selectedQid;
  } else {
    activePrompt = DOM.customPromptText.value.trim();
    activeQid = DOM.customQidInput.value.trim() || 'Q_TEST';
  }

  if (!activePrompt) {
    showToast('⚠️ Prompt cannot be empty.');
    return;
  }

  const model = DOM.modelSelect.value;
  const temp = parseFloat(DOM.tempSlider.value);
  const numCtx = parseInt(DOM.numCtxInput.value);
  const numPredict = parseInt(DOM.numPredictInput.value);

  // Set UI to running state
  state.isGenerating = true;
  DOM.runBenchmarkBtn.disabled = true;
  DOM.runBenchmarkBtn.innerText = '⏳ Benchmarking in Progress...';

  DOM.outputSection.style.display = 'block';
  DOM.outputMetaBadge.innerText = `${state.models[model]?.alias || model} • ${activeQid}`;

  // Reset output cards
  DOM.metricTotalTime.innerText = '...';
  DOM.metricTokens.innerText = '...';
  DOM.metricSpeed.innerText = '...';
  DOM.metricVram.innerText = '...';
  DOM.metricStatus.innerText = 'Generating...';
  DOM.metricStatus.style.color = 'var(--accent-cyan)';

  DOM.thinkDrawer.style.display = 'none';
  DOM.thinkContent.innerText = '';
  DOM.thinkContent.classList.remove('collapsed');
  DOM.thinkToggleIcon.innerText = '▼';

  DOM.finalAnswerContainer.innerHTML = '<span class="streaming-cursor">▌</span>';
  DOM.savedArtifactBanner.style.display = 'none';

  // Smooth scroll to output
  DOM.outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  let accumulatedRaw = '';

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: model,
        prompt: activePrompt,
        question_id: activeQid,
        mode: state.mode,
        temperature: temp,
        num_ctx: numCtx,
        num_predict: numPredict,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep partial line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.error) {
              showToast(`Inference Error: ${data.error}`);
              DOM.metricStatus.innerText = 'Error';
              DOM.metricStatus.style.color = 'var(--accent-rose)';
              break;
            }

            if (data.token) {
              accumulatedRaw += data.token;
              renderStreamingProgress(accumulatedRaw);
            }

            if (data.done) {
              // Completed!
              finishInference(data);
              break;
            }

          } catch (pe) {
            console.error('SSE parse error:', pe);
          }
        }
      }
    }

  } catch (err) {
    showToast('Streaming failed: ' + err.message);
    DOM.metricStatus.innerText = 'Failed';
    DOM.metricStatus.style.color = 'var(--accent-rose)';
  } finally {
    state.isGenerating = false;
    DOM.runBenchmarkBtn.disabled = false;
    DOM.runBenchmarkBtn.innerText = '▶️ Run Benchmark Test & Stream Telemetry';
    await refreshProgress();
    await refreshLogs();
    await refreshStatus();
  }
});

// Dynamic live stream parser for <think> and final answer
function renderStreamingProgress(rawText) {
  const thinkMatch = rawText.match(/<think>(.*?)(?:<\/think>|$)/s);

  if (thinkMatch) {
    DOM.thinkDrawer.style.display = 'block';
    DOM.thinkContent.innerText = thinkMatch[1].trim();

    const answerPart = rawText.replace(/<think>.*?<\/think>/s, '').replace(/<think>.*$/s, '').trim();
    DOM.finalAnswerContainer.innerHTML = escapeHtml(answerPart) + ' <span class="streaming-cursor">▌</span>';
  } else {
    DOM.finalAnswerContainer.innerHTML = escapeHtml(rawText) + ' <span class="streaming-cursor">▌</span>';
  }
}

function finishInference(data) {
  // 1. Telemetry Metrics
  const metrics = data.metrics || {};
  DOM.metricTotalTime.innerText = `${metrics.total_duration_s || 0} s`;
  DOM.metricTokens.innerText = `${metrics.output_tokens || 0}`;
  DOM.metricSpeed.innerText = `${metrics.tokens_per_sec || 0} tok/s`;
  DOM.metricVram.innerText = `${Math.round(metrics.peak_vram_mb || 0)} MB`;
  DOM.metricStatus.innerText = metrics.completion_status || 'Completed';
  DOM.metricStatus.style.color = 'var(--accent-emerald)';

  // 2. Parsed Output with KaTeX & Markdown Rendering
  if (data.thought) {
    DOM.thinkDrawer.style.display = 'block';
    renderRichContent(DOM.thinkContent, data.thought);
  }
  renderRichContent(DOM.finalAnswerContainer, data.final_ans || data.raw_response);

  // 3. Saved Artifact
  if (data.saved_filepath) {
    DOM.savedArtifactPath.innerText = data.saved_filepath;
    DOM.savedArtifactBanner.style.display = 'block';
    showToast('💾 Artifact saved to: ' + data.saved_filepath);
  }
}

/**
 * Renders rich Markdown and KaTeX math formulas strictly for UI display
 * without mutating the saved raw data or interfering with the LLM.
 */
function renderRichContent(container, rawText) {
  if (!rawText) {
    container.innerHTML = '';
    return;
  }

  let text = String(rawText);

  // 1. Protect currency amounts like $80 or \$80 from triggering math delimiters
  text = text.replace(/\\?\$(\d+(?:\.\d+)?)/g, (match, amount) => `&#36;${amount}`);

  // 2. Wrap naked \boxed{...} expressions that lack math delimiters
  text = text.replace(/(?<![\$\\\(\[])\\boxed\{((?:[^{}]+|\{[^{}]*\})+)\}/g, (match, inner) => `$\\boxed{${inner}}$`);

  // 3. Normalize LaTeX display delimiters \[ ... \] to $$ ... $$ and inline \( ... \) to $ ... $ safely using callbacks
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, formula) => `\n\n$$${formula}$$\n\n`);
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, formula) => `$${formula}$`);

  // 4. Pre-render LaTeX math blocks with KaTeX into HTML tokens BEFORE Marked parses Markdown
  const mathTokens = [];
  const TOKEN_PREFIX = 'KATEXMATHPLACEHOLDER';
  const TOKEN_SUFFIX = 'END';

  // 4a. Display Math: $$ ... $$
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
    const idx = mathTokens.length;
    let html = '';
    const cleanFormula = formula.trim();
    if (window.katex && typeof window.katex.renderToString === 'function') {
      try {
        html = window.katex.renderToString(cleanFormula, { displayMode: true, throwOnError: false });
      } catch (e) {
        html = `<div class="katex-display">${escapeHtml(match)}</div>`;
      }
    } else {
      html = `<div class="katex-display">${escapeHtml(match)}</div>`;
    }
    mathTokens.push(html);
    return `\n\n${TOKEN_PREFIX}${idx}${TOKEN_SUFFIX}\n\n`;
  });

  // 4b. Inline Math: $ ... $
  text = text.replace(/\$([^\$\n]+?)\$/g, (match, formula) => {
    const idx = mathTokens.length;
    let html = '';
    const cleanFormula = formula.trim();
    if (window.katex && typeof window.katex.renderToString === 'function') {
      try {
        html = window.katex.renderToString(cleanFormula, { displayMode: false, throwOnError: false });
      } catch (e) {
        html = `<span class="katex-inline">${escapeHtml(match)}</span>`;
      }
    } else {
      html = `<span class="katex-inline">${escapeHtml(match)}</span>`;
    }
    mathTokens.push(html);
    return `${TOKEN_PREFIX}${idx}${TOKEN_SUFFIX}`;
  });

  // 5. Parse Markdown safely on the clean non-math structure
  let parsedHtml = '';
  if (window.marked && typeof window.marked.parse === 'function') {
    parsedHtml = window.marked.parse(text);
  } else {
    parsedHtml = escapeHtml(text);
  }

  // 6. Restore pre-rendered KaTeX blocks
  // Unpack any <p> tags wrapping standalone display math blocks
  parsedHtml = parsedHtml.replace(/<p>\s*(KATEXMATHPLACEHOLDER\d+END)\s*<\/p>/g, '$1');

  // Replace all placeholder tokens with their rendered KaTeX HTML
  parsedHtml = parsedHtml.replace(/KATEXMATHPLACEHOLDER(\d+)END/g, (match, idx) => {
    return mathTokens[parseInt(idx, 10)] || '';
  });

  container.innerHTML = parsedHtml;

  // 7. Fallback DOM pass if any delimiters remained or if KaTeX finished loading
  if (window.renderMathInElement) {
    try {
      window.renderMathInElement(container, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false,
      });
    } catch (katexErr) {
      console.warn('KaTeX auto-render fallback warning:', katexErr);
    }
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}

// Start application on DOM load
document.addEventListener('DOMContentLoaded', initApp);

