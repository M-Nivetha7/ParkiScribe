/**
 * AirAssist PD - Interactive Air-Writing Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const airCanvas = document.getElementById('airCanvas');
  const simCanvas = document.getElementById('simCanvas');
  const airCtx = airCanvas.getContext('2d');
  const simCtx = simCanvas.getContext('2d');

  const constructedTextEl = document.getElementById('constructedText');
  const wordSuggestionsEl = document.getElementById('wordSuggestions');
  const btnSpeak = document.getElementById('btnSpeak');
  const btnSpace = document.getElementById('btnSpace');
  const btnBackspace = document.getElementById('btnBackspace');
  const btnClearText = document.getElementById('btnClearText');
  const btnClearCanvas = document.getElementById('btnClearCanvas');

  const topPredictedChar = document.getElementById('topPredictedChar');
  const topConfidenceScore = document.getElementById('topConfidenceScore');
  const candidatesList = document.getElementById('candidatesList');
  const attentionTimeline = document.getElementById('attentionTimeline');

  const valTremorPower = document.getElementById('valTremorPower');
  const barTremorPower = document.getElementById('barTremorPower');
  const valJerk = document.getElementById('valJerk');
  const barJerk = document.getElementById('barJerk');
  const valVelocity = document.getElementById('valVelocity');
  const barVelocity = document.getElementById('barVelocity');
  const valDuration = document.getElementById('valDuration');
  const barDuration = document.getElementById('barDuration');
  const valAmplitude = document.getElementById('valAmplitude');
  const barAmplitude = document.getElementById('barAmplitude');
  const valMicrographia = document.getElementById('valMicrographia');
  const barMicrographia = document.getElementById('barMicrographia');

  const inputTremorMode = document.getElementById('inputTremorMode');
  const dwellSlider = document.getElementById('dwellSlider');
  const dwellVal = document.getElementById('dwellVal');
  const canvasStrokeStatus = document.getElementById('canvasStrokeStatus');

  const simCharSelect = document.getElementById('simCharSelect');
  const simSeveritySelect = document.getElementById('simSeveritySelect');
  const btnRunSimulation = document.getElementById('btnRunSimulation');
  const btnRecognizeCanvas = document.getElementById('btnRecognizeCanvas');

  // --- State Variables ---
  let isDrawing = false;
  let allStrokes = []; // Array of strokes: [ [[x,y], ...], [[x,y], ...] ]
  let currentStroke = [];
  let dwellTimer = null;
  let dwellTimeoutMs = 1200;
  let needsClearOnNextDraw = false;
  let constructedText = "HELP ";
  let characterHistoryPosteriors = [];

  // Initialize constructed text view
  updateConstructedTextView();
  fetchWordSuggestions();

  // --- Tab Navigation ---
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      document.getElementById(targetTab).classList.add('active');
    });
  });

  // --- Dwell Slider Listener ---
  dwellSlider.addEventListener('input', (e) => {
    dwellTimeoutMs = parseInt(e.target.value);
    dwellVal.textContent = `${dwellTimeoutMs}ms`;
  });

  // --- Canvas Air-Drawing Handlers ---
  function getCanvasCoords(e) {
    const rect = airCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (airCanvas.width / rect.width),
      y: (clientY - rect.top) * (airCanvas.height / rect.height)
    };
  }

  function startDrawing(e) {
    if (needsClearOnNextDraw) {
      allStrokes = [];
      currentStroke = [];
      airCtx.clearRect(0, 0, airCanvas.width, airCanvas.height);
      needsClearOnNextDraw = false;
    }

    isDrawing = true;
    currentStroke = [];
    clearTimeout(dwellTimer);
    canvasStrokeStatus.textContent = "Writing...";
    canvasStrokeStatus.className = "stroke-status writing";

    const coords = getCanvasCoords(e);
    addPointToStroke(coords.x, coords.y);
    redrawAirCanvas();
  }

  function drawMove(e) {
    if (!isDrawing) return;
    clearTimeout(dwellTimer);
    const coords = getCanvasCoords(e);
    addPointToStroke(coords.x, coords.y);
    redrawAirCanvas();
  }

  function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    clearTimeout(dwellTimer);

    if (currentStroke.length >= 2) {
      allStrokes.push([...currentStroke]);
    }
    currentStroke = [];
    redrawAirCanvas();

    canvasStrokeStatus.textContent = "Stroke drawn (finish letter or wait)";
    canvasStrokeStatus.className = "stroke-status idle";

    // Start auto-recognition timer after user pauses
    dwellTimer = setTimeout(() => {
      if (allStrokes.length > 0) {
        finishWriting();
      }
    }, dwellTimeoutMs);
  }

  function addPointToStroke(x, y) {
    const tremorMode = inputTremorMode.value;
    let jitterX = 0;
    let jitterY = 0;

    if (tremorMode === 'mild') {
      const f = 5.0;
      const t = currentStroke.length / 30.0;
      jitterX = Math.sin(2 * Math.PI * f * t) * 3.0 + (Math.random() - 0.5) * 2;
      jitterY = Math.cos(2 * Math.PI * f * t) * 3.0 + (Math.random() - 0.5) * 2;
    } else if (tremorMode === 'moderate') {
      const f = 5.5;
      const t = currentStroke.length / 30.0;
      jitterX = Math.sin(2 * Math.PI * f * t) * 6.5 + (Math.random() - 0.5) * 4;
      jitterY = Math.cos(2 * Math.PI * f * t) * 6.5 + (Math.random() - 0.5) * 4;
    } else if (tremorMode === 'severe') {
      const f = 6.0;
      const t = currentStroke.length / 30.0;
      jitterX = (Math.sin(2 * Math.PI * f * t) + 0.3 * Math.sin(4 * Math.PI * f * t)) * 11.0 + (Math.random() - 0.5) * 5;
      jitterY = (Math.cos(2 * Math.PI * f * t) + 0.3 * Math.cos(4 * Math.PI * f * t)) * 11.0 + (Math.random() - 0.5) * 5;
    }

    currentStroke.push([x + jitterX, y + jitterY, 0.0]);
  }

  function redrawAirCanvas() {
    airCtx.clearRect(0, 0, airCanvas.width, airCanvas.height);

    airCtx.lineWidth = 6;
    airCtx.lineCap = 'round';
    airCtx.lineJoin = 'round';
    airCtx.strokeStyle = '#38bdf8';
    airCtx.shadowColor = 'rgba(56, 189, 248, 0.6)';
    airCtx.shadowBlur = 8;

    // Draw all completed strokes
    allStrokes.forEach(stroke => {
      if (stroke.length >= 2) {
        airCtx.beginPath();
        airCtx.moveTo(stroke[0][0], stroke[0][1]);
        for (let i = 1; i < stroke.length; i++) {
          airCtx.lineTo(stroke[i][0], stroke[i][1]);
        }
        airCtx.stroke();
      }
    });

    // Draw active stroke
    if (currentStroke.length >= 2) {
      airCtx.beginPath();
      airCtx.moveTo(currentStroke[0][0], currentStroke[0][1]);
      for (let i = 1; i < currentStroke.length; i++) {
        airCtx.lineTo(currentStroke[i][0], currentStroke[i][1]);
      }
      airCtx.stroke();
    }

    airCtx.shadowBlur = 0;
  }

  function resetStroke() {
    isDrawing = false;
    allStrokes = [];
    currentStroke = [];
    airCtx.clearRect(0, 0, airCanvas.width, airCanvas.height);
    canvasStrokeStatus.textContent = "Ready";
    canvasStrokeStatus.className = "stroke-status idle";
  }

  async function finishWriting() {
    clearTimeout(dwellTimer);
    if (isDrawing && currentStroke.length >= 2) {
      allStrokes.push([...currentStroke]);
      currentStroke = [];
    }

    if (allStrokes.length === 0) {
      return;
    }

    canvasStrokeStatus.textContent = "Recognizing...";
    canvasStrokeStatus.className = "stroke-status writing";

    // Export high-resolution rendered canvas image
    const b64Canvas = airCanvas.toDataURL('image/png');

    // Combine all strokes into trajectory list for kinematics
    const allPoints = [];
    allStrokes.forEach(s => {
      s.forEach(pt => allPoints.push(pt));
    });

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: b64Canvas,
          trajectory: allPoints,
          top_k: 5
        })
      });
      const data = await resp.json();

      if (data.predicted_char || data.predicted_word) {
        displayPrediction(data);
        appendRecognizedText(data);
        if (data.candidates && !data.is_word) {
          characterHistoryPosteriors.push(data.candidates.map(c => [c.char, c.confidence]));
        } else if (data.is_word) {
          characterHistoryPosteriors = [];
        }
        fetchWordSuggestions();
        if (data.is_word) {
          canvasStrokeStatus.textContent = `Recognized Word '${data.predicted_word}'!`;
        } else {
          canvasStrokeStatus.textContent = `Recognized '${data.predicted_char}'!`;
        }
        canvasStrokeStatus.className = "stroke-status idle";
        needsClearOnNextDraw = true; // Auto-clear for next letter/word
      }
    } catch (err) {
      console.error("Prediction error:", err);
      canvasStrokeStatus.textContent = "Error recognizing";
    }
  }

  airCanvas.addEventListener('mousedown', startDrawing);
  airCanvas.addEventListener('mousemove', drawMove);
  window.addEventListener('mouseup', stopDrawing);

  airCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e); }, { passive: false });
  airCanvas.addEventListener('touchmove', (e) => { e.preventDefault(); drawMove(e); }, { passive: false });
  window.addEventListener('touchend', stopDrawing);

  btnClearCanvas.addEventListener('click', resetStroke);
  if (btnRecognizeCanvas) {
    btnRecognizeCanvas.addEventListener('click', finishWriting);
  }

  // --- Display Prediction & Biomarkers ---
  function displayPrediction(data) {
    const isWord = Boolean(data.is_word);
    const displayText = isWord ? (data.predicted_word || data.predicted_char) : (data.predicted_char || "?");
    topPredictedChar.textContent = displayText;

    if (displayText.length > 3) {
      topPredictedChar.style.fontSize = "2.8rem";
      topPredictedChar.style.letterSpacing = "2px";
    } else {
      topPredictedChar.style.fontSize = "4.5rem";
      topPredictedChar.style.letterSpacing = "normal";
    }

    const topConf = Math.round(data.confidence * 1000) / 10;
    topConfidenceScore.textContent = `${topConf}%`;

    // Render Candidates List & Segmented Letters Breakdown
    candidatesList.innerHTML = '';

    // If multi-character word, show letters breakdown badges
    if (isWord && data.characters && data.characters.length > 0) {
      const segRow = document.createElement('div');
      segRow.className = 'segmented-chars-banner';
      segRow.style.padding = '8px 10px';
      segRow.style.marginBottom = '10px';
      segRow.style.background = 'rgba(56, 189, 248, 0.1)';
      segRow.style.border = '1px solid rgba(56, 189, 248, 0.25)';
      segRow.style.borderRadius = '8px';
      segRow.style.display = 'flex';
      segRow.style.gap = '6px';
      segRow.style.alignItems = 'center';
      segRow.style.flexWrap = 'wrap';

      const titleSpan = document.createElement('span');
      titleSpan.textContent = 'Letters: ';
      titleSpan.style.fontSize = '0.8rem';
      titleSpan.style.color = '#94a3b8';
      titleSpan.style.fontWeight = '600';
      segRow.appendChild(titleSpan);

      data.characters.forEach((ch) => {
        const badge = document.createElement('span');
        badge.style.background = '#0f172a';
        badge.style.border = '1px solid #38bdf8';
        badge.style.color = '#38bdf8';
        badge.style.padding = '2px 8px';
        badge.style.borderRadius = '6px';
        badge.style.fontWeight = '700';
        badge.style.fontSize = '0.85rem';
        badge.textContent = `${ch.char} (${Math.round(ch.confidence)}%)`;
        segRow.appendChild(badge);
      });
      candidatesList.appendChild(segRow);
    }

    if (data.candidates) {
      data.candidates.forEach(cand => {
        const pct = Math.round(cand.confidence * 1000) / 10;
        const row = document.createElement('div');
        row.className = 'candidate-row';
        row.innerHTML = `
          <span class="cand-char" style="${isWord ? 'min-width: 60px; font-weight:700;' : ''}">${cand.char}</span>
          <div class="cand-bar-wrap">
            <div class="cand-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <span class="cand-pct">${pct}%</span>
        `;
        candidatesList.appendChild(row);
      });
    }

    // Render Bio-Markers (Module 11)
    if (data.biomarkers) {
      const bm = data.biomarkers;
      const tremorPct = (bm.tremor_band_power_ratio * 100).toFixed(1);
      valTremorPower.innerHTML = `${tremorPct}% <small>${bm.trajectory_variation}</small>`;
      barTremorPower.style.width = `${Math.min(100, Math.round(bm.tremor_intensity_score))}%`;

      valJerk.innerHTML = `${bm.stroke_smoothness_score.toFixed(1)}% <small>NJ: ${Math.round(bm.normalized_jerk)}</small>`;
      barJerk.style.width = `${Math.round(bm.stroke_smoothness_score)}%`;
      barJerk.className = bm.stroke_smoothness_score > 60 ? 'fill ok' : 'fill warn';

      valVelocity.innerHTML = `${Math.round(bm.average_speed_px_sec)} <small>px/s</small>`;
      barVelocity.style.width = `${Math.min(100, Math.round(bm.average_speed_px_sec / 15))}%`;

      if (valDuration) {
        valDuration.innerHTML = `${bm.writing_duration_sec.toFixed(2)} <small>sec</small>`;
        barDuration.style.width = `${Math.min(100, Math.round(bm.writing_duration_sec * 25))}%`;
      }

      if (valAmplitude) {
        valAmplitude.innerHTML = `${Math.round(bm.movement_amplitude_width_px)}x${Math.round(bm.movement_amplitude_height_px)} <small>px</small>`;
        barAmplitude.style.width = `${Math.min(100, Math.round(bm.movement_amplitude_width_px / 4))}%`;
      }

      if (valMicrographia) {
        valMicrographia.innerHTML = bm.micrographia_detected ? `Shrinkage <small>Slope ${bm.micrographia_slope.toFixed(1)}</small>` : `Normal <small>Stable</small>`;
        barMicrographia.style.width = bm.micrographia_detected ? '100%' : '20%';
        barMicrographia.className = bm.micrographia_detected ? 'fill warn' : 'fill ok';
      }
    } else if (data.kinematics) {
      const k = data.kinematics;
      const tremorRatio = Math.min(1.0, (k.tremor_power_ratio_4_7hz || 0.0));
      valTremorPower.innerHTML = `${tremorRatio.toFixed(2)} <small>ratio</small>`;
      barTremorPower.style.width = `${Math.round(tremorRatio * 100)}%`;

      const jerkNorm = (k.jerk_normalized || 0.0);
      valJerk.innerHTML = `${jerkNorm.toFixed(1)} <small>norm</small>`;
      barJerk.style.width = `${Math.min(100, Math.round(jerkNorm * 4))}%`;

      const meanVel = (k.vel_mean_speed || 0.0);
      valVelocity.innerHTML = `${meanVel.toFixed(2)} <small>u/s</small>`;
      barVelocity.style.width = `${Math.min(100, Math.round(meanVel * 30))}%`;

      const microArea = (k.micrographia_area_ratio || 1.0);
      valMicrographia.innerHTML = `${microArea.toFixed(2)} <small>area</small>`;
      barMicrographia.style.width = `${Math.min(100, Math.round(microArea * 100))}%`;
    }

    // Render Attention Timeline
    if (data.attention_weights && data.attention_weights.length > 0) {
      attentionTimeline.innerHTML = '';
      const weights = data.attention_weights;
      const maxW = Math.max(...weights) || 1.0;
      weights.forEach(w => {
        const normW = w / maxW;
        const block = document.createElement('div');
        block.className = 'attn-block';
        block.style.background = `rgba(56, 189, 248, ${0.1 + normW * 0.9})`;
        attentionTimeline.appendChild(block);
      });
    }
  }

  // --- Simulation Runner ---
  btnRunSimulation.addEventListener('click', async () => {
    const char = simCharSelect.value;
    const severity = simSeveritySelect.value;

    btnRunSimulation.disabled = true;
    btnRunSimulation.textContent = "Simulating...";

    try {
      const resp = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ char, severity })
      });
      const res = await resp.json();

      drawSimulationTrajectories(res.raw_trajectory, res.filtered_trajectory);
      displayPrediction(res.prediction);
      appendRecognizedText(res.prediction);
      fetchWordSuggestions();

    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      btnRunSimulation.disabled = false;
      btnRunSimulation.textContent = "Generate & Recognize";
    }
  });

  function drawSimulationTrajectories(rawTraj, filteredTraj) {
    simCtx.clearRect(0, 0, simCanvas.width, simCanvas.height);
    const w = simCanvas.width;
    const h = simCanvas.height;

    // Draw Raw Parkinsonian Trajectory in Red
    if (rawTraj && rawTraj.length > 1) {
      simCtx.lineWidth = 3;
      simCtx.strokeStyle = '#f43f5e';
      simCtx.beginPath();
      simCtx.moveTo(rawTraj[0][0] * w, rawTraj[0][1] * h);
      for (let i = 1; i < rawTraj.length; i++) {
        simCtx.lineTo(rawTraj[i][0] * w, rawTraj[i][1] * h);
      }
      simCtx.stroke();
    }

    // Draw Filtered Intention Trajectory in Cyan
    if (filteredTraj && filteredTraj.length > 1) {
      simCtx.lineWidth = 4;
      simCtx.strokeStyle = '#38bdf8';
      simCtx.shadowColor = 'rgba(56, 189, 248, 0.7)';
      simCtx.shadowBlur = 8;
      simCtx.beginPath();
      simCtx.moveTo(filteredTraj[0][0] * w, filteredTraj[0][1] * h);
      for (let i = 1; i < filteredTraj.length; i++) {
        simCtx.lineTo(filteredTraj[i][0] * w, filteredTraj[i][1] * h);
      }
      simCtx.stroke();
      simCtx.shadowBlur = 0;
    }
  }

  // --- Constructed Text & NLP Suggestions ---
  function appendRecognizedText(data) {
    if (!data) return;
    if (data.is_word) {
      const word = (data.predicted_word || data.predicted_char || "").trim();
      if (!word) return;
      if (constructedText.trim().length > 0 && !constructedText.endsWith(" ")) {
        constructedText += " " + word + " ";
      } else {
        constructedText += word + " ";
      }
    } else if (data.predicted_char) {
      constructedText += data.predicted_char;
    }
    updateConstructedTextView();
  }

  function appendCharacter(char) {
    constructedText += char;
    updateConstructedTextView();
  }

  function updateConstructedTextView() {
    constructedTextEl.textContent = constructedText || "[Empty]";
  }

  async function fetchWordSuggestions() {
    const tokens = constructedText.trim().split(/\s+/);
    const lastToken = tokens[tokens.length - 1] || "";

    try {
      const resp = await fetch('/api/decode_word', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: lastToken,
          posteriors: characterHistoryPosteriors.slice(-lastToken.length)
        })
      });
      const data = await resp.json();

      wordSuggestionsEl.innerHTML = '';
      if (data.suggestions && data.suggestions.length > 0) {
        data.suggestions.forEach(item => {
          const chip = document.createElement('button');
          chip.className = 'chip-btn';
          chip.textContent = item.word;
          chip.addEventListener('click', () => {
            // Replace last token with selected word
            tokens[tokens.length - 1] = item.word;
            constructedText = tokens.join(' ') + ' ';
            characterHistoryPosteriors = [];
            updateConstructedTextView();
            fetchWordSuggestions();
            speakText(item.word);
          });
          wordSuggestionsEl.appendChild(chip);
        });
      }
    } catch (err) {
      console.error("Word suggestion error:", err);
    }
  }

  // --- Buttons Handlers ---
  btnSpace.addEventListener('click', () => {
    constructedText += ' ';
    characterHistoryPosteriors = [];
    updateConstructedTextView();
    fetchWordSuggestions();
  });

  btnBackspace.addEventListener('click', () => {
    if (constructedText.length > 0) {
      constructedText = constructedText.slice(0, -1);
      if (characterHistoryPosteriors.length > 0) {
        characterHistoryPosteriors.pop();
      }
      updateConstructedTextView();
      fetchWordSuggestions();
    }
  });

  btnClearText.addEventListener('click', () => {
    constructedText = '';
    characterHistoryPosteriors = [];
    updateConstructedTextView();
    wordSuggestionsEl.innerHTML = '';
  });

  // --- Keyboard Shortcuts ---
  window.addEventListener('keydown', (e) => {
    // Ignore if typing inside input/select
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      return;
    }
    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      btnSpace.click();
    } else if (e.key === 'Backspace') {
      e.preventDefault();
      btnBackspace.click();
    } else if (e.key === 'c' || e.key === 'C') {
      e.preventDefault();
      if (isCamActive) {
        btnClearCam.click();
      } else {
        btnClearCanvas.click();
      }
    } else if (e.key === 'x' || e.key === 'X') {
      e.preventDefault();
      btnClearText.click();
    } else if (e.key === 'p' || e.key === 'P') {
      e.preventDefault();
      if (isCamActive) {
        recognizeWebcamDrawing();
      } else {
        finishWriting();
      }
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      btnSpeak.click();
    }
  });

  // --- Speech Synthesis ---
  function speakText(text) {
    const phrase = text || constructedText.trim();
    if (!phrase) return;

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(phrase);
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Text-to-Speech is not supported by your browser.");
    }
  }

  btnSpeak.addEventListener('click', () => speakText(constructedText));

  // Emergency Phrase Buttons
  document.querySelectorAll('.phrase-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const phrase = btn.getAttribute('data-phrase');
      constructedText = phrase + ' ';
      updateConstructedTextView();
      speakText(phrase);
    });
  });

  // --- Webcam Tracking Handler (Matches user's OpenCV starter code) ---
  const btnToggleCam = document.getElementById('btnToggleCam');
  const btnToggleMirror = document.getElementById('btnToggleMirror');
  const btnRecognizeCam = document.getElementById('btnRecognizeCam');
  const btnClearCam = document.getElementById('btnClearCam');
  const webcamVideo = document.getElementById('webcamVideo');
  const webcamCanvas = document.getElementById('webcamCanvas');
  const camStatusText = document.getElementById('camStatusText');
  const webcamCtx = webcamCanvas ? webcamCanvas.getContext('2d') : null;

  let camStream = null;
  let camInterval = null;
  let isCamActive = false;
  let isMirrored = true; // Natural selfie mirror view by default
  let camCompletedStrokes = [];
  let camCurrentStroke = [];
  let camDwellTimer = null;
  let camHasDrawn = false;
  let camNeedsClearOnNextDraw = false;
  let lastCamState = 'none';
  let smoothPx = null;
  let smoothPy = null;

  const frameGrabCanvas = document.createElement('canvas');
  frameGrabCanvas.width = 320;
  frameGrabCanvas.height = 240;
  const frameGrabCtx = frameGrabCanvas.getContext('2d');

  function syncCanvasDimensions() {
    if (!webcamVideo || !webcamCanvas) return;
    const vw = webcamVideo.videoWidth || 640;
    const vh = webcamVideo.videoHeight || 480;
    if (webcamCanvas.width !== vw || webcamCanvas.height !== vh) {
      webcamCanvas.width = vw;
      webcamCanvas.height = vh;
      redrawWebcamCanvas();
    }
  }

  if (btnToggleMirror) {
    btnToggleMirror.addEventListener('click', () => {
      isMirrored = !isMirrored;
      if (isMirrored) {
        webcamVideo.classList.remove('unmirrored');
        btnToggleMirror.textContent = '🪞 Mirror: ON';
      } else {
        webcamVideo.classList.add('unmirrored');
        btnToggleMirror.textContent = '🪞 Mirror: OFF';
      }
    });
  }

  function drawSmoothStroke(ctx, stroke) {
    if (!stroke || stroke.length < 1) return;
    if (stroke.length === 1) {
      ctx.beginPath();
      ctx.arc(stroke[0][0], stroke[0][1], ctx.lineWidth / 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    if (stroke.length === 2) {
      ctx.beginPath();
      ctx.moveTo(stroke[0][0], stroke[0][1]);
      ctx.lineTo(stroke[1][0], stroke[1][1]);
      ctx.stroke();
      return;
    }
    ctx.beginPath();
    ctx.moveTo(stroke[0][0], stroke[0][1]);
    for (let i = 1; i < stroke.length - 1; i++) {
      const midX = (stroke[i][0] + stroke[i + 1][0]) / 2;
      const midY = (stroke[i][1] + stroke[i + 1][1]) / 2;
      ctx.quadraticCurveTo(stroke[i][0], stroke[i][1], midX, midY);
    }
    ctx.lineTo(stroke[stroke.length - 1][0], stroke[stroke.length - 1][1]);
    ctx.stroke();
  }

  function redrawWebcamCanvas(cursorPoint = null, cursorState = 'none') {
    if (!webcamCtx || !webcamCanvas) return;
    webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCtx.lineWidth = 8;
    webcamCtx.lineCap = 'round';
    webcamCtx.lineJoin = 'round';
    webcamCtx.strokeStyle = '#38bdf8';
    webcamCtx.shadowColor = 'rgba(56, 189, 248, 0.85)';
    webcamCtx.shadowBlur = 10;

    camCompletedStrokes.forEach(stroke => {
      drawSmoothStroke(webcamCtx, stroke);
    });

    if (camCurrentStroke.length >= 1) {
      drawSmoothStroke(webcamCtx, camCurrentStroke);
    }

    webcamCtx.shadowBlur = 0;

    // Draw fingertip cursor overlay precisely matching finger position
    if (cursorPoint) {
      const [cx, cy] = cursorPoint;
      if (cursorState === 'draw') {
        webcamCtx.strokeStyle = '#10b981';
        webcamCtx.lineWidth = 3;
        webcamCtx.beginPath();
        webcamCtx.arc(cx, cy, 14, 0, Math.PI * 2);
        webcamCtx.stroke();

        webcamCtx.fillStyle = '#10b981';
        webcamCtx.beginPath();
        webcamCtx.arc(cx, cy, 6, 0, Math.PI * 2);
        webcamCtx.fill();

        webcamCtx.font = 'bold 12px Inter, sans-serif';
        webcamCtx.fillText("Drawing", cx + 16, cy - 8);
      } else if (cursorState === 'pen_lift') {
        webcamCtx.strokeStyle = '#38bdf8';
        webcamCtx.lineWidth = 2.5;
        webcamCtx.beginPath();
        webcamCtx.arc(cx, cy, 12, 0, Math.PI * 2);
        webcamCtx.stroke();

        webcamCtx.fillStyle = '#38bdf8';
        webcamCtx.font = 'bold 11px Inter, sans-serif';
        webcamCtx.fillText("Pen Up", cx + 14, cy - 8);
      } else if (cursorState === 'fist') {
        webcamCtx.strokeStyle = '#ef4444';
        webcamCtx.lineWidth = 3;
        webcamCtx.beginPath();
        webcamCtx.arc(cx, cy, 18, 0, Math.PI * 2);
        webcamCtx.stroke();

        webcamCtx.fillStyle = '#ef4444';
        webcamCtx.font = 'bold 11px Inter, sans-serif';
        webcamCtx.fillText("Clear Screen", cx + 18, cy - 8);
      }
    }
  }

  async function recognizeWebcamDrawing() {
    const allStrokesToRecognize = [...camCompletedStrokes];
    if (camCurrentStroke.length >= 2) {
      allStrokesToRecognize.push([...camCurrentStroke]);
    }
    if (allStrokesToRecognize.length === 0) return;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = webcamCanvas.width;
    offCanvas.height = webcamCanvas.height;
    const offCtx = offCanvas.getContext('2d');
    offCtx.fillStyle = '#000000';
    offCtx.fillRect(0, 0, offCanvas.width, offCanvas.height);

    offCtx.lineWidth = 14;
    offCtx.lineCap = 'round';
    offCtx.lineJoin = 'round';
    offCtx.strokeStyle = '#ffffff';

    allStrokesToRecognize.forEach(st => {
      drawSmoothStroke(offCtx, st);
    });

    const b64 = offCanvas.toDataURL('image/png');
    camStatusText.textContent = "AI Recognizing...";

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          canvas_image: b64,
          strokes: allStrokesToRecognize
        })
      });
      const data = await resp.json();
      if (data.predicted_char || data.predicted_word) {
        displayPrediction(data);
        appendRecognizedText(data);
        const confPct = (data.confidence * 100).toFixed(1);
        if (data.is_word) {
          camStatusText.textContent = `Word Recognized: '${data.predicted_word}' (${confPct}%)`;
        } else {
          camStatusText.textContent = `Recognized '${data.predicted_char}' (${confPct}%)`;
        }
        camNeedsClearOnNextDraw = true; // Auto-clear on next stroke
      }

    } catch (err) {
      console.error("Webcam prediction error:", err);
      camStatusText.textContent = "Recognition failed";
    }
  }

  if (btnRecognizeCam) {
    btnRecognizeCam.addEventListener('click', () => {
      recognizeWebcamDrawing();
    });
  }

  if (btnClearCam) {
    btnClearCam.addEventListener('click', () => {
      camCompletedStrokes = [];
      camCurrentStroke = [];
      camHasDrawn = false;
      camNeedsClearOnNextDraw = false;
      smoothPx = null;
      smoothPy = null;
      clearTimeout(camDwellTimer);
      redrawWebcamCanvas();
      camStatusText.textContent = "Canvas cleared";
    });
  }

  if (btnToggleCam && webcamVideo && webcamCanvas) {
    btnToggleCam.addEventListener('click', async () => {
      if (isCamActive) {
        // Stop Camera
        if (camStream) {
          camStream.getTracks().forEach(t => t.stop());
          camStream = null;
        }
        clearInterval(camInterval);
        clearTimeout(camDwellTimer);
        isCamActive = false;
        btnToggleCam.textContent = "Start Camera";
        btnToggleCam.className = "btn btn-primary";
        camStatusText.textContent = "Camera stopped";
        camCompletedStrokes = [];
        camCurrentStroke = [];
        smoothPx = null;
        smoothPy = null;
        redrawWebcamCanvas();
      } else {
        // Start Camera
        try {
          if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera access requires http://localhost or HTTPS. Open the app at http://localhost:5050.");
          }

          camStatusText.textContent = "Requesting webcam access...";
          camStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
          });
          webcamVideo.srcObject = camStream;
          await webcamVideo.play();
          webcamVideo.onloadedmetadata = () => {
            syncCanvasDimensions();
          };
          isCamActive = true;
          btnToggleCam.textContent = "Stop Camera";
          btnToggleCam.className = "btn btn-danger";
          camStatusText.textContent = "Tracking active - Point index finger to write live!";

          let isProcessing = false;

          // Processing loop: real-time touchless tracking with live on-screen rendering
          camInterval = setInterval(async () => {
            if (!isCamActive || webcamVideo.readyState < 2 || isProcessing) return;
            isProcessing = true;

            syncCanvasDimensions();
            frameGrabCtx.drawImage(webcamVideo, 0, 0, 320, 240);
            const b64Image = frameGrabCanvas.toDataURL('image/jpeg', 0.5);

            try {
              const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: b64Image })
              });
              const res = await resp.json();

              if (res.detected && res.point) {
                // When mirrored (standard selfie mode):
                // If user moves to the right, effectiveNormX increases to the right side!
                const effectiveNormX = isMirrored ? (1.0 - res.point[0]) : res.point[0];
                const targetPx = effectiveNormX * webcamCanvas.width;
                const targetPy = res.point[1] * webcamCanvas.height;

                // EMA smoothing removes jitter while remaining instantaneous
                if (smoothPx === null) {
                  smoothPx = targetPx;
                  smoothPy = targetPy;
                } else {
                  smoothPx = smoothPx * 0.25 + targetPx * 0.75;
                  smoothPy = smoothPy * 0.25 + targetPy * 0.75;
                }

                const px = Math.round(smoothPx);
                const py = Math.round(smoothPy);

                // --- Exact user starter code gesture states ---
                if (res.state === 'fist') {
                  // ✊ Fist -> Clear screen
                  camCompletedStrokes = [];
                  camCurrentStroke = [];
                  camHasDrawn = false;
                  camNeedsClearOnNextDraw = false;
                  clearTimeout(camDwellTimer);
                  camStatusText.textContent = "✊ Fist: Screen Cleared";
                } else if (res.state === 'pen_lift') {
                  // ✋ Two fingers -> Pen lift / Stop drawing (move freely)
                  if (camCurrentStroke.length >= 2) {
                    camCompletedStrokes.push([...camCurrentStroke]);
                    camCurrentStroke = [];
                  }
                  camStatusText.textContent = "✋ Pen Lifted (Move freely)";

                  // Auto-recognize after pause with pen lifted
                  if (camHasDrawn) {
                    clearTimeout(camDwellTimer);
                    camDwellTimer = setTimeout(() => {
                      if (camHasDrawn && (camCompletedStrokes.length > 0 || camCurrentStroke.length > 0)) {
                        recognizeWebcamDrawing();
                        camHasDrawn = false;
                      }
                    }, 1400);
                  }
                } else if (res.state === 'draw') {
                  // ✍️ Only index -> Draw smooth on-live line!
                  clearTimeout(camDwellTimer);
                  if (camNeedsClearOnNextDraw) {
                    camCompletedStrokes = [];
                    camCurrentStroke = [];
                    camNeedsClearOnNextDraw = false;
                  }
                  if (camCurrentStroke.length === 0) {
                    camCurrentStroke.push([px, py, Date.now()]);
                  } else {
                    const lastPt = camCurrentStroke[camCurrentStroke.length - 1];
                    const dist = Math.hypot(px - lastPt[0], py - lastPt[1]);
                    if (dist >= 2) {
                      camCurrentStroke.push([px, py, Date.now()]);
                    }
                  }
                  camHasDrawn = true;
                  camStatusText.textContent = "✍️ Writing on-screen...";
                } else {
                  if (camCurrentStroke.length >= 2) {
                    camCompletedStrokes.push([...camCurrentStroke]);
                    camCurrentStroke = [];
                  }
                }

                redrawWebcamCanvas([px, py], res.state);
                lastCamState = res.state;
              } else {
                smoothPx = null;
                smoothPy = null;
                if (camCurrentStroke.length >= 2) {
                  camCompletedStrokes.push([...camCurrentStroke]);
                  camCurrentStroke = [];
                }
                redrawWebcamCanvas(null, 'none');
              }
            } catch (err) {
              // Graceful frame drop
            } finally {
              isProcessing = false;
            }
          }, 45);
        } catch (err) {
          const errorMessages = {
            NotAllowedError: "Camera permission was denied. Allow camera access for this site in your browser settings, then try again.",
            PermissionDeniedError: "Camera permission was denied. Allow camera access for this site in your browser settings, then try again.",
            NotFoundError: "No camera was found. Connect a webcam and try again.",
            DevicesNotFoundError: "No camera was found. Connect a webcam and try again.",
            NotReadableError: "The camera is busy in another app. Close apps using it and try again.",
            TrackStartError: "The camera is busy in another app. Close apps using it and try again.",
            OverconstrainedError: "This camera does not support the requested settings. Try again with another camera.",
            SecurityError: "Camera access is blocked by browser security. Open the app at http://localhost:5050 or use HTTPS."
          };
          camStatusText.textContent = `Camera error: ${errorMessages[err.name] || err.message || "Unable to access the camera."}`;
          console.error("Camera access error:", err);
        }
      }
    });
  }


  // Auto-run initial simulation for visual showcase
  setTimeout(() => {
    btnRunSimulation.click();
  }, 300);
});
