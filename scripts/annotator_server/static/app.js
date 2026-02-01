(function () {
  'use strict';

  const API_BASE = ''; // same origin
  const viewer = document.getElementById('viewer');
  const viewport = document.getElementById('viewport');
  const img = document.getElementById('img');
  const overlay = document.getElementById('overlay');
  const loadingEl = document.getElementById('loading');
  const statusEl = document.getElementById('status');
  const btnComplete = document.getElementById('btnComplete');
  const btnClear = document.getElementById('btnClear');
  const btnRecompute = document.getElementById('btnRecompute');
  const showAnnotationsEl = document.getElementById('showAnnotations');
  const showSampledPointsEl = document.getElementById('showSampledPoints');
  const btnSettings = document.getElementById('btnSettings');
  const settingsPanel = document.getElementById('settingsPanel');
  const configNumberOfPoints = document.getElementById('configNumberOfPoints');
  const configNumSamples = document.getElementById('configNumSamples');
  const configNumEpochs = document.getElementById('configNumEpochs');
  const configClusters = document.getElementById('configClusters');
  const configBatchSize = document.getElementById('configBatchSize');
  const configBeta = document.getElementById('configBeta');
  const configClusterLossWeight = document.getElementById('configClusterLossWeight');
  const configCategoryLossWeight = document.getElementById('configCategoryLossWeight');
  const configPixelSampling = document.getElementById('configPixelSampling');
  const configPcaFitStep = document.getElementById('configPcaFitStep');
  const configClusterOnPca = document.getElementById('configClusterOnPca');
  const configClusterPcaDims = document.getElementById('configClusterPcaDims');
  const btnConfigSave = document.getElementById('btnConfigSave');
  const btnConfigApply = document.getElementById('btnConfigApply');
  const btnCopyClipboard = document.getElementById('btnCopyClipboard');
  const btnSettingsClose = document.getElementById('btnSettingsClose');
  const categoryIdEl = document.getElementById('categoryId');
  const btnAddCategory = document.getElementById('btnAddCategory');
  const btnClearCategories = document.getElementById('btnClearCategories');
  const categoryCountEl = document.getElementById('categoryCount');
  const inputFileSelect = document.getElementById('inputFileSelect');

  let scale = 1;
  let currentLoadedPath = null; // path of the .npy currently displayed (for image selector)
  let lastSentRoi = null; // last ROI sent to Recompute (so we can re-run without redrawing when adding categories)
  let tx = 0;
  let ty = 0;
  let lastMouse = { x: 0, y: 0 };
  let dragging = false;
  let drawMode = false;
  let drawingStroke = false; // true while mouse is down in annotation mode
  const MIN_POINT_DIST = 3; // min pixels (image space) between points when drawing
  let polygon = []; // [{ x, y }] in image pixel coords (current ROI or in-progress)
  let categoryAnnotations = []; // [{ polygon: [{x,y},...], category: 0|1|2|... }]
  let sampledPoints = []; // [{ x, y }] from API (col, row)
  let referencePoints = [];
  let imageWidth = 0;
  let imageHeight = 0;

  const CATEGORY_COLORS = ['rgba(100, 150, 255, 0.25)', 'rgba(100, 255, 150, 0.25)', 'rgba(255, 150, 255, 0.25)', 'rgba(255, 200, 100, 0.25)', 'rgba(100, 255, 255, 0.25)'];
  const CATEGORY_STROKES = ['rgba(100, 150, 255, 0.9)', 'rgba(100, 255, 150, 0.9)', 'rgba(255, 150, 255, 0.9)', 'rgba(255, 200, 100, 0.9)', 'rgba(100, 255, 255, 0.9)'];

  function setStatus(text, className = '') {
    statusEl.textContent = text;
    statusEl.className = 'status-bar ' + (className || '');
  }

  function showLoading(show) {
    loadingEl.style.display = show ? 'flex' : 'none';
  }

  function applyTransform() {
    viewport.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  }

  function viewerCoordsToImage(clientX, clientY) {
    const rect = viewer.getBoundingClientRect();
    const vx = clientX - rect.left;
    const vy = clientY - rect.top;
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const ix = (vx - cx - tx) / scale;
    const iy = (vy - cy - ty) / scale;
    return { x: ix, y: iy };
  }

  function drawOverlay() {
    const ctx = overlay.getContext('2d');
    overlay.width = imageWidth;
    overlay.height = imageHeight;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    // Sampled points (green) and reference points (red, larger)
    if (showSampledPointsEl && showSampledPointsEl.checked) {
      sampledPoints.forEach(function (p) {
        ctx.fillStyle = 'rgba(80, 220, 80, 0.7)';
        ctx.strokeStyle = 'rgba(80, 220, 80, 0.9)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
      referencePoints.forEach(function (p) {
        ctx.fillStyle = 'rgba(255, 80, 80, 0.6)';
        ctx.strokeStyle = 'rgba(255, 80, 80, 0.95)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }
    // ROI and category polygons (only when "Show annotations" is checked)
    if (showAnnotationsEl && showAnnotationsEl.checked) {
      const thinLine = 0.5;
      const vertexRadius = 1.5;
      // Category annotation polygons (different color per category) – thin stroke, light fill
      categoryAnnotations.forEach(function (ann) {
        const p = ann.polygon;
        if (!p || p.length < 2) return;
        const c = Math.min(ann.category, CATEGORY_COLORS.length - 1);
        ctx.fillStyle = CATEGORY_COLORS[c].replace(/0\.25\)/, '0.08)');
        ctx.strokeStyle = CATEGORY_STROKES[c];
        ctx.lineWidth = thinLine;
        ctx.beginPath();
        ctx.moveTo(p[0].x, p[0].y);
        for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      });
      // Current ROI polygon (yellow) – thin stroke, light fill
      if (polygon.length >= 2) {
        ctx.strokeStyle = 'rgba(255, 200, 0, 0.9)';
        ctx.lineWidth = thinLine;
        ctx.fillStyle = 'rgba(255, 200, 0, 0.08)';
        ctx.beginPath();
        ctx.moveTo(polygon[0].x, polygon[0].y);
        for (let i = 1; i < polygon.length; i++) {
          ctx.lineTo(polygon[i].x, polygon[i].y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        polygon.forEach(function (p, i) {
          ctx.fillStyle = 'rgba(255, 200, 0, 0.9)';
          ctx.beginPath();
          ctx.arc(p.x, p.y, vertexRadius, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    }
  }

  function updateCategoryCount() {
    if (categoryCountEl) categoryCountEl.textContent = categoryAnnotations.length ? categoryAnnotations.length + ' category regions' : '';
  }

  function updateButtons() {
    const hasPolygon = polygon.length >= 3;
    const hasRoiToUse = hasPolygon || (lastSentRoi && lastSentRoi.length >= 3);
    btnComplete.disabled = !drawMode;
    btnRecompute.disabled = !hasRoiToUse;
    btnClear.disabled = polygon.length === 0;
    if (btnAddCategory) btnAddCategory.disabled = !hasPolygon;
    if (btnClearCategories) btnClearCategories.disabled = categoryAnnotations.length === 0;
    updateCategoryCount();
  }

  function enterAnnotationMode() {
    drawMode = true;
    viewer.classList.add('draw');
    overlay.classList.add('pointer-events');
    overlay.style.pointerEvents = 'auto';
    setStatus('Annotation mode: hold mouse and drag to draw ROI, release to finish. Press A or Esc to exit.');
    updateButtons();
  }

  function exitAnnotationMode() {
    drawMode = false;
    drawingStroke = false;
    viewer.classList.remove('draw');
    overlay.classList.remove('pointer-events');
    setStatus('Pan/zoom mode. Press A to enter annotation mode.');
    updateButtons();
  }

  function loadImageFromBase64(b64, preserveView) {
    return new Promise(function (resolve, reject) {
      const dataUrl = 'data:image/png;base64,' + b64;
      img.onload = function () {
        imageWidth = img.naturalWidth;
        imageHeight = img.naturalHeight;
        overlay.width = imageWidth;
        overlay.height = imageHeight;
        overlay.style.width = imageWidth + 'px';
        overlay.style.height = imageHeight + 'px';
        if (!preserveView) {
          scale = 1;
          tx = -imageWidth / 2;
          ty = -imageHeight / 2;
        }
        applyTransform();
        drawOverlay();
        resolve();
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  async function fetchSampledPoints() {
    try {
      const r = await fetch(API_BASE + '/api/sampled_points');
      if (!r.ok) return;
      const data = await r.json();
      sampledPoints = (data.sampled || []).map(function (xy) { return { x: xy[0], y: xy[1] }; });
      referencePoints = (data.reference || []).map(function (xy) { return { x: xy[0], y: xy[1] }; });
      drawOverlay();
    } catch (e) { /* ignore */ }
  }

  async function fetchInputFiles() {
    try {
      const r = await fetch(API_BASE + '/api/input_files');
      if (!r.ok) return;
      const data = await r.json();
      const paths = data.paths || [];
      const current = data.current || null;
      inputFileSelect.innerHTML = '';
      if (paths.length === 0) {
        inputFileSelect.appendChild(new Option('—', ''));
        return;
      }
      paths.forEach(function (p) {
        const label = p.split(/[/\\]/).pop() || p;
        const opt = new Option(label, p);
        inputFileSelect.appendChild(opt);
      });
      currentLoadedPath = current;
      if (current && paths.indexOf(current) >= 0) {
        inputFileSelect.value = current;
      } else if (paths.length) {
        inputFileSelect.value = paths[0];
      }
    } catch (e) { /* ignore */ }
  }

  async function fetchViz() {
    setStatus('Loading visualization…', 'busy');
    showLoading(true);
    try {
      const r = await fetch(API_BASE + '/api/viz');
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      await loadImageFromBase64(data.image_base64);
      sampledPoints = [];
      referencePoints = [];
      if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
      await fetchInputFiles();
      setStatus('Ready. Press A to enter annotation mode; hold mouse and drag to draw ROI.');
      showLoading(false);
    } catch (e) {
      setStatus('Error: ' + e.message, 'error');
      showLoading(false);
    }
  }

  async function recompute() {
    const roiPoints = polygon.length >= 3 ? polygon : lastSentRoi;
    if (!roiPoints || roiPoints.length < 3) return;
    setStatus('Recomputing…', 'busy');
    showLoading(true);
    btnRecompute.disabled = true;
    try {
      const roi = roiPoints.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; });
      lastSentRoi = roiPoints.slice(); // keep so we can Recompute again without redrawing ROI
      const category_annotations = categoryAnnotations.map(function (a) {
        return { polygon: a.polygon.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; }), category: a.category };
      });
      const body = { roi: roi };
      if (category_annotations.length > 0) body.category_annotations = category_annotations;
      const r = await fetch(API_BASE + '/api/recompute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      await loadImageFromBase64(data.image_base64, true);
      sampledPoints = [];
      referencePoints = [];
      if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
      setStatus('Updated. You can draw a new ROI and Recompute again.');
    } catch (e) {
      setStatus('Error: ' + e.message, 'error');
    } finally {
      showLoading(false);
      updateButtons();
    }
  }

  if (inputFileSelect) {
    inputFileSelect.addEventListener('change', async function () {
      const path = inputFileSelect.value;
      if (!path || path === currentLoadedPath) return;
      setStatus('Loading image…', 'busy');
      showLoading(true);
      try {
        const r = await fetch(API_BASE + '/api/load_image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: path })
        });
        if (!r.ok) throw new Error((await r.json().catch(function () { return { detail: r.statusText }; })).detail || r.statusText);
        const data = await r.json();
        await loadImageFromBase64(data.image_base64, true);
        currentLoadedPath = path;
        lastSentRoi = null;
        sampledPoints = [];
        referencePoints = [];
        if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
        setStatus('Image loaded.');
      } catch (e) {
        setStatus('Load failed: ' + e.message, 'error');
      }
      showLoading(false);
    });
  }
  if (showAnnotationsEl) showAnnotationsEl.addEventListener('change', drawOverlay);
  showSampledPointsEl.addEventListener('change', function () {
    if (showSampledPointsEl.checked) fetchSampledPoints();
    else {
      sampledPoints = [];
      referencePoints = [];
      drawOverlay();
    }
  });

  async function fetchConfig() {
    try {
      const r = await fetch(API_BASE + '/api/config');
      if (!r.ok) return;
      const data = await r.json();
      const m = data.model || {};
      if (configNumberOfPoints) configNumberOfPoints.value = m.number_of_points ?? '';
      if (configNumSamples) configNumSamples.value = m.num_samples ?? '';
      if (configNumEpochs) configNumEpochs.value = m.num_epochs ?? '';
      if (configClusters) configClusters.value = m.clusters ?? '';
      if (configBatchSize) configBatchSize.value = m.batch_size ?? '';
      if (configBeta) configBeta.value = m.beta ?? '';
      if (configClusterLossWeight) configClusterLossWeight.value = m.cluster_loss_weight ?? '';
      if (configCategoryLossWeight) configCategoryLossWeight.value = m.category_loss_weight ?? '';
      if (configPixelSampling) configPixelSampling.value = (m.pixel_sampling || 'superpixel').toLowerCase();
      if (configPcaFitStep) configPcaFitStep.value = m.pca_fit_step ?? '';
      if (configClusterOnPca) configClusterOnPca.checked = !!m.cluster_on_pca;
      if (configClusterPcaDims) configClusterPcaDims.value = m.cluster_pca_dims ?? '';
    } catch (e) { /* ignore */ }
  }

  function openSettings() {
    settingsPanel.classList.toggle('open');
    if (settingsPanel.classList.contains('open')) fetchConfig();
  }

  function getConfigFromForm() {
    const model = {};
    if (configNumberOfPoints && configNumberOfPoints.value !== '') model.number_of_points = parseInt(configNumberOfPoints.value, 10);
    if (configNumSamples && configNumSamples.value !== '') model.num_samples = parseInt(configNumSamples.value, 10);
    if (configNumEpochs && configNumEpochs.value !== '') model.num_epochs = parseInt(configNumEpochs.value, 10);
    if (configClusters && configClusters.value.trim() !== '') model.clusters = configClusters.value.trim();
    if (configBatchSize && configBatchSize.value !== '') model.batch_size = parseInt(configBatchSize.value, 10);
    if (configBeta && configBeta.value !== '') model.beta = parseFloat(configBeta.value);
    if (configClusterLossWeight && configClusterLossWeight.value !== '') model.cluster_loss_weight = parseFloat(configClusterLossWeight.value);
    if (configCategoryLossWeight && configCategoryLossWeight.value !== '') model.category_loss_weight = parseFloat(configCategoryLossWeight.value);
    if (configPixelSampling && configPixelSampling.value) model.pixel_sampling = configPixelSampling.value.trim().toLowerCase() || 'superpixel';
    if (configPcaFitStep && configPcaFitStep.value !== '') model.pca_fit_step = Math.max(1, parseInt(configPcaFitStep.value, 10));
    if (configClusterOnPca) model.cluster_on_pca = configClusterOnPca.checked;
    if (configClusterPcaDims && configClusterPcaDims.value !== '') model.cluster_pca_dims = Math.max(1, parseInt(configClusterPcaDims.value, 10));
    return { model: Object.keys(model).length ? model : null };
  }

  async function saveConfig() {
    try {
      const body = getConfigFromForm();
      const r = await fetch(API_BASE + '/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) throw new Error((await r.json().catch(function () { return {}; })).detail || r.statusText);
      setStatus('Config saved. Recompute or reload viz to apply.');
    } catch (e) {
      setStatus('Config save failed: ' + e.message, 'error');
    }
  }

  async function applyAndReloadConfig() {
    try {
      const body = getConfigFromForm();
      const rPatch = await fetch(API_BASE + '/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!rPatch.ok) throw new Error((await rPatch.json().catch(function () { return {}; })).detail || rPatch.statusText);
      setStatus('Reloading visualization with new config…', 'busy');
      showLoading(true);
      const roi = polygon.length >= 3 ? polygon.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; }) : null;
      const reloadBody = roi ? { roi: roi } : {};
      if (categoryAnnotations.length > 0) {
        reloadBody.category_annotations = categoryAnnotations.map(function (a) {
          return { polygon: a.polygon.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; }), category: a.category };
        });
      }
      const rReload = await fetch(API_BASE + '/api/reload_viz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reloadBody)
      });
      if (!rReload.ok) throw new Error((await rReload.json().catch(function () { return {}; })).detail || rReload.statusText);
      const data = await rReload.json();
      await loadImageFromBase64(data.image_base64, true);
      sampledPoints = [];
      referencePoints = [];
      if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
      setStatus(reloadBody.roi ? 'Config applied. Viz reloaded (ROI).' : 'Config applied. Viz reloaded (full image).');
      showLoading(false);
    } catch (e) {
      setStatus('Apply failed: ' + e.message, 'error');
      showLoading(false);
    }
  }

  async function copyVizToClipboard() {
    if (!img.src || img.naturalWidth === 0) {
      setStatus('No image to copy.', 'error');
      return;
    }
    try {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const blob = await new Promise(function (resolve) {
        canvas.toBlob(resolve, 'image/png');
      });
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
      setStatus('Visualization copied to clipboard.');
    } catch (e) {
      setStatus('Copy failed: ' + e.message + ' (may need HTTPS or user permission)', 'error');
    }
  }

  if (btnSettings) btnSettings.addEventListener('click', openSettings);
  if (btnSettingsClose) btnSettingsClose.addEventListener('click', function () { settingsPanel.classList.remove('open'); });
  if (btnConfigSave) btnConfigSave.addEventListener('click', saveConfig);
  if (btnConfigApply) btnConfigApply.addEventListener('click', applyAndReloadConfig);
  if (btnCopyClipboard) btnCopyClipboard.addEventListener('click', copyVizToClipboard);

  function onWheel(e) {
    e.preventDefault();
    const rect = viewer.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.2, Math.min(10, scale * factor));
    tx -= dx * (newScale / scale - 1);
    ty -= dy * (newScale / scale - 1);
    scale = newScale;
    applyTransform();
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    if (drawMode) {
      const pt = viewerCoordsToImage(e.clientX, e.clientY);
      const x = Math.round(pt.x);
      const y = Math.round(pt.y);
      // Start new polygon (replaces current in-progress one; saved category annotations are kept)
      polygon = [{ x: x, y: y }];
      drawingStroke = true;
      drawOverlay();
      updateButtons();
      return;
    }
    lastMouse.x = e.clientX;
    lastMouse.y = e.clientY;
    dragging = true;
    viewer.classList.add('dragging');
  }

  function onMouseMove(e) {
    if (drawMode && drawingStroke) {
      const pt = viewerCoordsToImage(e.clientX, e.clientY);
      const x = Math.round(pt.x);
      const y = Math.round(pt.y);
      const last = polygon[polygon.length - 1];
      const dist = Math.hypot(x - last.x, y - last.y);
      if (dist >= MIN_POINT_DIST) {
        polygon.push({ x: x, y: y });
        drawOverlay();
        updateButtons();
      }
      return;
    }
    if (dragging) {
      const dx = e.clientX - lastMouse.x;
      const dy = e.clientY - lastMouse.y;
      lastMouse.x = e.clientX;
      lastMouse.y = e.clientY;
      tx += dx;
      ty += dy;
      applyTransform();
    }
  }

  function onMouseUp(e) {
    if (e.button !== 0) return;
    if (drawMode && drawingStroke) {
      drawingStroke = false;
      updateButtons();
      return;
    }
    dragging = false;
    viewer.classList.remove('dragging');
  }

  function onKeyDown(e) {
    if (e.key === 'a' || e.key === 'A') {
      e.preventDefault();
      drawMode = !drawMode;
      if (drawMode) enterAnnotationMode();
      else exitAnnotationMode();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      if (drawMode) {
        drawMode = false;
        drawingStroke = false;
        viewer.classList.remove('draw');
        overlay.classList.remove('pointer-events');
        setStatus('Pan/zoom mode. Press A to enter annotation mode.');
        updateButtons();
      }
    }
  }

  btnComplete.addEventListener('click', function () {
    exitAnnotationMode();
  });

  btnClear.addEventListener('click', function () {
    polygon = [];
    drawOverlay();
    updateButtons();
  });

  if (btnAddCategory) {
    btnAddCategory.addEventListener('click', function () {
      if (polygon.length < 3) return;
      const cat = parseInt(categoryIdEl && categoryIdEl.value !== '' ? categoryIdEl.value : 0, 10);
      // Deep copy so clearing polygon later never affects saved annotations
      const copy = polygon.map(function (p) { return { x: p.x, y: p.y }; });
      categoryAnnotations.push({ polygon: copy, category: cat });
      polygon = [];
      drawOverlay();
      updateButtons();
      setStatus('Added as category ' + cat + ' (' + categoryAnnotations.length + ' total). Draw another or Recompute.');
    });
  }
  if (btnClearCategories) {
    btnClearCategories.addEventListener('click', function () {
      categoryAnnotations = [];
      drawOverlay();
      updateButtons();
      setStatus('Category annotations cleared.');
    });
  }

  btnRecompute.addEventListener('click', recompute);

  viewer.addEventListener('wheel', onWheel, { passive: false });
  viewer.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);
  window.addEventListener('keydown', onKeyDown);

  fetchViz();
})();
