(function () {
  'use strict';

  const API_BASE = ''; // same origin
  const viewer = document.getElementById('viewer');
  const viewport = document.getElementById('viewport');
  const vizContainer = document.getElementById('vizContainer');
  const img = document.getElementById('img');
  const overlay = document.getElementById('overlay');
  const emptyState = document.getElementById('emptyState');
  const loadingEl = document.getElementById('loading');
  const statusEl = document.getElementById('status');
  const btnComplete = document.getElementById('btnComplete');
  const btnClear = document.getElementById('btnClear');
  const btnRecompute = document.getElementById('btnRecompute');
  const btnRankNmfRegion = document.getElementById('btnRankNmfRegion');
  const showAnnotationsEl = document.getElementById('showAnnotations');
  const showSampledPointsEl = document.getElementById('showSampledPoints');
  const btnSettings = document.getElementById('btnSettings');
  const settingsPanel = document.getElementById('settingsPanel');
  const configModelType = document.getElementById('configModelType');

  /** Ensure Model type lists classical msi_visual methods (covers browsers serving cached old index.html). */
  function ensureModelTypeOptions() {
    if (!configModelType) return;
    const entries = [
      ['top3', 'TOP3 (pseudo-RGB)'],
      ['percentile_ratio', 'Percentile ratio'],
      ['nmf_3d', 'NMF3D (RGB)'],
      ['kmeans_segmentation', 'K-means segmentation'],
      ['nmf_segmentation', 'NMF segmentation'],
      ['nonparametric_umap', 'Nonparametric UMAP (3D)'],
    ];
    const have = new Set();
    for (let i = 0; i < configModelType.options.length; i++) {
      have.add(configModelType.options[i].value);
    }
    entries.forEach(function (row) {
      if (!have.has(row[0])) {
        const o = document.createElement('option');
        o.value = row[0];
        o.textContent = row[1];
        configModelType.appendChild(o);
      }
    });
  }
  const configNumberOfPoints = document.getElementById('configNumberOfPoints');
  const configNumSamples = document.getElementById('configNumSamples');
  const configNumEpochs = document.getElementById('configNumEpochs');
  const configInputPath = document.getElementById('configInputPath');
  const configAppOutput = document.getElementById('configAppOutput');
  const configPanelOutput = document.getElementById('configPanelOutput');
  const btnSaveOutputPaths = document.getElementById('btnSaveOutputPaths');
  const btnResetOutputPaths = document.getElementById('btnResetOutputPaths');
  const configNumberOfComponents = document.getElementById('configNumberOfComponents');
  const configNumLayers = document.getElementById('configNumLayers');
  const configFactor = document.getElementById('configFactor');
  const configLr = document.getElementById('configLr');
  const configWarmupEpochs = document.getElementById('configWarmupEpochs');
  const configClusters = document.getElementById('configClusters');
  const configBatchSize = document.getElementById('configBatchSize');
  const configBeta = document.getElementById('configBeta');
  const configClusterLossWeight = document.getElementById('configClusterLossWeight');
  const configCategoryLossWeight = document.getElementById('configCategoryLossWeight');
  const configPixelSampling = document.getElementById('configPixelSampling');
  const configPcaFitStep = document.getElementById('configPcaFitStep');
  const configClusterOnPca = document.getElementById('configClusterOnPca');
  const configClusterPcaDims = document.getElementById('configClusterPcaDims');
  const configUmapNeighbors = document.getElementById('configUmapNeighbors');
  const configUmapTrainingEpochs = document.getElementById('configUmapTrainingEpochs');
  const configUmapEpochs = document.getElementById('configUmapEpochs');
  const configSimpleK = document.getElementById('configSimpleK');
  const configNmf3dMaxIter = document.getElementById('configNmf3dMaxIter');
  const configNmf3dNComponents = document.getElementById('configNmf3dNComponents');
  const configNmfSegMaxIter = document.getElementById('configNmfSegMaxIter');
  const configTop3Low = document.getElementById('configTop3Low');
  const configTop3Norm = document.getElementById('configTop3Norm');
  const configUmapNpMinDist = document.getElementById('configUmapNpMinDist');
  const configUmapNpNeighbors = document.getElementById('configUmapNpNeighbors');
  const configUmapNpMetric = document.getElementById('configUmapNpMetric');
  const btnConfigApply = document.getElementById('btnConfigApply');
  const btnConfigExport = document.getElementById('btnConfigExport');
  const btnFastPreset = document.getElementById('btnFastPreset');
  const configImportSelect = document.getElementById('configImportSelect');
  const btnConfigImport = document.getElementById('btnConfigImport');
  const actionConfigImportSelect = document.getElementById('actionConfigImportSelect');
  const btnActionConfigImport = document.getElementById('btnActionConfigImport');
  const autoLoadOnSelect = document.getElementById('autoLoadOnSelect');
  const btnCopyClipboard = document.getElementById('btnCopyClipboard');
  const btnSaveImage = document.getElementById('btnSaveImage');
  const btnComposePanel = document.getElementById('btnComposePanel');
  const btnParamSweep = document.getElementById('btnParamSweep');
  const sweepModal = document.getElementById('sweepModal');
  const btnCloseSweep = document.getElementById('btnCloseSweep');
  const sweepMeta = document.getElementById('sweepMeta');
  const sweepFields = document.getElementById('sweepFields');
  const sweepOverridePreview = document.getElementById('sweepOverridePreview');
  const sweepMaxRuns = document.getElementById('sweepMaxRuns');
  const sweepPanelCols = document.getElementById('sweepPanelCols');
  const sweepBuildPanel = document.getElementById('sweepBuildPanel');
  const btnSweepStart = document.getElementById('btnSweepStart');
  const btnSweepCancel = document.getElementById('btnSweepCancel');
  const sweepProgress = document.getElementById('sweepProgress');
  const sweepConfigSelect = document.getElementById('sweepConfigSelect');
  const sweepConfigName = document.getElementById('sweepConfigName');
  const btnSweepConfigLoad = document.getElementById('btnSweepConfigLoad');
  const btnSweepConfigSave = document.getElementById('btnSweepConfigSave');
  const btnSweepConfigDelete = document.getElementById('btnSweepConfigDelete');
  const sweepInputSelect = document.getElementById('sweepInputSelect');
  const clusterSweepEnable = document.getElementById('clusterSweepEnable');
  const clusterSweepMinK = document.getElementById('clusterSweepMinK');
  const clusterSweepMaxK = document.getElementById('clusterSweepMaxK');
  const clusterSweepConsecutive = document.getElementById('clusterSweepConsecutive');
  const clusterSweepPool = document.getElementById('clusterSweepPool');
  const clusterSweepPreview = document.getElementById('clusterSweepPreview');
  const btnClusterSweepApply = document.getElementById('btnClusterSweepApply');
  const composePanelModal = document.getElementById('composePanelModal');
  const btnCloseComposePanel = document.getElementById('btnCloseComposePanel');
  const panelFolderSelect = document.getElementById('panelFolderSelect');
  const panelCustomPath = document.getElementById('panelCustomPath');
  const btnPanelLoadPath = document.getElementById('btnPanelLoadPath');
  const panelOutputPath = document.getElementById('panelOutputPath');
  const panelSearch = document.getElementById('panelSearch');
  const panelPickGrid = document.getElementById('panelPickGrid');
  const panelSelectedList = document.getElementById('panelSelectedList');
  const panelCols = document.getElementById('panelCols');
  const panelRows = document.getElementById('panelRows');
  const panelCellMax = document.getElementById('panelCellMax');
  const panelNoCaptions = document.getElementById('panelNoCaptions');
  const panelKeyList = document.getElementById('panelKeyList');
  const btnPanelPreview = document.getElementById('btnPanelPreview');
  const btnPanelSave = document.getElementById('btnPanelSave');
  const panelPreviewImg = document.getElementById('panelPreviewImg');
  const btnSettingsClose = document.getElementById('btnSettingsClose');
  const btnRotateLeft = document.getElementById('btnRotateLeft');
  const btnRotateRight = document.getElementById('btnRotateRight');
  const btnResetView = document.getElementById('btnResetView');
  const btnFitView = document.getElementById('btnFitView');
  const categoryIdEl = document.getElementById('categoryId');
  const btnAddCategory = document.getElementById('btnAddCategory');
  const btnClearCategories = document.getElementById('btnClearCategories');
  const categoryCountEl = document.getElementById('categoryCount');
  const inputFileSelect = document.getElementById('inputFileSelect');
  const nmfComponents = document.getElementById('nmfComponents');
  const nmfRankModal = document.getElementById('nmfRankModal');
  const btnCloseNmfModal = document.getElementById('btnCloseNmfModal');
  const nmfRankTitle = document.getElementById('nmfRankTitle');
  const nmfRankSummary = document.getElementById('nmfRankSummary');
  const nmfRankGrid = document.getElementById('nmfRankGrid');

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
  let rotationDeg = 0;
  let selectedInputPath = null;

  const CATEGORY_COLORS = ['rgba(100, 150, 255, 0.25)', 'rgba(100, 255, 150, 0.25)', 'rgba(255, 150, 255, 0.25)', 'rgba(255, 200, 100, 0.25)', 'rgba(100, 255, 255, 0.25)'];
  const CATEGORY_STROKES = ['rgba(100, 150, 255, 0.9)', 'rgba(100, 255, 150, 0.9)', 'rgba(255, 150, 255, 0.9)', 'rgba(255, 200, 100, 0.9)', 'rgba(100, 255, 255, 0.9)'];

  function setStatus(text, className = '') {
    statusEl.textContent = text;
    statusEl.className = 'status-bar ' + (className || '');
  }


  function showLoading(show) {
    loadingEl.style.display = show ? 'flex' : 'none';
  }

  function showEmptyState(show) {
    if (!emptyState) return;
    emptyState.classList.toggle('show', !!show);
  }

  function applyTransform() {
    viewport.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    if (vizContainer) {
      vizContainer.style.transform = `rotate(${rotationDeg}deg)`;
    }
  }

  function getBaseDims() {
    if (rotationDeg % 180 !== 0) {
      return { w: imageHeight, h: imageWidth };
    }
    return { w: imageWidth, h: imageHeight };
  }

  function fitToViewer() {
    if (!imageWidth || !imageHeight) return;
    const rect = viewer.getBoundingClientRect();
    const base = getBaseDims();
    const s = Math.min(rect.width / base.w, rect.height / base.h);
    scale = Math.max(0.05, s);
    tx = -base.w / 2;
    ty = -base.h / 2;
    applyTransform();
  }

  function rotateBy(delta) {
    rotationDeg = (rotationDeg + delta + 360) % 360;
    fitToViewer();
  }

  function resetView() {
    rotationDeg = 0;
    fitToViewer();
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
    if (btnRankNmfRegion) btnRankNmfRegion.disabled = !hasRoiToUse;
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
        showEmptyState(false);
        imageWidth = img.naturalWidth;
        imageHeight = img.naturalHeight;
        overlay.width = imageWidth;
        overlay.height = imageHeight;
        overlay.style.width = imageWidth + 'px';
        overlay.style.height = imageHeight + 'px';
        if (!preserveView) {
          rotationDeg = 0;
          fitToViewer();
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
        let label = p;
        const baseDir = configInputPath && configInputPath.value ? configInputPath.value.trim() : '';
        if (baseDir && !baseDir.toLowerCase().endsWith('.npy')) {
          const baseNorm = baseDir.replace(/\\/g, '/').replace(/\/+$/, '');
          const pathNorm = p.replace(/\\/g, '/');
          if (pathNorm.toLowerCase().startsWith(baseNorm.toLowerCase() + '/')) {
            label = pathNorm.slice(baseNorm.length + 1);
          }
        } else {
          label = p.split(/[/\\]/).pop() || p;
        }
        const opt = new Option(label, p);
        inputFileSelect.appendChild(opt);
      });
      currentLoadedPath = current;
      if (current && paths.indexOf(current) >= 0) {
        inputFileSelect.value = current;
      } else if (paths.length) {
        inputFileSelect.value = paths[0];
      }
      selectedInputPath = inputFileSelect.value || null;
    } catch (e) { /* ignore */ }
  }

  async function fetchViz() {
    setStatus('Loading visualization…', 'busy');
    showLoading(true);
    showEmptyState(false);
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
    const inputPath = selectedInputPath || currentLoadedPath;
    if (!roiPoints || roiPoints.length < 3) {
      if (!inputPath) return;
    }
    setStatus('Recomputing…', 'busy');
    showLoading(true);
    btnRecompute.disabled = true;
    try {
      let roi = null;
      if (roiPoints && roiPoints.length >= 3) {
        roi = roiPoints.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; });
        lastSentRoi = roiPoints.slice(); // keep so we can Recompute again without redrawing ROI
      }
      const category_annotations = categoryAnnotations.map(function (a) {
        return { polygon: a.polygon.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; }), category: a.category };
      });
      const body = {};
      if (roi) body.roi = roi;
      if (inputPath) body.input_path = inputPath;
      if (category_annotations.length > 0) body.category_annotations = category_annotations;
      const r = await fetch(API_BASE + '/api/reload_viz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      await loadImageFromBase64(data.image_base64, false);
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

  function closeNmfModal() {
    if (!nmfRankModal) return;
    nmfRankModal.classList.remove('open');
  }

  function openNmfModal() {
    if (!nmfRankModal) return;
    nmfRankModal.classList.add('open');
  }

  function renderNmfRanking(data) {
    if (!nmfRankGrid || !nmfRankSummary || !nmfRankTitle) return;
    const comps = Array.isArray(data.components) ? data.components : [];
    nmfRankTitle.textContent = 'NMF component ranking (LR Spearman)';
    nmfRankSummary.textContent = `Components: ${data.n_components || comps.length} | Region pixels: ${data.n_region_pixels || 0}`;
    nmfRankGrid.innerHTML = '';
    comps.forEach(function (c) {
      const card = document.createElement('div');
      card.className = 'nmf-card';
      const score = Number(c.lr_spearman);
      const scoreText = Number.isFinite(score) ? score.toFixed(4) : 'NaN';
      card.innerHTML =
        `<img alt="NMF component ${c.component}" src="data:image/png;base64,${c.image_base64}" />` +
        `<div class="nmf-card-meta">` +
        `#${c.rank} | comp ${c.component}<br/>lr_spearman: ${scoreText}` +
        `</div>`;
      nmfRankGrid.appendChild(card);
    });
    openNmfModal();
  }

  async function rankNmfForRegion() {
    const roiPoints = polygon.length >= 3 ? polygon : lastSentRoi;
    if (!roiPoints || roiPoints.length < 3) {
      setStatus('Draw/select an ROI first.', 'error');
      return;
    }
    const nComp = nmfComponents && nmfComponents.value ? parseInt(nmfComponents.value, 10) : 32;
    console.info('[annotator] rank_nmf_for_region start', { roiPoints: roiPoints.length, nComp: nComp });
    setStatus('Scoring NMF components in ROI…', 'busy');
    showLoading(true);
    if (btnRankNmfRegion) btnRankNmfRegion.disabled = true;
    try {
      const roi = roiPoints.map(function (p) { return [Math.round(p.x), Math.round(p.y)]; });
      const r = await fetch(API_BASE + '/api/region_nmf_ranking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi: roi, n_components: Number.isFinite(nComp) ? nComp : 32 })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      console.info('[annotator] rank_nmf_for_region success', {
        n_components: data.n_components,
        n_region_pixels: data.n_region_pixels,
        returned_components: Array.isArray(data.components) ? data.components.length : 0
      });
      renderNmfRanking(data);
      setStatus('NMF ranking ready.');
    } catch (e) {
      console.error('[annotator] rank_nmf_for_region failed', e);
      setStatus('NMF ranking failed: ' + e.message, 'error');
    } finally {
      showLoading(false);
      updateButtons();
    }
  }

  if (inputFileSelect) {
    inputFileSelect.addEventListener('change', async function () {
      const path = inputFileSelect.value;
      if (!path || path === currentLoadedPath) return;
      selectedInputPath = path;
      lastSentRoi = null;
      sampledPoints = [];
      referencePoints = [];
      drawOverlay();
      updateButtons();
      if (autoLoadOnSelect && autoLoadOnSelect.checked) {
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
          await loadImageFromBase64(data.image_base64, false);
          currentLoadedPath = path;
          if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
          setStatus('Image loaded.');
          updateButtons();
        } catch (e) {
          setStatus('Load failed: ' + e.message, 'error');
        } finally {
          showLoading(false);
        }
      } else {
        setStatus('Image selected. Press Apply or Recompute to load.');
        btnRecompute.disabled = false;
      }
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

  async function fetchOutputPaths() {
    try {
      const r = await fetch(API_BASE + '/api/output_paths');
      if (!r.ok) return;
      const data = await r.json();
      if (configAppOutput) configAppOutput.value = data.app_output || '';
      if (configPanelOutput) configPanelOutput.value = data.panel_output || '';
      if (panelOutputPath) panelOutputPath.value = data.panel_output || '';
      if (configAppOutput && data.defaults) {
        configAppOutput.placeholder = data.defaults.app_output || configAppOutput.placeholder;
      }
      if (configPanelOutput && data.defaults) {
        configPanelOutput.placeholder = data.defaults.panel_output || configPanelOutput.placeholder;
      }
      if (panelOutputPath && data.defaults) {
        panelOutputPath.placeholder = data.defaults.panel_output || panelOutputPath.placeholder;
      }
    } catch (e) { /* ignore */ }
  }

  async function saveOutputPaths(reset) {
    try {
      const body = reset
        ? { app_output: '', panel_output: '' }
        : {
            app_output: configAppOutput ? configAppOutput.value.trim() : '',
            panel_output: configPanelOutput ? configPanelOutput.value.trim() : ''
          };
      const r = await fetch(API_BASE + '/api/output_paths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      if (configAppOutput) configAppOutput.value = data.app_output || '';
      if (configPanelOutput) configPanelOutput.value = data.panel_output || '';
      if (panelOutputPath) panelOutputPath.value = data.panel_output || '';
      setStatus(reset
        ? 'Output paths reset to defaults.'
        : 'Output paths saved. App: ' + data.app_output + ' · Panels: ' + data.panel_output);
    } catch (e) {
      setStatus('Output paths failed: ' + e.message, 'error');
    }
  }

  async function fetchConfig() {
    try {
      const r = await fetch(API_BASE + '/api/config');
      if (!r.ok) return;
      const data = await r.json();
      const m = data.model || {};
      const d = data.data || {};
      if (configInputPath) configInputPath.value = d.input_path ?? '';
      if (configModelType) configModelType.value = (m.model_type || 'mics_lmc').toLowerCase();
      if (configNumberOfPoints) configNumberOfPoints.value = m.number_of_points ?? '';
      if (configNumSamples) configNumSamples.value = m.num_samples ?? '';
      if (configNumEpochs) configNumEpochs.value = m.num_epochs ?? '';
      if (configNumberOfComponents) configNumberOfComponents.value = m.number_of_components ?? '';
      if (configNumLayers) configNumLayers.value = m.num_layers ?? '';
      if (configFactor) configFactor.value = m.factor ?? '';
      if (configLr) configLr.value = m.lr ?? '';
      if (configWarmupEpochs) configWarmupEpochs.value = m.warmup_epochs ?? '';
      if (configClusters) configClusters.value = m.clusters ?? '';
      if (configBatchSize) configBatchSize.value = m.batch_size ?? '';
      if (configBeta) configBeta.value = m.beta ?? '';
      if (configClusterLossWeight) configClusterLossWeight.value = m.cluster_loss_weight ?? '';
      if (configCategoryLossWeight) configCategoryLossWeight.value = m.category_loss_weight ?? '';
      if (configPixelSampling) configPixelSampling.value = (m.pixel_sampling || 'superpixel').toLowerCase();
      if (configPcaFitStep) configPcaFitStep.value = m.pca_fit_step ?? '';
      if (configClusterOnPca) configClusterOnPca.checked = !!m.cluster_on_pca;
      if (configClusterPcaDims) configClusterPcaDims.value = m.cluster_pca_dims ?? '';
      if (configUmapNeighbors) configUmapNeighbors.value = m.umap_n_neighbors ?? '';
      if (configUmapTrainingEpochs) configUmapTrainingEpochs.value = m.umap_n_training_epochs ?? '';
      if (configUmapEpochs) configUmapEpochs.value = m.umap_n_epochs ?? '';
      if (configSimpleK) configSimpleK.value = m.simple_k ?? '';
      if (configNmf3dMaxIter) configNmf3dMaxIter.value = m.nmf_3d_max_iter ?? '';
      if (configNmf3dNComponents) configNmf3dNComponents.value = m.nmf_3d_n_components ?? '';
      if (configNmfSegMaxIter) configNmfSegMaxIter.value = m.nmf_segmentation_max_iter ?? '';
      if (configTop3Low) configTop3Low.value = m.top3_low ?? '';
      if (configTop3Norm) configTop3Norm.value = m.top3_norm_percentile ?? '';
      if (configUmapNpMinDist) configUmapNpMinDist.value = m.umap_np_min_dist ?? '';
      if (configUmapNpNeighbors) configUmapNpNeighbors.value = m.umap_np_n_neighbors ?? '';
      if (configUmapNpMetric) configUmapNpMetric.value = m.umap_np_metric ?? '';
    } catch (e) { /* ignore */ }
  }

  function openSettings() {
    settingsPanel.classList.toggle('open');
    if (settingsPanel.classList.contains('open')) {
      ensureModelTypeOptions();
      fetchConfig();
      fetchConfigFiles();
    }
  }

  async function importConfigFile(filename) {
    if (!filename) return;
    setStatus('Loading config…', 'busy');
    showLoading(true);
    try {
      const r = await fetch(API_BASE + '/api/configs/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      await r.json();
      await fetchConfig();
      await fetchInputFiles();
      setStatus('Config loaded. Press apply to recompute.');
    } catch (e) {
      setStatus('Import failed: ' + e.message, 'error');
    } finally {
      showLoading(false);
      updateButtons();
    }
  }

  function getConfigFromForm() {
    const model = {};
    const data = {};
    if (configInputPath) data.input_path = configInputPath.value.trim();
    if (configModelType && configModelType.value) model.model_type = configModelType.value.trim();
    if (configNumberOfPoints && configNumberOfPoints.value !== '') model.number_of_points = parseInt(configNumberOfPoints.value, 10);
    if (configNumSamples && configNumSamples.value !== '') model.num_samples = parseInt(configNumSamples.value, 10);
    if (configNumEpochs && configNumEpochs.value !== '') model.num_epochs = parseInt(configNumEpochs.value, 10);
    if (configNumberOfComponents && configNumberOfComponents.value !== '') model.number_of_components = parseInt(configNumberOfComponents.value, 10);
    if (configNumLayers && configNumLayers.value !== '') model.num_layers = Math.max(1, parseInt(configNumLayers.value, 10));
    if (configFactor && configFactor.value !== '') model.factor = parseFloat(configFactor.value);
    if (configLr && configLr.value !== '') model.lr = parseFloat(configLr.value);
    if (configWarmupEpochs && configWarmupEpochs.value !== '') model.warmup_epochs = parseInt(configWarmupEpochs.value, 10);
    if (configClusters && configClusters.value.trim() !== '') model.clusters = configClusters.value.trim();
    if (configBatchSize && configBatchSize.value !== '') model.batch_size = parseInt(configBatchSize.value, 10);
    if (configBeta && configBeta.value !== '') model.beta = parseFloat(configBeta.value);
    if (configClusterLossWeight && configClusterLossWeight.value !== '') model.cluster_loss_weight = parseFloat(configClusterLossWeight.value);
    if (configCategoryLossWeight && configCategoryLossWeight.value !== '') model.category_loss_weight = parseFloat(configCategoryLossWeight.value);
    if (configPixelSampling && configPixelSampling.value) model.pixel_sampling = configPixelSampling.value.trim().toLowerCase() || 'superpixel';
    if (configPcaFitStep && configPcaFitStep.value !== '') model.pca_fit_step = Math.max(1, parseInt(configPcaFitStep.value, 10));
    if (configClusterOnPca) model.cluster_on_pca = configClusterOnPca.checked;
    if (configClusterPcaDims && configClusterPcaDims.value !== '') model.cluster_pca_dims = Math.max(1, parseInt(configClusterPcaDims.value, 10));
    if (configUmapNeighbors && configUmapNeighbors.value !== '') model.umap_n_neighbors = Math.max(1, parseInt(configUmapNeighbors.value, 10));
    if (configUmapTrainingEpochs && configUmapTrainingEpochs.value !== '') model.umap_n_training_epochs = Math.max(1, parseInt(configUmapTrainingEpochs.value, 10));
    if (configUmapEpochs && configUmapEpochs.value !== '') model.umap_n_epochs = Math.max(1, parseInt(configUmapEpochs.value, 10));
    if (configSimpleK && configSimpleK.value !== '') model.simple_k = Math.max(2, parseInt(configSimpleK.value, 10));
    if (configNmf3dMaxIter && configNmf3dMaxIter.value !== '') model.nmf_3d_max_iter = Math.max(1, parseInt(configNmf3dMaxIter.value, 10));
    if (configNmf3dNComponents && configNmf3dNComponents.value !== '') model.nmf_3d_n_components = Math.max(1, parseInt(configNmf3dNComponents.value, 10));
    if (configNmfSegMaxIter && configNmfSegMaxIter.value !== '') model.nmf_segmentation_max_iter = Math.max(1, parseInt(configNmfSegMaxIter.value, 10));
    if (configTop3Low && configTop3Low.value !== '') model.top3_low = parseFloat(configTop3Low.value);
    if (configTop3Norm && configTop3Norm.value !== '') model.top3_norm_percentile = parseFloat(configTop3Norm.value);
    if (configUmapNpMinDist && configUmapNpMinDist.value !== '') model.umap_np_min_dist = parseFloat(configUmapNpMinDist.value);
    if (configUmapNpNeighbors && configUmapNpNeighbors.value !== '') model.umap_np_n_neighbors = Math.max(2, parseInt(configUmapNpNeighbors.value, 10));
    if (configUmapNpMetric && configUmapNpMetric.value.trim() !== '') model.umap_np_metric = configUmapNpMetric.value.trim();
    return {
      model: Object.keys(model).length ? model : null,
      data: Object.keys(data).length ? data : null
    };
  }

  async function updateInputPathOnly() {
    if (!configInputPath) return;
    const body = { data: { input_path: configInputPath.value.trim() } };
    try {
      const r = await fetch(API_BASE + '/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      await fetchInputFiles();
      setStatus('Input folder updated.');
    } catch (e) {
      setStatus('Update failed: ' + e.message, 'error');
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
      const inputPath = selectedInputPath || currentLoadedPath;
      if (inputPath) reloadBody.input_path = inputPath;
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
      await loadImageFromBase64(data.image_base64, false);
      sampledPoints = [];
      referencePoints = [];
      if (showSampledPointsEl && showSampledPointsEl.checked) await fetchSampledPoints();
      await fetchConfig();
      await fetchInputFiles();
      setStatus(reloadBody.roi ? 'Config applied. Viz reloaded (ROI).' : 'Config applied. Viz reloaded (full image).');
      showLoading(false);
    } catch (e) {
      setStatus('Apply failed: ' + e.message, 'error');
      showLoading(false);
    }
  }

  async function exportConfig() {
    try {
      const r = await fetch(API_BASE + '/api/config/export');
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      const filename = data.filename || 'config_export.yaml';
      setStatus('Config exported to annotator configs (' + filename + ').');
      fetchConfigFiles();
    } catch (e) {
      setStatus('Export failed: ' + e.message, 'error');
    }
  }

  function applyFastPreset() {
    if (configPixelSampling) configPixelSampling.value = 'random';
    if (configPcaFitStep) configPcaFitStep.value = '32';
    if (configNumSamples) configNumSamples.value = '2000';
    if (configNumberOfPoints) configNumberOfPoints.value = '500';
    if (configNumEpochs) configNumEpochs.value = '50';
    if (configBatchSize) configBatchSize.value = '2048';
    if (configClusterOnPca) configClusterOnPca.checked = true;
    if (configClusterPcaDims) configClusterPcaDims.value = '20';
    if (configClusters) configClusters.value = '8-16-32';
    setStatus('Fast preset applied. Press apply to recompute.');
  }

  async function fetchConfigFiles() {
    if (!configImportSelect && !actionConfigImportSelect) return;
    try {
      let files = [];
      if (configImportSelect) {
        const r = await fetch(API_BASE + '/api/configs/list');
        if (r.ok) {
          const data = await r.json();
          files = data.files || [];
          const current = configImportSelect.value;
          configImportSelect.innerHTML = '';
          configImportSelect.appendChild(new Option('— Select config —', ''));
          files.forEach(function (name) {
            configImportSelect.appendChild(new Option(name, name));
          });
          if (current && files.indexOf(current) >= 0) configImportSelect.value = current;
        }
      }
      if (actionConfigImportSelect) {
        const r = await fetch(API_BASE + '/api/saved_images/list');
        if (r.ok) {
          const data = await r.json();
          files = data.files || [];
          const current = actionConfigImportSelect.value;
          actionConfigImportSelect.innerHTML = '';
        actionConfigImportSelect.appendChild(new Option('— Select yaml —', ''));
          files.forEach(function (name) {
            actionConfigImportSelect.appendChild(new Option(name, name));
          });
          if (current && files.indexOf(current) >= 0) actionConfigImportSelect.value = current;
        }
      }
    } catch (e) { /* ignore */ }
  }

  async function importConfig(fromActions) {
    const selectEl = fromActions ? actionConfigImportSelect : configImportSelect;
    if (!selectEl || !selectEl.value) {
      setStatus('Select a yaml first.', 'error');
      return;
    }
    setStatus(fromActions ? 'Loading yaml…' : 'Loading config…', 'busy');
    showLoading(true);
    try {
      const url = fromActions ? '/api/saved_images/import' : '/api/configs/import';
      const r = await fetch(API_BASE + url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectEl.value })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      await fetchConfig();
      await fetchInputFiles();
      setStatus('Yaml loaded. Press apply to recompute.');
    } catch (e) {
      setStatus('Import failed: ' + e.message, 'error');
    } finally {
      showLoading(false);
      updateButtons();
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

  async function saveImage() {
    if (!img.src || img.naturalWidth === 0) {
      setStatus('No image to save.', 'error');
      return;
    }
    try {
      const r = await fetch(API_BASE + '/api/save_image', { method: 'POST' });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      const savedPath = data.saved_path || '';
      const name = savedPath ? savedPath.split(/[/\\]/).pop() : '';
      setStatus(name ? ('Image saved (' + name + ').') : 'Image saved.');
    } catch (e) {
      setStatus('Save failed: ' + e.message, 'error');
    }
  }

  if (btnSettings) btnSettings.addEventListener('click', openSettings);
  if (btnSettingsClose) btnSettingsClose.addEventListener('click', function () { settingsPanel.classList.remove('open'); });
  if (btnConfigApply) btnConfigApply.addEventListener('click', applyAndReloadConfig);
  if (btnConfigExport) btnConfigExport.addEventListener('click', exportConfig);
  if (btnFastPreset) btnFastPreset.addEventListener('click', applyFastPreset);
  if (btnConfigImport) btnConfigImport.addEventListener('click', function () { importConfig(false); });
  if (btnActionConfigImport) btnActionConfigImport.addEventListener('click', function () { importConfig(true); });
  if (btnSaveOutputPaths) btnSaveOutputPaths.addEventListener('click', function () { saveOutputPaths(false); });
  if (btnResetOutputPaths) btnResetOutputPaths.addEventListener('click', function () { saveOutputPaths(true); });
  if (configInputPath) configInputPath.addEventListener('change', updateInputPathOnly);

  if (actionConfigImportSelect) {
    actionConfigImportSelect.addEventListener('focus', fetchConfigFiles);
    actionConfigImportSelect.addEventListener('click', fetchConfigFiles);
  }
  if (configImportSelect) {
    configImportSelect.addEventListener('focus', fetchConfigFiles);
    configImportSelect.addEventListener('click', fetchConfigFiles);
  }
  if (btnCopyClipboard) btnCopyClipboard.addEventListener('click', copyVizToClipboard);
  if (btnSaveImage) btnSaveImage.addEventListener('click', saveImage);

  const CAPTION_KEYS_STORAGE = 'annotator.panel.captionKeys';
  const NO_CAPTIONS_STORAGE = 'annotator.panel.noCaptions';
  const CUSTOM_PATH_STORAGE = 'annotator.panel.customPath';
  let panelUseCustomPath = false;
  const DEFAULT_CAPTION_KEYS = [
    'model.model_type', 'model.num_epochs', 'model.clusters', 'model.pixel_sampling', 'label'
  ];
  let panelCatalog = { items: [], caption_keys: [], folder: 'results' };
  let panelSelectedIds = [];
  let panelItemCache = {};
  let panelThumbObserver = null;
  const THUMB_PLACEHOLDER = 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="96"><rect width="100%" height="100%" fill="#1a1a1f"/></svg>'
  );

  function loadSavedCaptionKeys() {
    try {
      const raw = localStorage.getItem(CAPTION_KEYS_STORAGE);
      if (!raw) return DEFAULT_CAPTION_KEYS.slice();
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch (e) { /* ignore */ }
    return DEFAULT_CAPTION_KEYS.slice();
  }

  function saveCaptionKeys(keys) {
    try {
      localStorage.setItem(CAPTION_KEYS_STORAGE, JSON.stringify(keys || []));
    } catch (e) { /* ignore */ }
  }

  function loadSavedNoCaptions() {
    try {
      return localStorage.getItem(NO_CAPTIONS_STORAGE) === '1';
    } catch (e) {
      return false;
    }
  }

  function saveNoCaptions(off) {
    try {
      localStorage.setItem(NO_CAPTIONS_STORAGE, off ? '1' : '0');
    } catch (e) { /* ignore */ }
  }

  function closeComposePanel() {
    if (composePanelModal) composePanelModal.classList.remove('open');
  }

  function panelThumbUrl(id) {
    const parts = String(id || '').split('/');
    return API_BASE + '/api/panel/thumb/' + parts.map(encodeURIComponent).join('/');
  }

  function panelItemById(id) {
    if (panelItemCache[id]) return panelItemCache[id];
    const items = panelCatalog.items || [];
    for (let i = 0; i < items.length; i++) {
      if (items[i].id === id) return items[i];
    }
    return null;
  }

  function renderPanelKeys() {
    if (!panelKeyList) return;
    const available = panelCatalog.caption_keys || [];
    const checkedNow = {};
    let haveUiState = false;
    panelKeyList.querySelectorAll('input[type="checkbox"]').forEach(function (el) {
      checkedNow[el.value] = el.checked;
      haveUiState = true;
    });
    const preferred = haveUiState
      ? Object.keys(checkedNow).filter(function (k) { return checkedNow[k]; })
      : loadSavedCaptionKeys();
    const preferredSet = {};
    preferred.forEach(function (k) { preferredSet[k] = true; });

    // Selected first (saved/current order), then the rest (catalog order).
    const selected = [];
    preferred.forEach(function (k) {
      if (available.indexOf(k) >= 0) selected.push(k);
    });
    const rest = available.filter(function (k) { return !preferredSet[k]; });
    const ordered = selected.concat(rest);

    panelKeyList.innerHTML = '';
    ordered.forEach(function (key) {
      const lab = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = key;
      if (haveUiState && Object.prototype.hasOwnProperty.call(checkedNow, key)) {
        cb.checked = checkedNow[key];
      } else {
        cb.checked = !!preferredSet[key];
      }
      cb.addEventListener('change', function () {
        const current = selectedCaptionKeys(true);
        const prev = loadSavedCaptionKeys();
        const kept = prev.filter(function (k) { return available.indexOf(k) < 0; });
        saveCaptionKeys(current.concat(kept.filter(function (k) { return current.indexOf(k) < 0; })));
        renderPanelKeys();
      });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(key));
      panelKeyList.appendChild(lab);
    });
    if (panelNoCaptions && !haveUiState) {
      panelNoCaptions.checked = loadSavedNoCaptions();
    }
    syncPanelCaptionEnabled();
  }

  function syncPanelCaptionEnabled() {
    const off = !!(panelNoCaptions && panelNoCaptions.checked);
    if (panelKeyList) panelKeyList.style.opacity = off ? '0.4' : '1';
    if (panelKeyList) {
      panelKeyList.querySelectorAll('input').forEach(function (el) { el.disabled = off; });
    }
  }

  function selectedCaptionKeys(includeWhenNoCaptions) {
    if (!includeWhenNoCaptions && panelNoCaptions && panelNoCaptions.checked) return [];
    const keys = [];
    if (!panelKeyList) return keys;
    panelKeyList.querySelectorAll('input[type="checkbox"]').forEach(function (el) {
      if (el.checked) keys.push(el.value);
    });
    return keys;
  }

  function autoPanelRows() {
    if (!panelRows || !panelCols) return;
    const n = panelSelectedIds.length;
    const cols = Math.max(1, parseInt(panelCols.value, 10) || 3);
    panelRows.value = String(Math.max(1, Math.ceil(n / cols) || 1));
  }

  function renderPanelSelected() {
    if (!panelSelectedList) return;
    panelSelectedList.innerHTML = '';
    panelSelectedIds.forEach(function (id, idx) {
      const item = panelItemById(id);
      const li = document.createElement('li');
      const label = document.createElement('span');
      label.textContent = (idx + 1) + '. ' + (item ? (item.tag || item.name) : id);
      label.title = id;
      const up = document.createElement('button');
      up.type = 'button';
      up.textContent = '↑';
      up.title = 'Move up';
      up.addEventListener('click', function () {
        if (idx === 0) return;
        const tmp = panelSelectedIds[idx - 1];
        panelSelectedIds[idx - 1] = panelSelectedIds[idx];
        panelSelectedIds[idx] = tmp;
        renderPanelSelected();
        syncPanelPickSelection();
      });
      const down = document.createElement('button');
      down.type = 'button';
      down.textContent = '↓';
      down.title = 'Move down';
      down.addEventListener('click', function () {
        if (idx >= panelSelectedIds.length - 1) return;
        const tmp = panelSelectedIds[idx + 1];
        panelSelectedIds[idx + 1] = panelSelectedIds[idx];
        panelSelectedIds[idx] = tmp;
        renderPanelSelected();
        syncPanelPickSelection();
      });
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.textContent = '✕';
      rm.title = 'Remove';
      rm.addEventListener('click', function () {
        panelSelectedIds = panelSelectedIds.filter(function (x) { return x !== id; });
        autoPanelRows();
        renderPanelSelected();
        syncPanelPickSelection();
      });
      li.appendChild(label);
      li.appendChild(up);
      li.appendChild(down);
      li.appendChild(rm);
      panelSelectedList.appendChild(li);
    });
  }

  function syncPanelPickSelection() {
    if (!panelPickGrid) return;
    panelPickGrid.querySelectorAll('.panel-pick').forEach(function (btn) {
      const id = btn.getAttribute('data-id');
      btn.classList.toggle('selected', panelSelectedIds.indexOf(id) >= 0);
    });
  }

  function togglePanelItem(id) {
    const i = panelSelectedIds.indexOf(id);
    if (i >= 0) panelSelectedIds.splice(i, 1);
    else panelSelectedIds.push(id);
    autoPanelRows();
    renderPanelSelected();
    syncPanelPickSelection();
  }

  function bindPanelThumbs() {
    if (!panelPickGrid) return;
    if (panelThumbObserver) {
      panelThumbObserver.disconnect();
      panelThumbObserver = null;
    }
    panelThumbObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        const img = en.target;
        const src = img.getAttribute('data-src');
        if (src) {
          img.src = src;
          img.removeAttribute('data-src');
        }
        panelThumbObserver.unobserve(img);
      });
    }, { root: panelPickGrid, rootMargin: '120px', threshold: 0.01 });
    panelPickGrid.querySelectorAll('img[data-src]').forEach(function (img) {
      panelThumbObserver.observe(img);
    });
    // First rows: don't wait on observer (overflow roots can miss the initial intersection).
    requestAnimationFrame(function () {
      const imgs = panelPickGrid.querySelectorAll('img[data-src]');
      let n = 0;
      imgs.forEach(function (img) {
        if (n >= 24) return;
        const src = img.getAttribute('data-src');
        if (!src) return;
        img.src = src;
        img.removeAttribute('data-src');
        if (panelThumbObserver) panelThumbObserver.unobserve(img);
        n += 1;
      });
    });
  }

  function renderPanelPicker() {
    if (!panelPickGrid) return;
    const q = (panelSearch && panelSearch.value ? panelSearch.value : '').toLowerCase().trim();
    const items = panelCatalog.items || [];
    panelPickGrid.innerHTML = '';
    let shown = 0;
    items.forEach(function (item) {
      const hay = ((item.name || '') + ' ' + (item.tag || '') + ' ' + (item.model_type || '') + ' ' + (item.rel || '')).toLowerCase();
      if (q && hay.indexOf(q) < 0) return;
      shown += 1;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'panel-pick' + (panelSelectedIds.indexOf(item.id) >= 0 ? ' selected' : '');
      btn.setAttribute('data-id', item.id);
      const imgEl = document.createElement('img');
      imgEl.alt = item.name || '';
      imgEl.src = THUMB_PLACEHOLDER;
      imgEl.setAttribute('data-src', panelThumbUrl(item.id));
      imgEl.addEventListener('error', function () {
        if (imgEl.getAttribute('data-src')) return;
        imgEl.style.background = '#2a2a32';
        imgEl.alt = '(no preview)';
      });
      const meta = document.createElement('div');
      meta.className = 'panel-pick-meta';
      const strong = document.createElement('strong');
      strong.textContent = item.tag || item.name;
      meta.appendChild(strong);
      meta.appendChild(document.createTextNode(item.model_type || (item.has_yaml ? 'yaml' : item.source)));
      btn.appendChild(imgEl);
      btn.appendChild(meta);
      btn.addEventListener('click', function () { togglePanelItem(item.id); });
      panelPickGrid.appendChild(btn);
    });
    if (!shown) {
      const empty = document.createElement('div');
      empty.style.color = 'var(--text-muted)';
      empty.style.fontSize = '0.8rem';
      empty.textContent = 'No PNG/JPG images in this folder.';
      panelPickGrid.appendChild(empty);
    }
    bindPanelThumbs();
  }

  async function loadPanelFolders() {
    if (!panelFolderSelect) return;
    const r = await fetch(API_BASE + '/api/panel/folders');
    if (!r.ok) {
      const err = await r.json().catch(function () { return { detail: r.statusText }; });
      throw new Error(err.detail || r.statusText);
    }
    const data = await r.json();
    const folders = data.folders || [];
    const current = panelFolderSelect.value || 'results';
    panelFolderSelect.innerHTML = '';
    folders.forEach(function (f) {
      panelFolderSelect.appendChild(new Option(f.label || f.id, f.id));
    });
    if (!folders.length) {
      panelFolderSelect.appendChild(new Option('No folders with PNGs', ''));
      return;
    }
    const ids = folders.map(function (f) { return f.id; });
    panelFolderSelect.value = ids.indexOf(current) >= 0 ? current : folders[0].id;
  }

  async function loadPanelCatalog(forcedCustomPath) {
    const customRaw = (forcedCustomPath != null)
      ? String(forcedCustomPath)
      : (panelUseCustomPath && panelCustomPath ? panelCustomPath.value : '');
    const customPath = normalizeLocalPath(customRaw);
    let r;
    if (customPath) {
      panelUseCustomPath = true;
      try { localStorage.setItem(CUSTOM_PATH_STORAGE, customPath); } catch (e) { /* ignore */ }
      if (panelCustomPath && panelCustomPath.value.trim() !== customPath) {
        panelCustomPath.value = customPath;
      }
      // Mark dropdown so UI reflects custom load (without firing change→clear)
      if (panelFolderSelect) {
        let opt = null;
        for (let i = 0; i < panelFolderSelect.options.length; i++) {
          if (panelFolderSelect.options[i].value === '__custom__') {
            opt = panelFolderSelect.options[i];
            break;
          }
        }
        const label = 'custom: ' + customPath;
        if (!opt) {
          opt = new Option(label, '__custom__');
          panelFolderSelect.insertBefore(opt, panelFolderSelect.firstChild);
        } else {
          opt.text = label;
        }
        panelFolderSelect.value = '__custom__';
      }
      r = await fetch(API_BASE + '/api/panel/list_path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: customPath })
      });
    } else {
      const folder = (panelFolderSelect && panelFolderSelect.value) || 'results';
      if (folder === '__custom__') {
        throw new Error('Paste a folder path and press Load');
      }
      r = await fetch(API_BASE + '/api/panel/list?folder=' + encodeURIComponent(folder));
    }
    if (!r.ok) {
      const err = await r.json().catch(function () { return { detail: r.statusText }; });
      throw new Error(err.detail || r.statusText);
    }
    panelCatalog = await r.json();
    (panelCatalog.items || []).forEach(function (item) {
      panelItemCache[item.id] = item;
    });
    renderPanelKeys();
    renderPanelPicker();
    renderPanelSelected();
  }

  function normalizeLocalPath(raw) {
    let s = String(raw || '').trim();
    if ((s.charAt(0) === '"' && s.charAt(s.length - 1) === '"') ||
        (s.charAt(0) === "'" && s.charAt(s.length - 1) === "'")) {
      s = s.slice(1, -1).trim();
    }
    return s;
  }

  async function openComposePanel() {
    if (!composePanelModal) return;
    composePanelModal.classList.add('open');
    setStatus('Loading folders…', 'busy');
    try {
      if (panelCustomPath) {
        try {
          const saved = localStorage.getItem(CUSTOM_PATH_STORAGE);
          if (saved && !panelCustomPath.value) panelCustomPath.value = saved;
        } catch (e) { /* ignore */ }
      }
      await loadPanelFolders();
      await fetchOutputPaths();
      // If a custom path is already in the field, load it (not the dropdown default)
      const pending = normalizeLocalPath(panelCustomPath && panelCustomPath.value);
      if (pending && panelUseCustomPath) {
        await loadPanelCatalog(pending);
      } else {
        panelUseCustomPath = false;
        await loadPanelCatalog();
      }
      const where = panelCatalog.path || panelCatalog.folder || 'folder';
      setStatus('Folder: ' + ((panelCatalog.items || []).length) + ' images (' + where + '). Click to add to the panel.');
    } catch (e) {
      setStatus('Panel list failed: ' + e.message, 'error');
    }
  }

  async function composePanel(save) {
    if (!panelSelectedIds.length) {
      setStatus('Select at least one visualization.', 'error');
      return;
    }
    const cols = Math.max(1, parseInt(panelCols && panelCols.value, 10) || 3);
    const rows = Math.max(1, parseInt(panelRows && panelRows.value, 10) || 1);
    const cellMax = Math.max(64, parseInt(panelCellMax && panelCellMax.value, 10) || 480);
    setStatus(save ? 'Saving panel…' : 'Building panel…', 'busy');
    showLoading(true);
    try {
      const r = await fetch(API_BASE + '/api/panel/compose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ids: panelSelectedIds,
          rows: rows,
          cols: cols,
          caption_keys: selectedCaptionKeys(),
          cell_max: cellMax,
          save: !!save,
          output_path: (save && panelOutputPath && panelOutputPath.value.trim())
            ? panelOutputPath.value.trim()
            : null
        })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      if (panelPreviewImg && data.image_base64) {
        panelPreviewImg.src = 'data:image/png;base64,' + data.image_base64;
      }
      let msg = 'Panel ' + (save ? 'saved' : 'preview ready') + ' (' + data.n + ' cells).';
      if (data.truncated) msg += ' Truncated ' + data.truncated + ' extra image(s) to fit rows×cols.';
      if (data.saved_path) msg += ' ' + data.saved_path;
      setStatus(msg);
    } catch (e) {
      setStatus('Compose failed: ' + e.message, 'error');
    } finally {
      showLoading(false);
    }
  }

  if (btnComposePanel) btnComposePanel.addEventListener('click', openComposePanel);
  if (btnCloseComposePanel) btnCloseComposePanel.addEventListener('click', closeComposePanel);
  if (composePanelModal) {
    composePanelModal.addEventListener('click', function (e) {
      if (e.target === composePanelModal) closeComposePanel();
    });
  }
  if (panelFolderSelect) {
    panelFolderSelect.addEventListener('change', function () {
      if (panelFolderSelect.value === '__custom__') {
        // Keep custom mode; re-load from the path field
        loadCustomPanelPath();
        return;
      }
      panelUseCustomPath = false;
      setStatus('Loading folder…', 'busy');
      loadPanelCatalog().then(function () {
        setStatus('Folder: ' + ((panelCatalog.items || []).length) + ' images.');
      }).catch(function (e) {
        setStatus('Folder load failed: ' + e.message, 'error');
      });
    });
  }
  async function loadCustomPanelPath() {
    if (!panelCustomPath) {
      setStatus('Custom path field missing — hard-refresh the page.', 'error');
      return;
    }
    const path = normalizeLocalPath(panelCustomPath.value);
    if (!path) {
      setStatus('Paste a folder path first.', 'error');
      return;
    }
    panelCustomPath.value = path;
    panelUseCustomPath = true;
    setStatus('Loading custom path…', 'busy');
    try {
      await loadPanelCatalog(path);
      const where = panelCatalog.path || path;
      const n = (panelCatalog.items || []).length;
      setStatus('Custom folder: ' + n + ' images — ' + where);
      if (n === 0) {
        setStatus('No PNG/JPG images found under ' + where + ' (checked subfolders too).', 'error');
      }
    } catch (e) {
      setStatus('Custom path failed: ' + e.message, 'error');
    }
  }
  if (btnPanelLoadPath) btnPanelLoadPath.addEventListener('click', loadCustomPanelPath);
  if (panelCustomPath) {
    panelCustomPath.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        loadCustomPanelPath();
      }
    });
  }
  if (panelSearch) panelSearch.addEventListener('input', renderPanelPicker);
  if (panelCols) panelCols.addEventListener('change', autoPanelRows);
  if (panelNoCaptions) {
    panelNoCaptions.addEventListener('change', function () {
      saveNoCaptions(!!panelNoCaptions.checked);
      syncPanelCaptionEnabled();
    });
  }
  if (btnPanelPreview) btnPanelPreview.addEventListener('click', function () { composePanel(false); });
  if (btnPanelSave) btnPanelSave.addEventListener('click', function () { composePanel(true); });

  let sweepPollTimer = null;
  let sweepSchema = null;
  let loadedSweepConfig = null;

  function closeSweepModal() {
    if (sweepModal) sweepModal.classList.remove('open');
  }

  function expandSweepSpec(raw, kind) {
    const s = String(raw || '').trim();
    if (!s) return [];
    if ((kind === 'int' || kind === 'float') && s.indexOf(':') >= 0 && s.indexOf(',') < 0) {
      const parts = s.split(':').map(function (p) { return p.trim(); }).filter(Boolean);
      if (parts.length < 2 || parts.length > 3) return [];
      let start = parseFloat(parts[0]);
      let stop = parseFloat(parts[1]);
      let step = parts.length === 3 ? parseFloat(parts[2]) : 1;
      if (!isFinite(start) || !isFinite(stop) || !isFinite(step) || step === 0) return [];
      if (stop < start && step > 0) step = -step;
      if (stop > start && step < 0) step = -step;
      const out = [];
      if (kind === 'int') {
        start = Math.trunc(start); stop = Math.trunc(stop); step = Math.trunc(step) || 1;
        let x = start;
        if (step > 0) {
          while (x <= stop) { out.push(x); x += step; if (out.length > 500) break; }
        } else {
          while (x >= stop) { out.push(x); x += step; if (out.length > 500) break; }
        }
      } else {
        const eps = Math.abs(step) * 1e-9 + 1e-12;
        let x = start;
        if (step > 0) {
          while (x <= stop + eps) { out.push(Number(x.toPrecision(10))); x += step; if (out.length > 500) break; }
        } else {
          while (x >= stop - eps) { out.push(Number(x.toPrecision(10))); x += step; if (out.length > 500) break; }
        }
      }
      return out;
    }
    return s.split(',').map(function (p) { return p.trim(); }).filter(Boolean);
  }

  function countSweepProduct(params, kinds) {
    let n = 1;
    Object.keys(params).forEach(function (k) {
      const kind = (kinds && kinds[k]) || 'str';
      const parts = expandSweepSpec(params[k], kind);
      n *= Math.max(1, parts.length);
    });
    return n;
  }

  function collectSweepParams() {
    const params = {};
    const kinds = {};
    if (!sweepFields) return { params: params, kinds: kinds };
    sweepFields.querySelectorAll('.sweep-row').forEach(function (row) {
      const cb = row.querySelector('input[type="checkbox"]');
      if (!cb || !cb.checked) return;
      const key = cb.getAttribute('data-key');
      const kind = cb.getAttribute('data-kind') || 'str';
      if (!key) return;
      const modeSel = row.querySelector('select.sweep-mode');
      const mode = modeSel ? modeSel.value : 'list';
      let val = '';
      if (mode === 'range') {
        const fromEl = row.querySelector('[data-range="from"]');
        const toEl = row.querySelector('[data-range="to"]');
        const stepEl = row.querySelector('[data-range="step"]');
        const a = fromEl && fromEl.value.trim();
        const b = toEl && toEl.value.trim();
        const c = stepEl && stepEl.value.trim();
        if (!a || !b) return;
        val = c ? (a + ':' + b + ':' + c) : (a + ':' + b);
      } else {
        const inp = row.querySelector('input.sweep-list');
        val = inp ? (inp.value || '').trim() : '';
      }
      if (!val) return;
      params[key] = val;
      kinds[key] = kind;
    });
    return { params: params, kinds: kinds };
  }

  function updateSweepPreview() {
    const collected = collectSweepParams();
    const params = collected.params;
    const n = Object.keys(params).length ? countSweepProduct(params, collected.kinds) : 0;
    const maxR = Math.max(1, parseInt(sweepMaxRuns && sweepMaxRuns.value, 10) || 32);
    const lines = Object.keys(params).map(function (k) {
      const expanded = expandSweepSpec(params[k], collected.kinds[k] || 'str');
      return 'model.' + k + '=' + params[k] + '  →  [' + expanded.join(', ') + ']';
    });
    if (sweepOverridePreview) {
      let text = lines.length
        ? ('Overrides (' + n + ' runs):\n' + lines.join('\n'))
        : 'Enable a field: use list (a,b,c) or range from / to / step.';
      if (n > maxR) {
        text += '\n\nToo many runs (' + n + ' > max ' + maxR + '). Raise Max runs or shrink ranges.';
      }
      sweepOverridePreview.textContent = text;
    }
    if (btnSweepStart) {
      const canRun = !!(sweepSchema && (sweepSchema.has_image || sweepSchema.can_load));
      btnSweepStart.disabled = !canRun || !n || n > maxR;
    }
  }

  function paramsFromLoadedConfig(cfg) {
    const params = {};
    const kinds = {};
    if (!cfg || !cfg.fields) return { params: params, kinds: kinds };
    cfg.fields.forEach(function (f) {
      if (!f || !f.enabled || !f.key) return;
      const kind = f.kind || 'str';
      let val = '';
      if (f.mode === 'range') {
        const a = f['from'] != null ? String(f['from']).trim() : '';
        const b = f['to'] != null ? String(f['to']).trim() : '';
        const c = f['step'] != null ? String(f['step']).trim() : '';
        if (!a || !b) return;
        val = c ? (a + ':' + b + ':' + c) : (a + ':' + b);
      } else {
        val = f.values != null ? String(f.values).trim() : '';
      }
      if (!val) return;
      params[f.key] = val;
      kinds[f.key] = kind;
    });
    return { params: params, kinds: kinds };
  }

  function ensureMaxRunsForProduct(n) {
    if (!sweepMaxRuns || !n) return;
    const cur = Math.max(1, parseInt(sweepMaxRuns.value, 10) || 32);
    if (n > cur) {
      const bumped = Math.min(5000, n);
      sweepMaxRuns.value = String(bumped);
    }
  }

  const CLUSTER_SWEEP_POOL_DEFAULT = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

  function parseClusterSweepPool() {
    const raw = clusterSweepPool ? clusterSweepPool.value.trim() : '';
    if (!raw) return CLUSTER_SWEEP_POOL_DEFAULT.slice();
    const nums = raw.split(/[,;\s]+/).map(function (s) {
      return parseInt(s.trim(), 10);
    }).filter(function (n) {
      return Number.isFinite(n) && n > 0;
    });
    // unique + ascending
    const seen = {};
    const out = [];
    nums.sort(function (a, b) { return a - b; }).forEach(function (n) {
      if (!seen[n]) {
        seen[n] = true;
        out.push(n);
      }
    });
    return out.length ? out : CLUSTER_SWEEP_POOL_DEFAULT.slice();
  }

  function chooseCombinations(arr, k) {
    const out = [];
    function rec(start, path) {
      if (path.length === k) {
        out.push(path.slice());
        return;
      }
      for (let i = start; i < arr.length; i++) {
        path.push(arr[i]);
        rec(i + 1, path);
        path.pop();
      }
    }
    rec(0, []);
    return out;
  }

  function buildClusterSweepLadders() {
    const pool = parseClusterSweepPool();
    let minK = Math.max(2, parseInt(clusterSweepMinK && clusterSweepMinK.value, 10) || 2);
    let maxK = Math.max(2, parseInt(clusterSweepMaxK && clusterSweepMaxK.value, 10) || 5);
    if (maxK > 5) maxK = 5;
    if (minK > maxK) {
      const t = minK;
      minK = maxK;
      maxK = t;
    }
    maxK = Math.min(maxK, pool.length);
    minK = Math.min(minK, maxK);
    const consecutive = !!(clusterSweepConsecutive && clusterSweepConsecutive.checked);
    const ladders = [];
    for (let k = minK; k <= maxK; k++) {
      if (consecutive) {
        for (let i = 0; i + k <= pool.length; i++) {
          ladders.push(pool.slice(i, i + k).join('-'));
        }
      } else {
        chooseCombinations(pool, k).forEach(function (c) {
          ladders.push(c.join('-'));
        });
      }
    }
    return { ladders: ladders, pool: pool, minK: minK, maxK: maxK, consecutive: consecutive };
  }

  function updateClusterSweepPreview() {
    if (!clusterSweepPreview) return;
    const enabled = !!(clusterSweepEnable && clusterSweepEnable.checked);
    if (!enabled) {
      clusterSweepPreview.textContent = 'Enable to auto-generate cluster ladders (e.g. 4-8-16). Other sweep fields stay optional.';
      return;
    }
    const built = buildClusterSweepLadders();
    const n = built.ladders.length;
    const samples = built.ladders.slice(0, 6).join(', ') + (n > 6 ? ', …' : '');
    const mode = built.consecutive ? 'consecutive' : 'all combinations';
    clusterSweepPreview.textContent =
      n + ' ladders (' + mode + ', levels ' + built.minK + '–' + built.maxK +
      ', pool ' + built.pool.length + '): ' + samples;
  }

  function ensureClustersRowFilled(ladders) {
    /** Enable/fill clusters only — does not clear other sweep checkboxes. */
    if (!ladders || !ladders.length || !sweepFields) return false;
    let row = null;
    sweepFields.querySelectorAll('.sweep-row').forEach(function (r) {
      const cb = r.querySelector('input[type="checkbox"]');
      if (cb && cb.getAttribute('data-key') === 'clusters') row = r;
    });
    if (!row) return false;
    const cb = row.querySelector('input[type="checkbox"]');
    const modeSel = row.querySelector('select.sweep-mode');
    const listInp = row.querySelector('input.sweep-list');
    if (cb) cb.checked = true;
    if (modeSel) {
      modeSel.value = 'list';
      syncSweepRowMode(row);
    }
    if (listInp) listInp.value = ladders.join(',');
    return true;
  }

  function applyClusterSweepToFields() {
    if (!(clusterSweepEnable && clusterSweepEnable.checked)) {
      setStatus('Enable Cluster sweep first.', 'error');
      return false;
    }
    const built = buildClusterSweepLadders();
    if (!built.ladders.length) {
      setStatus('Cluster sweep produced no ladders — check pool / levels.', 'error');
      return false;
    }
    let row = null;
    if (sweepFields) {
      sweepFields.querySelectorAll('.sweep-row').forEach(function (r) {
        const cb = r.querySelector('input[type="checkbox"]');
        if (cb && cb.getAttribute('data-key') === 'clusters') row = r;
      });
    }
    if (!row) {
      setStatus('No clusters field for this model_type (need mics_lmc).', 'error');
      return false;
    }
    // Fill button: leave other params at model defaults unless re-enabled
    if (sweepFields) {
      sweepFields.querySelectorAll('.sweep-row').forEach(function (r) {
        const otherCb = r.querySelector('input[type="checkbox"]');
        if (otherCb && otherCb.getAttribute('data-key') !== 'clusters') {
          otherCb.checked = false;
        }
      });
    }
    if (!ensureClustersRowFilled(built.ladders)) return false;
    ensureMaxRunsForProduct(built.ladders.length);
    updateClusterSweepPreview();
    updateSweepPreview();
    setStatus(
      'Cluster sweep: ' + built.ladders.length +
      ' ladders → clusters (other sweep fields cleared; re-enable to combine).'
    );
    return true;
  }

  function syncSweepRowMode(row) {
    const modeSel = row.querySelector('select.sweep-mode');
    const listWrap = row.querySelector('.sweep-list-wrap');
    const rangeWrap = row.querySelector('.sweep-range');
    if (!modeSel) return;
    const isRange = modeSel.value === 'range';
    if (listWrap) listWrap.style.display = isRange ? 'none' : '';
    if (rangeWrap) rangeWrap.style.display = isRange ? 'grid' : 'none';
  }

  function renderSweepFields(schema) {
    if (!sweepFields) return;
    sweepFields.innerHTML = '';
    const fields = (schema.fields || []).slice().sort(function (a, b) {
      return String(a.key || '').localeCompare(String(b.key || ''));
    });
    if (!fields.length) {
      sweepFields.textContent = 'No sweepable parameters for model_type=' + (schema.model_type || '?');
      return;
    }
    fields.forEach(function (f) {
      const row = document.createElement('div');
      row.className = 'sweep-row';
      const numeric = f.kind === 'int' || f.kind === 'float';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.setAttribute('data-key', f.key);
      cb.setAttribute('data-kind', f.kind || 'str');
      cb.checked = false;

      const name = document.createElement('span');
      name.textContent = f.key;
      name.title = (f.kind || '') + (f.hint ? (' — ' + f.hint) : '');

      const modeSel = document.createElement('select');
      modeSel.className = 'sweep-mode';
      modeSel.appendChild(new Option('list', 'list'));
      if (numeric) modeSel.appendChild(new Option('range', 'range'));
      modeSel.value = numeric ? 'range' : 'list';

      const valuesCell = document.createElement('div');

      const listWrap = document.createElement('div');
      listWrap.className = 'sweep-list-wrap';
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.className = 'sweep-list';
      inp.value = f.values || '';
      inp.placeholder = f.hint || 'a,b,c';
      listWrap.appendChild(inp);

      const rangeWrap = document.createElement('div');
      rangeWrap.className = 'sweep-range';
      [['from', 'from'], ['to', 'to'], ['step', 'step']].forEach(function (pair) {
        const lab = document.createElement('label');
        lab.textContent = pair[0];
        const num = document.createElement('input');
        num.type = 'number';
        num.setAttribute('data-range', pair[1]);
        num.step = f.kind === 'float' ? 'any' : '1';
        if (pair[1] === 'step') num.value = f.kind === 'float' ? '0.1' : '1';
        else if (f.current != null && f.current !== '' && pair[1] === 'from') num.value = String(f.current);
        lab.appendChild(num);
        rangeWrap.appendChild(lab);
        num.addEventListener('input', updateSweepPreview);
      });

      // Seed range to/from from hint like "20:50:10" if present
      if (numeric && f.hint && f.hint.indexOf(':') >= 0) {
        const hintRange = f.hint.split(' or ')[0].trim();
        const hp = hintRange.split(':');
        if (hp.length >= 2) {
          const fromEl = rangeWrap.querySelector('[data-range="from"]');
          const toEl = rangeWrap.querySelector('[data-range="to"]');
          const stepEl = rangeWrap.querySelector('[data-range="step"]');
          if (fromEl && !fromEl.value) fromEl.value = hp[0];
          if (toEl) toEl.value = hp[1];
          if (stepEl && hp[2]) stepEl.value = hp[2];
        }
      }

      valuesCell.appendChild(listWrap);
      valuesCell.appendChild(rangeWrap);

      cb.addEventListener('change', updateSweepPreview);
      modeSel.addEventListener('change', function () {
        syncSweepRowMode(row);
        updateSweepPreview();
      });
      inp.addEventListener('input', updateSweepPreview);

      row.appendChild(cb);
      row.appendChild(name);
      row.appendChild(modeSel);
      row.appendChild(valuesCell);
      sweepFields.appendChild(row);
      syncSweepRowMode(row);
    });
    updateSweepPreview();
  }

  function snapshotSweepConfig() {
    const fields = [];
    if (sweepFields) {
      sweepFields.querySelectorAll('.sweep-row').forEach(function (row) {
        const cb = row.querySelector('input[type="checkbox"]');
        if (!cb) return;
        const key = cb.getAttribute('data-key');
        if (!key) return;
        const modeSel = row.querySelector('select.sweep-mode');
        const mode = modeSel ? modeSel.value : 'list';
        const entry = {
          key: key,
          kind: cb.getAttribute('data-kind') || 'str',
          enabled: !!cb.checked,
          mode: mode
        };
        if (mode === 'range') {
          const fromEl = row.querySelector('[data-range="from"]');
          const toEl = row.querySelector('[data-range="to"]');
          const stepEl = row.querySelector('[data-range="step"]');
          entry.from = fromEl ? fromEl.value.trim() : '';
          entry.to = toEl ? toEl.value.trim() : '';
          entry.step = stepEl ? stepEl.value.trim() : '';
        } else {
          const inp = row.querySelector('input.sweep-list');
          entry.values = inp ? (inp.value || '').trim() : '';
        }
        fields.push(entry);
      });
    }
    return {
      model_type: (sweepSchema && sweepSchema.model_type) || null,
      max_runs: Math.max(1, parseInt(sweepMaxRuns && sweepMaxRuns.value, 10) || 32),
      panel_cols: Math.max(1, parseInt(sweepPanelCols && sweepPanelCols.value, 10) || 4),
      build_panel: !!(sweepBuildPanel && sweepBuildPanel.checked),
      fields: fields
    };
  }

  function applySweepConfig(cfg) {
    if (!cfg || typeof cfg !== 'object') return;
    if (sweepMaxRuns && cfg.max_runs != null) sweepMaxRuns.value = String(cfg.max_runs);
    if (sweepPanelCols && cfg.panel_cols != null) sweepPanelCols.value = String(cfg.panel_cols);
    if (sweepBuildPanel && cfg.build_panel != null) sweepBuildPanel.checked = !!cfg.build_panel;
    const byKey = {};
    (cfg.fields || []).forEach(function (f) {
      if (f && f.key) byKey[f.key] = f;
    });
    if (!sweepFields) return;
    sweepFields.querySelectorAll('.sweep-row').forEach(function (row) {
      const cb = row.querySelector('input[type="checkbox"]');
      if (!cb) return;
      const key = cb.getAttribute('data-key');
      const saved = byKey[key];
      if (!saved) {
        cb.checked = false;
        return;
      }
      cb.checked = !!saved.enabled;
      const modeSel = row.querySelector('select.sweep-mode');
      const mode = (saved.mode === 'range' && modeSel && modeSel.querySelector('option[value="range"]'))
        ? 'range'
        : 'list';
      if (modeSel) modeSel.value = mode;
      syncSweepRowMode(row);
      if (mode === 'range') {
        const fromEl = row.querySelector('[data-range="from"]');
        const toEl = row.querySelector('[data-range="to"]');
        const stepEl = row.querySelector('[data-range="step"]');
        const fromVal = saved['from'];
        const toVal = saved['to'];
        const stepVal = saved['step'];
        if (fromEl && fromVal != null && fromVal !== '') fromEl.value = String(fromVal);
        if (toEl && toVal != null && toVal !== '') toEl.value = String(toVal);
        if (stepEl && stepVal != null && stepVal !== '') stepEl.value = String(stepVal);
        // Also accept values as "a:b:c"
        if ((!fromVal || !toVal) && saved.values && String(saved.values).indexOf(':') >= 0) {
          const hp = String(saved.values).split(':');
          if (fromEl && hp[0]) fromEl.value = hp[0];
          if (toEl && hp[1]) toEl.value = hp[1];
          if (stepEl && hp[2]) stepEl.value = hp[2];
        }
      } else {
        const inp = row.querySelector('input.sweep-list');
        if (inp) {
          if (saved.values != null) inp.value = String(saved.values);
          else if (saved['from'] != null && saved['to'] != null) {
            inp.value = saved['step']
              ? (saved['from'] + ':' + saved['to'] + ':' + saved['step'])
              : (saved['from'] + ':' + saved['to']);
          }
        }
      }
    });
    if (cfg.name && sweepConfigName) sweepConfigName.value = String(cfg.name);
    // Auto-raise Max runs so Start is not silently disabled after load
    const fromCfg = paramsFromLoadedConfig(cfg);
    const nCfg = Object.keys(fromCfg.params).length
      ? countSweepProduct(fromCfg.params, fromCfg.kinds)
      : 0;
    ensureMaxRunsForProduct(nCfg);
    updateSweepPreview();
  }

  async function refreshSweepConfigList(selectName) {
    if (!sweepConfigSelect) return;
    try {
      const r = await fetch(API_BASE + '/api/sweep/configs');
      if (!r.ok) return;
      const data = await r.json();
      const files = data.files || [];
      const keep = selectName || sweepConfigSelect.value || '';
      sweepConfigSelect.innerHTML = '';
      sweepConfigSelect.appendChild(new Option('— Saved sweeps —', ''));
      files.forEach(function (f) {
        sweepConfigSelect.appendChild(new Option(f, f));
      });
      if (keep) {
        const ids = files.slice();
        if (ids.indexOf(keep) >= 0) sweepConfigSelect.value = keep;
      }
    } catch (e) { /* ignore */ }
  }

  async function saveSweepConfig() {
    const snap = snapshotSweepConfig();
    const name = sweepConfigName ? sweepConfigName.value.trim() : '';
    setStatus('Saving sweep config…', 'busy');
    try {
      const r = await fetch(API_BASE + '/api/sweep/configs/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name || null,
          model_type: snap.model_type,
          max_runs: snap.max_runs,
          panel_cols: snap.panel_cols,
          build_panel: snap.build_panel,
          fields: snap.fields
        })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      if (sweepConfigName && data.filename) {
        sweepConfigName.value = String(data.filename).replace(/\.ya?ml$/i, '');
      }
      await refreshSweepConfigList(data.filename);
      loadedSweepConfig = snap;
      if (data.filename) loadedSweepConfig.name = String(data.filename).replace(/\.ya?ml$/i, '');
      setStatus('Sweep config saved: ' + (data.filename || data.saved_path));
    } catch (e) {
      setStatus('Save sweep failed: ' + e.message, 'error');
    }
  }

  async function fetchAndApplySweepConfig(filename, options) {
    options = options || {};
    if (!filename) return false;
    if (!options.quiet) setStatus('Loading sweep config…', 'busy');
    try {
      const r = await fetch(API_BASE + '/api/sweep/configs/load?filename=' + encodeURIComponent(filename));
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      const cfg = data.config || {};
      loadedSweepConfig = cfg;
      if (!cfg.name && data.filename) {
        loadedSweepConfig.name = String(data.filename).replace(/\.ya?ml$/i, '');
      }
      applySweepConfig(cfg);
      if (!options.quiet) {
        if (cfg.model_type && sweepSchema && sweepSchema.model_type &&
            cfg.model_type !== sweepSchema.model_type) {
          setStatus(
            'Loaded ' + filename + ' (saved for ' + cfg.model_type +
            '; current model is ' + sweepSchema.model_type + '). Matching fields applied.',
            'busy'
          );
        } else {
          setStatus('Sweep config loaded: ' + filename);
        }
      }
      return true;
    } catch (e) {
      if (!options.quiet) setStatus('Load sweep failed: ' + e.message, 'error');
      return false;
    }
  }

  async function loadSweepConfig() {
    const filename = sweepConfigSelect && sweepConfigSelect.value;
    if (!filename) {
      setStatus('Select a saved sweep first.', 'error');
      return;
    }
    await fetchAndApplySweepConfig(filename, { quiet: false });
  }

  async function deleteSweepConfig() {
    const filename = sweepConfigSelect && sweepConfigSelect.value;
    if (!filename) {
      setStatus('Select a saved sweep to delete.', 'error');
      return;
    }
    if (!window.confirm('Delete sweep config "' + filename + '"?')) return;
    try {
      const r = await fetch(API_BASE + '/api/sweep/configs/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename })
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      await refreshSweepConfigList('');
      setStatus('Deleted sweep config: ' + filename);
    } catch (e) {
      setStatus('Delete sweep failed: ' + e.message, 'error');
    }
  }

  async function populateSweepInputSelect() {
    if (!sweepInputSelect) return;
    // Prefer options already in the main sidebar select
    const keep = selectedInputPath || currentLoadedPath || (sweepSchema && sweepSchema.input_path) || '';
    if (inputFileSelect && inputFileSelect.options.length > 1) {
      sweepInputSelect.innerHTML = '';
      Array.from(inputFileSelect.options).forEach(function (opt) {
        sweepInputSelect.appendChild(new Option(opt.textContent, opt.value));
      });
    } else {
      try {
        const r = await fetch(API_BASE + '/api/input_files');
        if (r.ok) {
          const data = await r.json();
          const paths = data.paths || [];
          sweepInputSelect.innerHTML = '';
          if (!paths.length) {
            sweepInputSelect.appendChild(new Option('— no .npy files —', ''));
          } else {
            paths.forEach(function (p) {
              const label = (p.split(/[/\\]/).pop() || p);
              sweepInputSelect.appendChild(new Option(label, p));
            });
          }
          if (data.current && !keep) {
            sweepInputSelect.value = data.current;
          }
        }
      } catch (e) { /* ignore */ }
    }
    if (keep) {
      const found = Array.from(sweepInputSelect.options).some(function (o) { return o.value === keep; });
      if (!found && keep) {
        sweepInputSelect.appendChild(new Option(keep.split(/[/\\]/).pop() || keep, keep));
      }
      sweepInputSelect.value = keep;
    }
    if (sweepInputSelect.value) {
      selectedInputPath = sweepInputSelect.value;
      if (sweepSchema) sweepSchema.input_path = sweepInputSelect.value;
      if (sweepSchema && !sweepSchema.has_image) sweepSchema.can_load = true;
    }
  }

  async function openSweepModal() {
    if (!sweepModal) return;
    sweepModal.classList.add('open');
    if (sweepProgress) sweepProgress.textContent = '';
    // Keep current ticks/values across cancel → reopen (do not reload last saved config)
    const keepSnap = (sweepFields && sweepFields.querySelector('.sweep-row'))
      ? snapshotSweepConfig()
      : null;
    setStatus('Loading sweep schema…', 'busy');
    try {
      const pathHint = selectedInputPath || currentLoadedPath || '';
      const q = pathHint ? ('?input_path=' + encodeURIComponent(pathHint)) : '';
      const r = await fetch(API_BASE + '/api/sweep/schema' + q);
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      sweepSchema = await r.json();
      await populateSweepInputSelect();
      if (sweepMeta) {
        const imgStatus = sweepSchema.has_image
          ? 'image loaded'
          : (sweepSchema.can_load
            ? '<span style="color:var(--warn,#b8860b)">will auto-load on start</span>'
            : '<span style="color:var(--danger)">no image path — select a .npy below</span>');
        sweepMeta.innerHTML =
          '<strong>' + (sweepSchema.model_type || '') + '</strong> · ' +
          imgStatus +
          '<br/>' + (sweepSchema.hydra_note || '');
      }
      const pendingConfig = sweepConfigSelect && sweepConfigSelect.value;
      if (sweepMaxRuns && sweepSchema.max_runs_default && !keepSnap && !pendingConfig && !loadedSweepConfig) {
        sweepMaxRuns.value = String(sweepSchema.max_runs_default);
      }
      renderSweepFields(sweepSchema);
      await refreshSweepConfigList(pendingConfig || (loadedSweepConfig && loadedSweepConfig.name ? (loadedSweepConfig.name + '.yaml') : null));
      if (keepSnap && (!keepSnap.model_type || keepSnap.model_type === sweepSchema.model_type)) {
        applySweepConfig(keepSnap);
      } else if (!keepSnap && pendingConfig) {
        await fetchAndApplySweepConfig(pendingConfig, { quiet: true });
      } else if (!keepSnap && loadedSweepConfig) {
        applySweepConfig(loadedSweepConfig);
      } else {
        updateSweepPreview();
      }
      updateClusterSweepPreview();
      setStatus('Parameter sweep ready.');
    } catch (e) {
      setStatus('Sweep schema failed: ' + e.message, 'error');
    }
  }

  function stopSweepPoll() {
    if (sweepPollTimer) {
      clearInterval(sweepPollTimer);
      sweepPollTimer = null;
    }
  }

  async function pollSweepStatus() {
    try {
      const r = await fetch(API_BASE + '/api/sweep/status');
      if (!r.ok) return;
      const data = await r.json();
      const cur = data.current || 0;
      const tot = data.total || 0;
      const st = data.status || '';
      if (sweepProgress) {
        let msg = st + (tot ? ('  ' + cur + '/' + tot) : '');
        if (data.run_dir) msg += '  →  ' + data.run_dir;
        if (data.error) msg += '  error: ' + data.error;
        if (data.panel_path) msg += '  panel: ' + data.panel_path;
        sweepProgress.textContent = msg;
      }
      if (st === 'running') {
        setStatus('Sweep ' + cur + '/' + tot + '…', 'busy');
        showLoading(true);
      }
      if (st === 'done' || st === 'error' || st === 'cancelled') {
        stopSweepPoll();
        showLoading(false);
        if (st === 'done') {
          setStatus('Sweep done' + (data.run_dir ? (' (' + data.run_dir + ')') : '') + '.');
          if (data.image_base64) {
            await loadImageFromBase64(data.image_base64, true);
            await fetchConfig();
          }
        } else if (st === 'cancelled') {
          setStatus('Sweep cancelled.');
        } else {
          setStatus('Sweep failed: ' + (data.error || 'unknown'), 'error');
        }
        if (btnSweepStart) {
          btnSweepStart.disabled = false;
          updateSweepPreview();
        }
      }
    } catch (e) { /* ignore poll errors */ }
  }

  async function startSweep() {
    // Form is source of truth — do not reload saved YAML (that reset ticks after cancel)
    if (clusterSweepEnable && clusterSweepEnable.checked) {
      const built = buildClusterSweepLadders();
      if (!built.ladders.length) {
        setStatus('Cluster sweep produced no ladders — check pool / levels.', 'error');
        return;
      }
      // Refresh clusters values only; keep other enabled fields as the user left them
      if (!ensureClustersRowFilled(built.ladders)) {
        setStatus('No clusters field for this model_type (need mics_lmc).', 'error');
        return;
      }
      ensureMaxRunsForProduct(built.ladders.length);
      updateSweepPreview();
    }
    const collected = collectSweepParams();
    const params = collected.params;
    const kinds = collected.kinds;
    if (!Object.keys(params).length) {
      setStatus('Enable at least one sweep field (or Cluster sweep + Fill).', 'error');
      return;
    }
    const n = countSweepProduct(params, kinds);
    ensureMaxRunsForProduct(n);
    const inputPath = (sweepInputSelect && sweepInputSelect.value)
      || selectedInputPath
      || currentLoadedPath
      || (sweepSchema && sweepSchema.input_path)
      || null;
    if (!inputPath && !(sweepSchema && sweepSchema.has_image)) {
      setStatus('Select an input .npy file first.', 'error');
      return;
    }
    if (inputPath) selectedInputPath = inputPath;
    const body = {
      params: params,
      max_runs: Math.max(1, parseInt(sweepMaxRuns && sweepMaxRuns.value, 10) || 32),
      panel_cols: Math.max(1, parseInt(sweepPanelCols && sweepPanelCols.value, 10) || 4),
      build_panel: !!(sweepBuildPanel && sweepBuildPanel.checked),
      keep_last_as_current: true,
      input_path: inputPath
    };
    if (n > body.max_runs) {
      setStatus('Too many combinations (' + n + ') for max runs (' + body.max_runs + ').', 'error');
      updateSweepPreview();
      return;
    }
    setStatus('Starting sweep…', 'busy');
    showLoading(true);
    if (btnSweepStart) btnSweepStart.disabled = true;
    try {
      const r = await fetch(API_BASE + '/api/sweep/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!r.ok) {
        const err = await r.json().catch(function () { return { detail: r.statusText }; });
        throw new Error(err.detail || r.statusText);
      }
      const data = await r.json();
      if (sweepProgress) {
        sweepProgress.textContent = 'started ' + (data.n_runs || '?') + ' runs…';
      }
      stopSweepPoll();
      sweepPollTimer = setInterval(pollSweepStatus, 10000);
      pollSweepStatus();
    } catch (e) {
      showLoading(false);
      if (btnSweepStart) btnSweepStart.disabled = false;
      setStatus('Sweep start failed: ' + e.message, 'error');
      updateSweepPreview();
    }
  }

  async function cancelSweep() {
    try {
      await fetch(API_BASE + '/api/sweep/cancel', { method: 'POST' });
      setStatus('Cancel requested…', 'busy');
    } catch (e) {
      setStatus('Cancel failed: ' + e.message, 'error');
    }
  }

  if (btnParamSweep) btnParamSweep.addEventListener('click', openSweepModal);
  if (btnCloseSweep) btnCloseSweep.addEventListener('click', closeSweepModal);
  if (sweepModal) {
    sweepModal.addEventListener('click', function (e) {
      if (e.target === sweepModal) closeSweepModal();
    });
  }
  if (btnSweepStart) btnSweepStart.addEventListener('click', startSweep);
  if (btnSweepCancel) btnSweepCancel.addEventListener('click', cancelSweep);
  if (sweepMaxRuns) sweepMaxRuns.addEventListener('input', updateSweepPreview);
  if (btnClusterSweepApply) btnClusterSweepApply.addEventListener('click', applyClusterSweepToFields);
  function onClusterSweepUiChange() {
    updateClusterSweepPreview();
    if (clusterSweepEnable && clusterSweepEnable.checked) {
      // Live count only — do not auto-fill until Apply / Start (large lists)
    }
  }
  if (clusterSweepEnable) clusterSweepEnable.addEventListener('change', onClusterSweepUiChange);
  if (clusterSweepMinK) clusterSweepMinK.addEventListener('input', onClusterSweepUiChange);
  if (clusterSweepMaxK) clusterSweepMaxK.addEventListener('input', onClusterSweepUiChange);
  if (clusterSweepConsecutive) clusterSweepConsecutive.addEventListener('change', onClusterSweepUiChange);
  if (clusterSweepPool) clusterSweepPool.addEventListener('input', onClusterSweepUiChange);
  if (btnSweepConfigSave) btnSweepConfigSave.addEventListener('click', saveSweepConfig);
  if (btnSweepConfigLoad) btnSweepConfigLoad.addEventListener('click', loadSweepConfig);
  if (btnSweepConfigDelete) btnSweepConfigDelete.addEventListener('click', deleteSweepConfig);
  if (sweepConfigSelect) {
    sweepConfigSelect.addEventListener('change', function () {
      const filename = sweepConfigSelect.value;
      if (filename) fetchAndApplySweepConfig(filename, { quiet: false });
      else loadedSweepConfig = null;
    });
  }
  if (sweepInputSelect) {
    sweepInputSelect.addEventListener('change', function () {
      const path = sweepInputSelect.value || null;
      selectedInputPath = path;
      if (path && inputFileSelect) {
        const found = Array.from(inputFileSelect.options).some(function (o) { return o.value === path; });
        if (found) inputFileSelect.value = path;
      }
      if (sweepSchema) {
        sweepSchema.input_path = path;
        if (path) sweepSchema.can_load = true;
      }
      updateSweepPreview();
    });
  }

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
    const active = document.activeElement;
    const isTextInput = active && (
      active.tagName === 'INPUT' ||
      active.tagName === 'TEXTAREA' ||
      active.tagName === 'SELECT' ||
      active.isContentEditable
    );
    if (isTextInput) return;
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
  if (btnRankNmfRegion) btnRankNmfRegion.addEventListener('click', rankNmfForRegion);
  if (btnCloseNmfModal) btnCloseNmfModal.addEventListener('click', closeNmfModal);
  if (nmfRankModal) {
    nmfRankModal.addEventListener('click', function (e) {
      if (e.target === nmfRankModal) closeNmfModal();
    });
  }
  if (btnRotateLeft) btnRotateLeft.addEventListener('click', function () { rotateBy(-90); });
  if (btnRotateRight) btnRotateRight.addEventListener('click', function () { rotateBy(90); });
  if (btnResetView) btnResetView.addEventListener('click', resetView);
  if (btnFitView) btnFitView.addEventListener('click', fitToViewer);

  viewer.addEventListener('wheel', onWheel, { passive: false });
  viewer.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);
  window.addEventListener('keydown', onKeyDown);

  settingsPanel.classList.add('open');
  ensureModelTypeOptions();
  fetchConfig();
  fetchConfigFiles();
  fetchOutputPaths();
  showEmptyState(true);
  importConfigFile('default.yaml');
})();
