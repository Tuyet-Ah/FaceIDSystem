/**
 * Multi-Model Object & PC Case Counting Web Application
 * Supports YOLO11 (Bounding Box) & LOCA (Low-Shot Density Map)
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  let currentFile = null;
  let currentPreviewUrl = null;
  let annotatedDataUrl = null;
  let mediaStream = null;
  let isProcessing = false;
  let maxUploadBytes = 64 * 1024 * 1024;
  let maxUploadMb = 64;

  let currentModel = 'yolo11'; // 'yolo11' or 'loca'
  let locaMode = 'zero_shot';  // 'zero_shot' or 'few_shot'
  let exemplarBoxes = [];      // [[x1, y1, x2, y2], ...] in original image coordinates
  let clickSquareSize = 110;   // Default square box size in original pixels for 1-click

  // Mouse drawing state
  let isMouseDown = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let mouseDownTimestamp = 0;

  // Header & Status
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');
  const dropFormatsText = document.getElementById('drop-formats-text');

  // Input Tabs
  const tabUpload = document.getElementById('tab-upload');
  const tabWebcam = document.getElementById('tab-webcam');
  const tabSample = document.getElementById('tab-sample');
  const uploadSection = document.getElementById('upload-section');
  const webcamSection = document.getElementById('webcam-section');
  const sampleSection = document.getElementById('sample-section');

  // Upload & Preview
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const previewBox = document.getElementById('preview-box');
  const previewImage = document.getElementById('preview-image');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const btnRemoveImage = document.getElementById('btn-remove-image');

  // Model Selector
  const btnModelYolo = document.getElementById('btn-model-yolo');
  const btnModelLoca = document.getElementById('btn-model-loca');
  const btnModelDino = document.getElementById('btn-model-dino');
  const yoloConfigGroup = document.getElementById('yolo-config-group');
  const locaConfigGroup = document.getElementById('loca-config-group');
  const dinoConfigGroup = document.getElementById('dino-config-group');

  // YOLO Parameters
  const imgszSelect = document.getElementById('imgsz-select');
  const confidenceRange = document.getElementById('confidence-range');
  const confidenceVal = document.getElementById('confidence-val');
  const checkSliced = document.getElementById('check-sliced');
  const checkEnhance = document.getElementById('check-enhance');

  // LOCA Parameters
  const radioLocaZero = document.getElementById('radio-loca-zero');
  const radioLocaFew = document.getElementById('radio-loca-few');
  const locaFewShotHelper = document.getElementById('loca-few-shot-helper');
  const exemplarCountBadge = document.getElementById('exemplar-count-badge');
  const btnClearExemplars = document.getElementById('btn-clear-exemplars');
  const locaThresholdRange = document.getElementById('loca-threshold-range');
  const locaThresholdVal = document.getElementById('loca-threshold-val');
  const sizeChoiceButtons = document.querySelectorAll('.btn-size-choice');

  // Grounding DINO Parameters
  const dinoPromptInput = document.getElementById('dino-prompt-input');
  const dinoBoxThreshRange = document.getElementById('dino-box-thresh-range');
  const dinoBoxThreshVal = document.getElementById('dino-box-thresh-val');
  const dinoTextThreshRange = document.getElementById('dino-text-thresh-range');
  const dinoTextThreshVal = document.getElementById('dino-text-thresh-val');
  const promptPillButtons = document.querySelectorAll('.btn-prompt-pill');

  // Detect Action
  const btnDetect = document.getElementById('btn-detect');
  const detectSpinner = document.getElementById('detect-spinner');
  const detectBtnText = document.getElementById('detect-btn-text');

  // Interactive Stage & Results
  const emptyState = document.getElementById('empty-state');
  const resultContent = document.getElementById('result-content');
  const workspaceBanner = document.getElementById('workspace-banner');
  const bannerIcon = document.getElementById('banner-icon');
  const bannerText = document.getElementById('banner-text');
  const btnBannerClear = document.getElementById('btn-banner-clear');

  const stageWrapper = document.getElementById('stage-wrapper');
  const imageOriginal = document.getElementById('image-original');
  const imageAnnotated = document.getElementById('image-annotated');
  const mainDrawCanvas = document.getElementById('main-draw-canvas');

  const statsWrapper = document.getElementById('stats-wrapper');
  const resultActions = document.getElementById('result-actions');
  const statLabelCount = document.getElementById('stat-label-count');
  const statCount = document.getElementById('stat-count');
  const statTime = document.getElementById('stat-time');
  const statRes = document.getElementById('stat-res');
  const statLabelConf = document.getElementById('stat-label-conf');
  const statConf = document.getElementById('stat-conf');

  const tableTitleText = document.getElementById('table-title-text');
  const tableCount = document.getElementById('table-count');
  const thScore = document.getElementById('th-score');
  const thCoords = document.getElementById('th-coords');
  const detectionsTbody = document.getElementById('detections-tbody');
  const btnViewAnnotated = document.getElementById('btn-view-annotated');
  const btnViewOriginal = document.getElementById('btn-view-original');
  const btnDownload = document.getElementById('btn-download');
  const btnFullscreen = document.getElementById('btn-fullscreen');

  // Webcam Elements
  const cameraFeed = document.getElementById('camera-feed');
  const cameraOverlay = document.getElementById('camera-overlay');
  const cameraErrorText = document.getElementById('camera-error-text');
  const btnSnapCamera = document.getElementById('btn-snap-camera');
  const btnToggleCamera = document.getElementById('btn-toggle-camera');

  // Sample
  const btnLoadSample = document.getElementById('btn-load-sample');

  // Modal
  const modalZoom = document.getElementById('modal-zoom');
  const modalImg = document.getElementById('modal-img');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const modalBackdrop = document.getElementById('modal-backdrop');

  // Toast Container
  const toastContainer = document.getElementById('toast-container');

  // ----------------------------------------------------
  // Toast Notifications
  // ----------------------------------------------------
  function showToast(message, type = 'error', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' : 'toast-success'}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-fade-out');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ----------------------------------------------------
  // Health & Initial Setup
  // ----------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();

      if (data.max_upload_mb) {
        maxUploadMb = data.max_upload_mb;
        maxUploadBytes = maxUploadMb * 1024 * 1024;
        if (dropFormatsText) {
          dropFormatsText.textContent = `Hỗ trợ JPG, PNG, WEBP (Tối đa ${maxUploadMb} MB)`;
        }
      }

      if (imgszSelect && data.default_imgsz) {
        imgszSelect.value = String(data.default_imgsz);
      }

      if (data.ready) {
        statusPill.className = 'status-pill status-ready';
        let readyList = [];
        if (data.yolo_ready) readyList.push('YOLO11');
        if (data.loca_ready) readyList.push('LOCA');
        if (data.dino_ready) readyList.push('Grounding DINO');
        statusText.textContent = `Mô hình: ${readyList.join(' + ')} (Sẵn sàng)`;
      } else {
        statusPill.className = 'status-pill status-error';
        statusText.textContent = 'Chưa sẵn sàng bộ trọng số';
      }
    } catch (err) {
      statusPill.className = 'status-pill status-error';
      statusText.textContent = 'Mất kết nối server';
    }
  }

  // ----------------------------------------------------
  // Model & Mode Switching Logic
  // ----------------------------------------------------
  function setModel(model) {
    currentModel = model;

    [btnModelYolo, btnModelLoca, btnModelDino].forEach(b => b.classList.remove('active'));
    [yoloConfigGroup, locaConfigGroup, dinoConfigGroup].forEach(g => g.classList.add('hidden'));

    if (model === 'yolo11') {
      btnModelYolo.classList.add('active');
      yoloConfigGroup.classList.remove('hidden');
      detectBtnText.textContent = 'Nhận diện & Đếm bằng YOLO11';
    } else if (model === 'loca') {
      btnModelLoca.classList.add('active');
      locaConfigGroup.classList.remove('hidden');
      detectBtnText.textContent = 'Đếm mật độ bằng LOCA';
    } else if (model === 'grounding_dino') {
      btnModelDino.classList.add('active');
      dinoConfigGroup.classList.remove('hidden');
      detectBtnText.textContent = 'Nhận diện & Đếm bằng Grounding DINO';
    }

    updateLocaModeUI();
  }

  btnModelYolo.addEventListener('click', () => setModel('yolo11'));
  btnModelLoca.addEventListener('click', () => setModel('loca'));
  btnModelDino.addEventListener('click', () => setModel('grounding_dino'));

  // Grounding DINO Quick Prompt Pills
  promptPillButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      promptPillButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      dinoPromptInput.value = btn.dataset.prompt;
    });
  });

  dinoBoxThreshRange.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    const pct = Math.round(parseFloat(e.target.value) * 100);
    dinoBoxThreshVal.textContent = `${val} (${pct}%)`;
  });

  dinoTextThreshRange.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    const pct = Math.round(parseFloat(e.target.value) * 100);
    dinoTextThreshVal.textContent = `${val} (${pct}%)`;
  });

  function updateLocaModeUI() {
    if (currentModel === 'loca') {
      if (radioLocaFew.checked) {
        locaMode = 'few_shot';
        locaFewShotHelper.classList.remove('hidden');
      } else {
        locaMode = 'zero_shot';
        locaFewShotHelper.classList.add('hidden');
      }
    }

    updateWorkspaceBanner();
    syncMainCanvasSize();
    redrawMainCanvas();
  }

  radioLocaZero.addEventListener('change', updateLocaModeUI);
  radioLocaFew.addEventListener('change', updateLocaModeUI);

  // Square Size Buttons
  sizeChoiceButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      sizeChoiceButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      clickSquareSize = parseInt(btn.dataset.size, 10) || 110;
      showToast(`Đã chọn cỡ khung vuông: ${clickSquareSize}px`, 'success', 1800);
    });
  });

  // ----------------------------------------------------
  // Workspace Banner Updates
  // ----------------------------------------------------
  function updateWorkspaceBanner() {
    if (!currentFile) {
      workspaceBanner.classList.add('hidden');
      return;
    }

    workspaceBanner.classList.remove('hidden');

    if (currentModel === 'loca' && locaMode === 'few_shot') {
      bannerIcon.textContent = '🎯';
      bannerText.innerHTML = `<strong>Chế độ Khoanh Mẫu (Few-Shot):</strong> Click vào thùng máy để đặt khung vuông, hoặc kéo chuột để khoanh! (Đã chọn: <strong>${exemplarBoxes.length}/3 mẫu</strong>)`;
      btnBannerClear.classList.remove('hidden');
      mainDrawCanvas.classList.remove('drawing-disabled');
    } else if (currentModel === 'loca') {
      bannerIcon.textContent = '⚡';
      bannerText.innerHTML = `<strong>Chế độ Zero-Shot:</strong> LOCA sẽ tự động phát hiện mọi đối tượng lặp lại mà không cần khoanh mẫu.`;
      btnBannerClear.classList.add('hidden');
      mainDrawCanvas.classList.add('drawing-disabled');
    } else if (currentModel === 'grounding_dino') {
      bannerIcon.textContent = '🏷️';
      bannerText.innerHTML = `<strong>Chế độ Grounding DINO:</strong> Đếm đối tượng linh hoạt bằng câu chữ mô tả (Prompt). Nhấn nút bên trái để đếm.`;
      btnBannerClear.classList.add('hidden');
      mainDrawCanvas.classList.add('drawing-disabled');
    } else {
      bannerIcon.textContent = '🔍';
      bannerText.innerHTML = `<strong>Chế độ YOLO11:</strong> Quét tự động các thùng máy PC và vẽ Bounding Box có độ tự tin cao.`;
      btnBannerClear.classList.add('hidden');
      mainDrawCanvas.classList.add('drawing-disabled');
    }
  }

  function clearAllExemplars() {
    exemplarBoxes = [];
    updateExemplarBadge();
    updateWorkspaceBanner();
    redrawMainCanvas();
    showToast('Đã xóa tất cả các khung mẫu.', 'success', 2000);
  }

  btnClearExemplars.addEventListener('click', clearAllExemplars);
  btnBannerClear.addEventListener('click', clearAllExemplars);

  function updateExemplarBadge() {
    exemplarCountBadge.textContent = `${exemplarBoxes.length} / 3 mẫu`;
    if (exemplarBoxes.length >= 3) {
      exemplarCountBadge.style.backgroundColor = '#10b981';
      exemplarCountBadge.style.color = '#fff';
    } else {
      exemplarCountBadge.style.backgroundColor = '#f59e0b';
      exemplarCountBadge.style.color = '#000';
    }
  }

  // ----------------------------------------------------
  // Main Canvas Setup & Drawing
  // ----------------------------------------------------
  function syncMainCanvasSize() {
    const activeImg = imageOriginal.classList.contains('hidden') ? imageAnnotated : imageOriginal;
    if (!activeImg.complete || !activeImg.naturalWidth) return;

    mainDrawCanvas.width = activeImg.clientWidth;
    mainDrawCanvas.height = activeImg.clientHeight;
  }

  window.addEventListener('resize', () => {
    syncMainCanvasSize();
    redrawMainCanvas();
  });

  imageOriginal.addEventListener('load', () => {
    syncMainCanvasSize();
    redrawMainCanvas();
  });

  imageAnnotated.addEventListener('load', () => {
    syncMainCanvasSize();
    redrawMainCanvas();
  });

  function getCanvasPos(e) {
    const rect = mainDrawCanvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  function redrawMainCanvas(currentDragBox = null) {
    const ctx = mainDrawCanvas.getContext('2d');
    ctx.clearRect(0, 0, mainDrawCanvas.width, mainDrawCanvas.height);

    if (!imageOriginal.naturalWidth) return;
    const scaleX = mainDrawCanvas.width / imageOriginal.naturalWidth;
    const scaleY = mainDrawCanvas.height / imageOriginal.naturalHeight;

    // Draw existing exemplar boxes
    exemplarBoxes.forEach((box, idx) => {
      const bx = box[0] * scaleX;
      const by = box[1] * scaleY;
      const bw = (box[2] - box[0]) * scaleX;
      const bh = (box[3] - box[1]) * scaleY;

      // Fill semi-transparent orange
      ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
      ctx.fillRect(bx, by, bw, bh);

      // Border with solid glow
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#f59e0b';
      ctx.strokeRect(bx, by, bw, bh);

      // Center crosshair mark (+)
      const cx = bx + bw / 2;
      const cy = by + bh / 2;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
      ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
      ctx.stroke();

      // Corner markers
      ctx.fillStyle = '#ffffff';
      const csize = 6;
      ctx.fillRect(bx - 3, by - 3, csize, csize);
      ctx.fillRect(bx + bw - 3, by - 3, csize, csize);
      ctx.fillRect(bx - 3, by + bh - 3, csize, csize);
      ctx.fillRect(bx + bw - 3, by + bh - 3, csize, csize);

      // Label badge
      ctx.fillStyle = '#f59e0b';
      const labelText = `Mẫu #${idx + 1} (${Math.round(box[2] - box[0])}×${Math.round(box[3] - box[1])})`;
      ctx.font = 'bold 12px sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const labelY = Math.max(0, by - 24);
      ctx.fillRect(bx, labelY, textWidth + 14, 22);

      ctx.fillStyle = '#000000';
      ctx.fillText(labelText, bx + 7, labelY + 16);
    });

    // Draw active drag preview
    if (currentDragBox) {
      const rx = Math.min(currentDragBox.x1, currentDragBox.x2);
      const ry = Math.min(currentDragBox.y1, currentDragBox.y2);
      const rw = Math.abs(currentDragBox.x2 - currentDragBox.x1);
      const rh = Math.abs(currentDragBox.y2 - currentDragBox.y1);

      ctx.fillStyle = 'rgba(56, 189, 248, 0.3)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#38bdf8';
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
    }
  }

  // ----------------------------------------------------
  // Mouse Events: 1-Click Square & Freeform Drag
  // ----------------------------------------------------
  mainDrawCanvas.addEventListener('mousedown', e => {
    if (currentModel !== 'loca' || locaMode !== 'few_shot') return;

    if (exemplarBoxes.length >= 3) {
      showToast('Đã chọn tối đa 3 mẫu đối tượng. Hãy bấm "Xóa tất cả" nếu muốn chọn lại.', 'error', 3000);
      return;
    }

    const pos = getCanvasPos(e);
    isMouseDown = true;
    dragStartX = pos.x;
    dragStartY = pos.y;
    mouseDownTimestamp = Date.now();
  });

  mainDrawCanvas.addEventListener('mousemove', e => {
    if (!isMouseDown) return;
    const pos = getCanvasPos(e);
    redrawMainCanvas({
      x1: dragStartX,
      y1: dragStartY,
      x2: pos.x,
      y2: pos.y
    });
  });

  function finishCanvasInteraction(e) {
    if (!isMouseDown) return;
    isMouseDown = false;

    const pos = getCanvasPos(e);
    const timeDiff = Date.now() - mouseDownTimestamp;
    const dist = Math.hypot(pos.x - dragStartX, pos.y - dragStartY);

    const scaleX = imageOriginal.naturalWidth / mainDrawCanvas.width;
    const scaleY = imageOriginal.naturalHeight / mainDrawCanvas.height;

    // CASE 1: 1-CLICK SQUARE CREATION (Super easy for user!)
    if (timeDiff < 300 && dist < 10) {
      const clickOrigX = pos.x * scaleX;
      const clickOrigY = pos.y * scaleY;
      const halfSize = clickSquareSize / 2;

      const x1 = Math.max(0, Math.round(clickOrigX - halfSize));
      const y1 = Math.max(0, Math.round(clickOrigY - halfSize));
      const x2 = Math.min(imageOriginal.naturalWidth, Math.round(clickOrigX + halfSize));
      const y2 = Math.min(imageOriginal.naturalHeight, Math.round(clickOrigY + halfSize));

      exemplarBoxes.push([x1, y1, x2, y2]);
      updateExemplarBadge();
      updateWorkspaceBanner();
      redrawMainCanvas();
      showToast(`Đã tạo khung vuông Mẫu #${exemplarBoxes.length} (${clickSquareSize}×${clickSquareSize}px)!`, 'success', 2200);
      return;
    }

    // CASE 2: DRAG BOX CREATION
    const x1 = Math.round(Math.min(dragStartX, pos.x) * scaleX);
    const y1 = Math.round(Math.min(dragStartY, pos.y) * scaleY);
    const x2 = Math.round(Math.max(dragStartX, pos.x) * scaleX);
    const y2 = Math.round(Math.max(dragStartY, pos.y) * scaleY);

    if (x2 - x1 < 15 || y2 - y1 < 15) {
      redrawMainCanvas();
      return;
    }

    exemplarBoxes.push([x1, y1, x2, y2]);
    updateExemplarBadge();
    updateWorkspaceBanner();
    redrawMainCanvas();
    showToast(`Đã khoanh Mẫu #${exemplarBoxes.length} thành công!`, 'success', 2200);
  }

  mainDrawCanvas.addEventListener('mouseup', finishCanvasInteraction);
  mainDrawCanvas.addEventListener('mouseleave', () => {
    if (isMouseDown) {
      isMouseDown = false;
      redrawMainCanvas();
    }
  });

  // ----------------------------------------------------
  // Sliders Bindings
  // ----------------------------------------------------
  confidenceRange.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    const pct = Math.round(parseFloat(e.target.value) * 100);
    confidenceVal.textContent = `${val} (${pct}%)`;
  });

  locaThresholdRange.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    const pct = Math.round(parseFloat(e.target.value) * 100);
    locaThresholdVal.textContent = `${val} (${pct}%)`;
  });

  // ----------------------------------------------------
  // Tabs
  // ----------------------------------------------------
  function switchTab(target) {
    [tabUpload, tabWebcam, tabSample].forEach(tab => tab.classList.remove('active'));
    [uploadSection, webcamSection, sampleSection].forEach(sec => sec.classList.add('hidden'));

    if (target === 'upload') {
      tabUpload.classList.add('active');
      uploadSection.classList.remove('hidden');
      stopWebcam();
    } else if (target === 'webcam') {
      tabWebcam.classList.add('active');
      webcamSection.classList.remove('hidden');
      startWebcam();
    } else if (target === 'sample') {
      tabSample.classList.add('active');
      sampleSection.classList.remove('hidden');
      stopWebcam();
    }
  }

  tabUpload.addEventListener('click', () => switchTab('upload'));
  tabWebcam.addEventListener('click', () => switchTab('webcam'));
  tabSample.addEventListener('click', () => switchTab('sample'));

  // ----------------------------------------------------
  // File Selection
  // ----------------------------------------------------
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function handleFileSelect(file) {
    if (!file) return;

    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
      showToast('Định dạng ảnh không hỗ trợ. Vui lòng chọn ảnh JPEG, PNG hoặc WEBP.', 'error');
      return;
    }

    if (file.size > maxUploadBytes) {
      showToast(`Ảnh vượt quá giới hạn cho phép (${maxUploadMb} MB).`, 'error');
      return;
    }

    currentFile = file;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    // Update left preview box
    previewImage.src = currentPreviewUrl;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);
    previewBox.classList.remove('hidden');
    btnDetect.disabled = false;

    // Immediately display image in main right panel
    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');
    statsWrapper.classList.add('hidden');
    resultActions.classList.add('hidden');

    imageOriginal.src = currentPreviewUrl;
    imageOriginal.classList.remove('hidden');
    imageAnnotated.classList.add('hidden');

    // Reset exemplars on new image
    exemplarBoxes = [];
    updateExemplarBadge();
    updateWorkspaceBanner();

    setTimeout(() => {
      syncMainCanvasSize();
      redrawMainCanvas();
    }, 150);
  }

  function clearSelectedFile() {
    currentFile = null;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
    fileInput.value = '';
    previewBox.classList.add('hidden');
    btnDetect.disabled = true;
    exemplarBoxes = [];
    updateExemplarBadge();

    emptyState.classList.remove('hidden');
    resultContent.classList.add('hidden');
    workspaceBanner.classList.add('hidden');
  }

  btnRemoveImage.addEventListener('click', clearSelectedFile);

  // Drag & drop
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', e => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelect(dt.files[0]);
    }
  });

  fileInput.addEventListener('change', e => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  });

  // ----------------------------------------------------
  // Webcam Handling
  // ----------------------------------------------------
  async function startWebcam() {
    if (mediaStream) return;
    try {
      cameraOverlay.classList.add('hidden');
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      });
      cameraFeed.srcObject = mediaStream;
      btnToggleCamera.textContent = 'Dừng camera';
    } catch (err) {
      cameraOverlay.classList.remove('hidden');
      cameraErrorText.textContent = 'Không thể mở camera thiết bị: ' + (err.message || 'Bị chặn quyền truy cập');
      showToast('Không thể kết nối camera.', 'error');
    }
  }

  function stopWebcam() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
      cameraFeed.srcObject = null;
      btnToggleCamera.textContent = 'Bật camera';
    }
  }

  btnToggleCamera.addEventListener('click', () => {
    if (mediaStream) {
      stopWebcam();
    } else {
      startWebcam();
    }
  });

  btnSnapCamera.addEventListener('click', () => {
    if (!cameraFeed.videoWidth) {
      showToast('Camera chưa sẵn sàng để chụp.', 'error');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
      if (blob) {
        const file = new File([blob], `camera_${Date.now()}.jpg`, { type: 'image/jpeg' });
        handleFileSelect(file);
        showToast('Đã chụp ảnh từ camera thành công!', 'success');
      }
    }, 'image/jpeg', 0.92);
  });

  // ----------------------------------------------------
  // Sample Image
  // ----------------------------------------------------
  btnLoadSample.addEventListener('click', async () => {
    try {
      const res = await fetch('/static/sample.jpg');
      if (!res.ok) throw new Error('Không tải được ảnh mẫu');
      const blob = await res.blob();
      const sampleFile = new File([blob], 'sample_desktop_case.jpg', { type: 'image/jpeg' });
      handleFileSelect(sampleFile);
      showToast('Đã nạp ảnh mẫu kiểm thử!', 'success');
    } catch (err) {
      showToast('Lỗi khi nạp ảnh mẫu.', 'error');
    }
  });

  // ----------------------------------------------------
  // Detection Execution
  // ----------------------------------------------------
  async function performDetection() {
    if (!currentFile || isProcessing) return;

    if (currentModel === 'loca' && locaMode === 'few_shot' && exemplarBoxes.length === 0) {
      showToast('Vui lòng Click hoặc Kéo chuột trên ảnh để khoanh ít nhất 1 thùng máy làm mẫu!', 'error', 4500);
      return;
    }

    isProcessing = true;
    btnDetect.disabled = true;
    detectSpinner.classList.remove('hidden');
    if (currentModel === 'loca') {
      detectBtnText.textContent = 'LOCA đang tính mật độ...';
    } else if (currentModel === 'grounding_dino') {
      detectBtnText.textContent = 'Grounding DINO đang phân tích...';
    } else {
      detectBtnText.textContent = 'YOLO đang quét nhận diện...';
    }

    const formData = new FormData();
    formData.append('image', currentFile, currentFile.name || 'image.jpg');
    formData.append('model_type', currentModel);

    if (currentModel === 'loca') {
      formData.append('loca_mode', locaMode);
      formData.append('threshold_rel', locaThresholdRange.value);
      formData.append('exemplars', JSON.stringify(exemplarBoxes));
    } else if (currentModel === 'grounding_dino') {
      formData.append('prompt', dinoPromptInput.value || 'pc case. computer tower.');
      formData.append('box_threshold', dinoBoxThreshRange.value);
      formData.append('text_threshold', dinoTextThreshRange.value);
    } else {
      formData.append('confidence', confidenceRange.value);
      if (imgszSelect) formData.append('imgsz', imgszSelect.value);
      if (checkSliced) formData.append('sliced', checkSliced.checked ? 'true' : 'false');
      if (checkEnhance) formData.append('enhance', checkEnhance.checked ? 'true' : 'false');
    }

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Xử lý thất bại.');
      }

      renderResults(data);
      const countMsg = currentModel === 'loca'
        ? `Đếm mật độ: ${data.count}`
        : `Tìm thấy: ${data.count} đối tượng`;
      showToast(`Hoàn tất (${data.model})! ${countMsg}.`, 'success');
    } catch (err) {
      showToast(err.message || 'Lỗi kết nối tới máy chủ.', 'error');
    } finally {
      isProcessing = false;
      btnDetect.disabled = false;
      detectSpinner.classList.add('hidden');
      if (currentModel === 'loca') {
        detectBtnText.textContent = 'Đếm mật độ bằng LOCA';
      } else if (currentModel === 'grounding_dino') {
        detectBtnText.textContent = 'Nhận diện & Đếm bằng Grounding DINO';
      } else {
        detectBtnText.textContent = 'Nhận diện & Đếm bằng YOLO11';
      }
    }
  }

  btnDetect.addEventListener('click', performDetection);

  // ----------------------------------------------------
  // Render Results
  // ----------------------------------------------------
  function renderResults(data) {
    statsWrapper.classList.remove('hidden');
    resultActions.classList.remove('hidden');

    statCount.textContent = data.count;
    statTime.textContent = `${data.inference_ms} ms`;
    statRes.textContent = `${data.width} × ${data.height}`;

    if (data.model_type === 'loca') {
      statLabelCount.textContent = 'Số lượng (Mật độ)';
      statLabelConf.textContent = 'Số đỉnh cực đại (Peaks)';
      statConf.textContent = `${data.peaks_count || 0}`;
      btnViewAnnotated.textContent = 'Bản đồ nhiệt LOCA';
      tableTitleText.textContent = 'Vị trí các đỉnh cực đại (Density Peaks)';
      thScore.textContent = 'Điểm mật độ';
      thCoords.textContent = 'Tọa độ tâm đỉnh [x, y]';
    } else if (data.model_type === 'grounding_dino') {
      statLabelCount.textContent = 'Số lượng đối tượng';
      statLabelConf.textContent = 'Độ tin cậy TB';
      btnViewAnnotated.textContent = 'Ảnh nhận diện Grounding DINO';
      tableTitleText.textContent = `Chi tiết đối tượng (${data.prompt || 'Prompt'})`;
      thScore.textContent = 'Độ khớp (Score)';
      thCoords.textContent = 'Tọa độ Bounding Box [x1, y1, x2, y2]';

      let avgConf = 0;
      if (data.detections && data.detections.length > 0) {
        const totalConf = data.detections.reduce((sum, d) => sum + (d.confidence || 0), 0);
        avgConf = Math.round((totalConf / data.detections.length) * 100);
      }
      statConf.textContent = data.detections.length > 0 ? `${avgConf}%` : '0%';
    } else {
      statLabelCount.textContent = 'Số lượng Case';
      statLabelConf.textContent = 'Độ tin cậy TB';
      btnViewAnnotated.textContent = 'Ảnh nhận diện Bounding Box';
      tableTitleText.textContent = 'Chi tiết các đối tượng phát hiện';
      thScore.textContent = 'Độ tin cậy';
      thCoords.textContent = 'Tọa độ Bounding Box [x1, y1, x2, y2]';

      let avgConf = 0;
      if (data.detections && data.detections.length > 0) {
        const totalConf = data.detections.reduce((sum, d) => sum + (d.confidence || 0), 0);
        avgConf = Math.round((totalConf / data.detections.length) * 100);
      }
      statConf.textContent = data.detections.length > 0 ? `${avgConf}%` : '0%';
    }

    annotatedDataUrl = data.image;
    imageAnnotated.src = annotatedDataUrl;
    imageOriginal.src = currentPreviewUrl;

    setViewMode('annotated');

    // Populate Detections Table
    tableCount.textContent = data.model_type === 'loca' ? (data.peaks_count || 0) : data.count;
    detectionsTbody.innerHTML = '';

    if (!data.detections || data.detections.length === 0) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 4;
      cell.textContent = data.model_type === 'loca'
        ? 'Không tìm thấy đỉnh mật độ nào vượt ngưỡng. Thử giảm ngưỡng độ nhạy đỉnh.'
        : 'Không có case nào vượt qua ngưỡng tin cậy này. Thử kéo giảm ngưỡng.';
      cell.className = 'text-dim';
      row.appendChild(cell);
      detectionsTbody.appendChild(row);
      return;
    }

    data.detections.forEach((item, index) => {
      const row = document.createElement('tr');

      // Col 1: Index
      const tdIdx = document.createElement('td');
      tdIdx.textContent = index + 1;
      row.appendChild(tdIdx);

      // Col 2: Class/Badge
      const tdClass = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = 'badge-class';
      if (data.model_type === 'loca') {
        badge.textContent = `Peak #${index + 1}`;
        badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        badge.style.color = '#fbbf24';
      } else {
        badge.textContent = item.class_name || 'pc_case';
      }
      tdClass.appendChild(badge);
      row.appendChild(tdClass);

      // Col 3: Confidence / Score
      const tdConf = document.createElement('td');
      if (data.model_type === 'loca') {
        tdConf.textContent = Number(item.confidence).toFixed(4);
      } else {
        const confPct = Math.round(item.confidence * 1000) / 10;
        const confPill = document.createElement('div');
        confPill.className = 'conf-pill';
        const confText = document.createElement('span');
        confText.textContent = `${confPct}%`;
        const confBar = document.createElement('div');
        confBar.className = 'conf-bar';
        const confFill = document.createElement('span');
        confFill.className = 'conf-fill';
        confFill.style.width = `${confPct}%`;
        confBar.appendChild(confFill);
        confPill.appendChild(confText);
        confPill.appendChild(confBar);
        tdConf.appendChild(confPill);
      }
      row.appendChild(tdConf);

      // Col 4: Coordinates
      const tdCoords = document.createElement('td');
      tdCoords.className = 'coords-cell';
      if (data.model_type === 'loca' && data.peaks && data.peaks[index]) {
        const p = data.peaks[index];
        tdCoords.textContent = `x = ${p.x}, y = ${p.y}`;
      } else {
        const [x1, y1, x2, y2] = item.xyxy.map(Math.round);
        tdCoords.textContent = `[${x1}, ${y1}, ${x2}, ${y2}] (${x2 - x1}×${y2 - y1} px)`;
      }
      row.appendChild(tdCoords);

      detectionsTbody.appendChild(row);
    });
  }

  // ----------------------------------------------------
  // View Toggle (Annotated vs Original)
  // ----------------------------------------------------
  function setViewMode(mode) {
    if (mode === 'annotated') {
      btnViewAnnotated.classList.add('active');
      btnViewOriginal.classList.remove('active');
      imageAnnotated.classList.remove('hidden');
      imageOriginal.classList.add('hidden');
      mainDrawCanvas.classList.add('hidden'); // hide drawing overlay on annotated result
    } else {
      btnViewOriginal.classList.add('active');
      btnViewAnnotated.classList.remove('active');
      imageOriginal.classList.remove('hidden');
      imageAnnotated.classList.add('hidden');
      mainDrawCanvas.classList.remove('hidden');
      syncMainCanvasSize();
      redrawMainCanvas();
    }
  }

  btnViewAnnotated.addEventListener('click', () => setViewMode('annotated'));
  btnViewOriginal.addEventListener('click', () => setViewMode('original'));

  // ----------------------------------------------------
  // Download Result
  // ----------------------------------------------------
  btnDownload.addEventListener('click', () => {
    if (!annotatedDataUrl) return;
    const a = document.createElement('a');
    a.href = annotatedDataUrl;
    const prefix = currentModel === 'loca' ? 'loca_counted' : 'yolo_counted';
    a.download = `${prefix}_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Đang tải ảnh kết quả về máy...', 'success', 2000);
  });

  // ----------------------------------------------------
  // Fullscreen Modal
  // ----------------------------------------------------
  btnFullscreen.addEventListener('click', () => {
    const isAnnotated = !imageAnnotated.classList.contains('hidden');
    modalImg.src = isAnnotated ? annotatedDataUrl : currentPreviewUrl;
    modalZoom.classList.remove('hidden');
  });

  function closeModal() {
    modalZoom.classList.add('hidden');
    modalImg.src = '';
  }

  btnCloseModal.addEventListener('click', closeModal);
  modalBackdrop.addEventListener('click', closeModal);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !modalZoom.classList.contains('hidden')) {
      closeModal();
    }
  });

  // Initialize
  checkHealth();
});
