/**
 * Multi-Model Object & PC Case Counting Web Application
 * Supports YOLO11, LOCA (Density Integral), and Grounding DINO
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

  let currentModel = 'yolo11'; // 'yolo11', 'loca', or 'grounding_dino'
  let locaMode = 'few_shot';   // 'few_shot' or 'zero_shot'
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

  // Universal Preprocessing Controls
  const checkUseSplit = document.getElementById('check-use-split');
  const checkUseRoi = document.getElementById('check-use-roi');

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
  const yoloNmsRange = document.getElementById('yolo-nms-range');
  const yoloNmsVal = document.getElementById('yolo-nms-val');
  const checkSliced = document.getElementById('check-sliced');
  const checkEnhance = document.getElementById('check-enhance');

  // LOCA Parameters
  const radioLocaZero = document.getElementById('radio-loca-zero');
  const radioLocaFew = document.getElementById('radio-loca-few');
  const checkLocaFixedExemplars = document.getElementById('check-loca-fixed-exemplars');
  const checkLocaPadSquare = document.getElementById('check-loca-pad-square');
  const locaFewShotHelper = document.getElementById('loca-few-shot-helper');
  const exemplarCountBadge = document.getElementById('exemplar-count-badge');
  const btnClearExemplars = document.getElementById('btn-clear-exemplars');
  const locaPeakRatioRange = document.getElementById('loca-peak-ratio-range');
  const locaPeakRatioVal = document.getElementById('loca-peak-ratio-val');
  const sizeChoiceButtons = document.querySelectorAll('.btn-size-choice');

  // Grounding DINO Parameters
  const dinoPromptInput = document.getElementById('dino-prompt-input');
  const dinoBoxThreshRange = document.getElementById('dino-box-thresh-range');
  const dinoBoxThreshVal = document.getElementById('dino-box-thresh-val');
  const dinoNmsThreshRange = document.getElementById('dino-nms-thresh-range');
  const dinoNmsThreshVal = document.getElementById('dino-nms-thresh-val');
  const checkDinoFilterLarge = document.getElementById('check-dino-filter-large');
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
  const statCount = document.getElementById('stat-count');
  const statLabelCount = document.getElementById('stat-label-count');
  const statTime = document.getElementById('stat-time');
  const statRes = document.getElementById('stat-res');
  const statConf = document.getElementById('stat-conf');
  const statLabelConf = document.getElementById('stat-label-conf');

  const resultActions = document.getElementById('result-actions');
  const btnViewAnnotated = document.getElementById('btn-view-annotated');
  const btnViewOriginal = document.getElementById('btn-view-original');
  const btnDownload = document.getElementById('btn-download');
  const btnFullscreen = document.getElementById('btn-fullscreen');

  const tableCount = document.getElementById('table-count');
  const tableTitleText = document.getElementById('table-title-text');
  const thScore = document.getElementById('th-score');
  const thCoords = document.getElementById('th-coords');
  const detectionsTbody = document.getElementById('detections-tbody');

  // Modal & Toast
  const modalZoom = document.getElementById('modal-zoom');
  const modalImg = document.getElementById('modal-img');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const modalBackdrop = document.getElementById('modal-backdrop');
  const toastContainer = document.getElementById('toast-container');

  // Camera Elements
  const cameraFeed = document.getElementById('camera-feed');
  const cameraOverlay = document.getElementById('camera-overlay');
  const cameraErrorText = document.getElementById('camera-error-text');
  const btnSnapCamera = document.getElementById('btn-snap-camera');
  const btnToggleCamera = document.getElementById('btn-toggle-camera');
  const btnLoadSample = document.getElementById('btn-load-sample');

  // ----------------------------------------------------
  // Toast Notifications
  // ----------------------------------------------------
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.animation = 'toastFadeOut 0.3s forwards ease';
      setTimeout(() => toast.remove(), 320);
    }, duration);
  }

  // ----------------------------------------------------
  // Health & Service Check
  // ----------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Mất kết nối máy chủ');
      const data = await res.json();
      statusPill.className = 'status-pill status-ready';
      statusText.textContent = `${data.model} sẵn sàng • ${data.device || 'CPU'}`;
      if (data.max_upload_mb) {
        maxUploadMb = data.max_upload_mb;
        maxUploadBytes = maxUploadMb * 1024 * 1024;
        dropFormatsText.textContent = `Hỗ trợ JPG, PNG, WEBP (Tối đa ${maxUploadMb} MB)`;
      }
    } catch (err) {
      statusPill.className = 'status-pill status-offline';
      statusText.textContent = 'Không thể kết nối Backend';
    }
  }
  checkHealth();

  // ----------------------------------------------------
  // Slider Value Bindings
  // ----------------------------------------------------
  if (confidenceRange && confidenceVal) {
    confidenceRange.addEventListener('input', e => {
      const val = parseFloat(e.target.value).toFixed(2);
      confidenceVal.textContent = `${val} (${Math.round(val * 100)}%)`;
    });
  }
  if (yoloNmsRange && yoloNmsVal) {
    yoloNmsRange.addEventListener('input', e => {
      yoloNmsVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }
  if (locaPeakRatioRange && locaPeakRatioVal) {
    locaPeakRatioRange.addEventListener('input', e => {
      const val = parseFloat(e.target.value).toFixed(2);
      locaPeakRatioVal.textContent = `${val} (${Math.round(val * 100)}%)`;
    });
  }
  if (dinoBoxThreshRange && dinoBoxThreshVal) {
    dinoBoxThreshRange.addEventListener('input', e => {
      const val = parseFloat(e.target.value).toFixed(2);
      dinoBoxThreshVal.textContent = `${val} (${Math.round(val * 100)}%)`;
    });
  }
  if (dinoNmsThreshRange && dinoNmsThreshVal) {
    dinoNmsThreshRange.addEventListener('input', e => {
      dinoNmsThreshVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // ----------------------------------------------------
  // Prompt Pills for Grounding DINO
  // ----------------------------------------------------
  promptPillButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      promptPillButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const prompt = btn.dataset.prompt;
      if (dinoPromptInput) {
        dinoPromptInput.value = prompt;
      }
    });
  });

  // ----------------------------------------------------
  // Switch Model Tabs
  // ----------------------------------------------------
  function switchModel(model) {
    currentModel = model;
    btnModelYolo.classList.toggle('active', model === 'yolo11');
    btnModelLoca.classList.toggle('active', model === 'loca');
    btnModelDino.classList.toggle('active', model === 'grounding_dino');

    yoloConfigGroup.classList.toggle('hidden', model !== 'yolo11');
    locaConfigGroup.classList.toggle('hidden', model !== 'loca');
    dinoConfigGroup.classList.toggle('hidden', model !== 'grounding_dino');

    if (model === 'loca') {
      detectBtnText.textContent = 'Đếm mật độ bằng LOCA';
      updateWorkspaceBanner();
    } else if (model === 'grounding_dino') {
      detectBtnText.textContent = 'Nhận diện & Đếm bằng Grounding DINO';
      workspaceBanner.classList.add('hidden');
    } else {
      detectBtnText.textContent = 'Nhận diện & Đếm bằng YOLO11';
      workspaceBanner.classList.add('hidden');
    }
    redrawMainCanvas();
  }

  btnModelYolo.addEventListener('click', () => switchModel('yolo11'));
  btnModelLoca.addEventListener('click', () => switchModel('loca'));
  btnModelDino.addEventListener('click', () => switchModel('grounding_dino'));

  // ----------------------------------------------------
  // LOCA Mode & Fixed Exemplars
  // ----------------------------------------------------
  if (radioLocaFew) {
    radioLocaFew.addEventListener('change', () => {
      locaMode = 'few_shot';
      updateLocaUi();
    });
  }
  if (radioLocaZero) {
    radioLocaZero.addEventListener('change', () => {
      locaMode = 'zero_shot';
      updateLocaUi();
    });
  }
  if (checkLocaFixedExemplars) {
    checkLocaFixedExemplars.addEventListener('change', updateLocaUi);
  }

  function updateLocaUi() {
    const isFixed = checkLocaFixedExemplars && checkLocaFixedExemplars.checked;
    if (locaMode === 'few_shot' && !isFixed) {
      locaFewShotHelper.classList.remove('hidden');
    } else {
      locaFewShotHelper.classList.add('hidden');
    }
    updateWorkspaceBanner();
    redrawMainCanvas();
  }

  function updateWorkspaceBanner() {
    if (currentModel !== 'loca' || locaMode !== 'few_shot' || !currentFile) {
      workspaceBanner.classList.add('hidden');
      return;
    }
    const isFixed = checkLocaFixedExemplars && checkLocaFixedExemplars.checked;
    if (isFixed) {
      workspaceBanner.classList.remove('hidden');
      bannerIcon.textContent = '🎯';
      bannerText.textContent = 'Đang dùng bộ 3 khung mẫu chuẩn cố định (Cả hàng trên & dưới).';
      btnBannerClear.classList.add('hidden');
      return;
    }

    workspaceBanner.classList.remove('hidden');
    btnBannerClear.classList.remove('hidden');
    const count = exemplarBoxes.length;
    bannerIcon.textContent = count >= 3 ? '✅' : '🎯';
    bannerText.textContent = count >= 3
      ? `Đã khoanh đủ 3 mẫu đối tượng! Sẵn sàng đếm.`
      : `Click vào thùng máy để đặt khung vuông mẫu (Đã khoanh: ${count}/3)`;
  }

  if (btnBannerClear) {
    btnBannerClear.addEventListener('click', () => {
      exemplarBoxes = [];
      updateExemplarBadge();
      updateWorkspaceBanner();
      redrawMainCanvas();
    });
  }

  function updateExemplarBadge() {
    if (exemplarCountBadge) {
      exemplarCountBadge.textContent = `${exemplarBoxes.length} / 3 mẫu`;
    }
  }

  if (btnClearExemplars) {
    btnClearExemplars.addEventListener('click', () => {
      exemplarBoxes = [];
      updateExemplarBadge();
      updateWorkspaceBanner();
      redrawMainCanvas();
    });
  }

  // ----------------------------------------------------
  // Input Tabs (Upload / Camera / Sample)
  // ----------------------------------------------------
  function activateTab(activeBtn, activeSection) {
    [tabUpload, tabWebcam, tabSample].forEach(t => t.classList.remove('active'));
    [uploadSection, webcamSection, sampleSection].forEach(s => s.classList.add('hidden'));
    activeBtn.classList.add('active');
    activeSection.classList.remove('hidden');
    if (activeBtn !== tabWebcam && mediaStream) {
      stopWebcam();
    }
  }

  tabUpload.addEventListener('click', () => activateTab(tabUpload, uploadSection));
  tabWebcam.addEventListener('click', () => {
    activateTab(tabWebcam, webcamSection);
    startWebcam();
  });
  tabSample.addEventListener('click', () => activateTab(tabSample, sampleSection));

  // ----------------------------------------------------
  // File Loading & Previews
  // ----------------------------------------------------
  function handleFileSelect(file) {
    if (!file) return;
    if (file.size > maxUploadBytes) {
      showToast(`Kích thước file (${(file.size / 1024 / 1024).toFixed(1)} MB) vượt quá giới hạn ${maxUploadMb} MB.`, 'error');
      return;
    }

    currentFile = file;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    previewImage.src = currentPreviewUrl;
    previewFilename.textContent = file.name || 'image.jpg';
    previewFilesize.textContent = `${(file.size / 1024).toFixed(0)} KB`;
    previewBox.classList.remove('hidden');
    btnDetect.disabled = false;

    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');
    statsWrapper.classList.add('hidden');
    resultActions.classList.add('hidden');

    imageOriginal.src = currentPreviewUrl;
    imageOriginal.classList.remove('hidden');
    imageAnnotated.classList.add('hidden');

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
  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, e => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, e => {
      e.preventDefault();
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

  // Sample image
  btnLoadSample.addEventListener('click', async () => {
    try {
      const res = await fetch('/static/sample.jpg');
      if (!res.ok) throw new Error('Không tải được ảnh mẫu');
      const blob = await res.blob();
      const sampleFile = new File([blob], 'sample_desktop_case.jpg', { type: 'image/jpeg' });
      handleFileSelect(sampleFile);
      showToast('Đã nạp ảnh mẫu kiểm thử kệ PC!', 'success');
    } catch (err) {
      showToast('Lỗi khi nạp ảnh mẫu.', 'error');
    }
  });

  // ----------------------------------------------------
  // Canvas Sizing & Drawing for Exemplars
  // ----------------------------------------------------
  function syncMainCanvasSize() {
    if (!imageOriginal.naturalWidth) return;
    mainDrawCanvas.width = imageOriginal.clientWidth;
    mainDrawCanvas.height = imageOriginal.clientHeight;
  }
  window.addEventListener('resize', () => {
    syncMainCanvasSize();
    redrawMainCanvas();
  });

  function redrawMainCanvas() {
    const ctx = mainDrawCanvas.getContext('2d');
    ctx.clearRect(0, 0, mainDrawCanvas.width, mainDrawCanvas.height);

    if (currentModel !== 'loca' || locaMode !== 'few_shot' || !imageOriginal.naturalWidth) return;
    const isFixed = checkLocaFixedExemplars && checkLocaFixedExemplars.checked;
    if (isFixed) return; // Do not draw manual boxes when fixed preset is on

    const scaleX = mainDrawCanvas.width / imageOriginal.naturalWidth;
    const scaleY = mainDrawCanvas.height / imageOriginal.naturalHeight;

    exemplarBoxes.forEach((box, i) => {
      const x1 = box[0] * scaleX;
      const y1 = box[1] * scaleY;
      const x2 = box[2] * scaleX;
      const y2 = box[3] * scaleY;

      ctx.strokeStyle = '#ff9900';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.fillStyle = '#ff9900';
      ctx.fillRect(x1, Math.max(0, y1 - 22), 65, 22);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(`Mẫu #${i + 1}`, x1 + 5, Math.max(15, y1 - 6));
    });
  }

  mainDrawCanvas.addEventListener('click', e => {
    if (currentModel !== 'loca' || locaMode !== 'few_shot') return;
    if (checkLocaFixedExemplars && checkLocaFixedExemplars.checked) return;
    if (exemplarBoxes.length >= 3) {
      showToast('Đã đủ 3 mẫu. Bấm "Xóa mẫu" nếu bạn muốn khoanh lại.', 'info');
      return;
    }

    const rect = mainDrawCanvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const scaleX = imageOriginal.naturalWidth / mainDrawCanvas.width;
    const scaleY = imageOriginal.naturalHeight / mainDrawCanvas.height;

    const origClickX = clickX * scaleX;
    const origClickY = clickY * scaleY;

    const half = clickSquareSize / 2;
    const x1 = Math.max(0, origClickX - half);
    const y1 = Math.max(0, origClickY - half);
    const x2 = Math.min(imageOriginal.naturalWidth, origClickX + half);
    const y2 = Math.min(imageOriginal.naturalHeight, origClickY + half);

    exemplarBoxes.push([Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)]);
    updateExemplarBadge();
    updateWorkspaceBanner();
    redrawMainCanvas();
    showToast(`Đã thêm mẫu #${exemplarBoxes.length}!`, 'success');
  });

  // ----------------------------------------------------
  // Perform Detection (Inference Request)
  // ----------------------------------------------------
  async function performDetection() {
    if (!currentFile || isProcessing) return;

    const isFixed = checkLocaFixedExemplars && checkLocaFixedExemplars.checked;
    if (currentModel === 'loca' && locaMode === 'few_shot' && !isFixed && exemplarBoxes.length === 0) {
      showToast('Vui lòng Click trên ảnh để chọn ít nhất 1 mẫu, hoặc bật "Dùng 3 khung mẫu chuẩn cố định"!', 'error', 4500);
      return;
    }

    isProcessing = true;
    btnDetect.disabled = true;
    detectSpinner.classList.remove('hidden');

    if (currentModel === 'loca') {
      detectBtnText.textContent = 'LOCA đang tích phân mật độ...';
    } else if (currentModel === 'grounding_dino') {
      detectBtnText.textContent = 'Grounding DINO đang nhận diện...';
    } else {
      detectBtnText.textContent = 'YOLO11 đang quét nhận diện...';
    }

    const formData = new FormData();
    formData.append('image', currentFile, currentFile.name || 'image.jpg');
    formData.append('model_type', currentModel);

    // Universal shelf pre-processing
    if (checkUseSplit) formData.append('use_split', checkUseSplit.checked ? 'true' : 'false');
    if (checkUseRoi) formData.append('use_roi', checkUseRoi.checked ? 'true' : 'false');

    if (currentModel === 'loca') {
      formData.append('loca_mode', locaMode);
      if (checkLocaFixedExemplars) {
        formData.append('use_fixed_exemplars', checkLocaFixedExemplars.checked ? 'true' : 'false');
      }
      if (checkLocaPadSquare) {
        formData.append('pad_square', checkLocaPadSquare.checked ? 'true' : 'false');
      }
      if (locaPeakRatioRange) {
        formData.append('peak_min_relative', locaPeakRatioRange.value);
      }
      formData.append('exemplars', JSON.stringify(exemplarBoxes));
    } else if (currentModel === 'grounding_dino') {
      formData.append('prompt', dinoPromptInput ? dinoPromptInput.value : 'pc case.');
      if (dinoBoxThreshRange) formData.append('box_threshold', dinoBoxThreshRange.value);
      if (dinoNmsThreshRange) formData.append('nms_threshold', dinoNmsThreshRange.value);
      if (checkDinoFilterLarge) {
        formData.append('filter_large_boxes', checkDinoFilterLarge.checked ? 'true' : 'false');
      }
    } else {
      if (confidenceRange) formData.append('confidence', confidenceRange.value);
      if (yoloNmsRange) formData.append('nms_iou', yoloNmsRange.value);
      if (imgszSelect) formData.append('imgsz', imgszSelect.value);
      if (checkSliced) formData.append('sliced', checkSliced.checked ? 'true' : 'false');
      if (checkEnhance) formData.append('enhance', checkEnhance.checked ? 'true' : 'false');
    }

    try {
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Xử lý thất bại.');

      renderResults(data);
      const countMsg = currentModel === 'loca'
        ? `Đếm mật độ: ${data.count} (Tổng: ${data.raw_count})`
        : `Tìm thấy: ${data.count} đối tượng`;
      showToast(`Hoàn tất (${data.model})! ${countMsg}.`, 'success');
    } catch (err) {
      showToast(err.message || 'Lỗi khi gửi yêu cầu tới máy chủ.', 'error');
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
    statRes.textContent = `${data.device || 'CPU'} • ${data.width}×${data.height}`;

    if (data.model_type === 'loca') {
      statLabelCount.textContent = 'Số lượng (Mật độ)';
      statLabelConf.textContent = 'Tổng mật độ thực';
      statConf.textContent = `${data.raw_count}`;
      btnViewAnnotated.textContent = 'Bản đồ nhiệt LOCA';
      tableTitleText.textContent = 'Vị trí các đỉnh mật độ (Peaks)';
      thScore.textContent = 'Điểm mật độ';
      thCoords.textContent = 'Tọa độ [x, y]';
    } else if (data.model_type === 'grounding_dino') {
      statLabelCount.textContent = 'Số lượng Case';
      statLabelConf.textContent = 'Độ tin cậy TB';
      btnViewAnnotated.textContent = 'Ảnh nhận diện Grounding DINO';
      tableTitleText.textContent = `Chi tiết đối tượng (${data.prompt})`;
      thScore.textContent = 'Score';
      thCoords.textContent = 'Tọa độ [x1, y1, x2, y2]';

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
      thCoords.textContent = 'Tọa độ [x1, y1, x2, y2]';

      let avgConf = 0;
      if (data.detections && data.detections.length > 0) {
        const totalConf = data.detections.reduce((sum, d) => sum + (d.confidence || 0), 0);
        avgConf = Math.round((totalConf / data.detections.length) * 100);
      }
      statConf.textContent = data.detections.length > 0 ? `${avgConf}%` : '0%';
    }

    annotatedDataUrl = data.image;
    imageAnnotated.src = annotatedDataUrl;
    imageAnnotated.classList.remove('hidden');
    imageOriginal.classList.add('hidden');
    btnViewAnnotated.classList.add('active');
    btnViewOriginal.classList.remove('active');

    // Populate detections table
    tableCount.textContent = data.detections ? data.detections.length : 0;
    detectionsTbody.innerHTML = '';
    if (data.detections && data.detections.length > 0) {
      data.detections.forEach((d, idx) => {
        const tr = document.createElement('tr');
        const coords = d.xyxy ? d.xyxy.map(c => Math.round(c)).join(', ') : '';
        const score = typeof d.confidence === 'number' ? (d.confidence > 1 ? d.confidence.toFixed(2) : `${(d.confidence * 100).toFixed(1)}%`) : '-';
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td><strong>${d.class_name || 'pc_case'}</strong></td>
          <td>${score}</td>
          <td><code>[${coords}]</code></td>
        `;
        detectionsTbody.appendChild(tr);
      });
    } else {
      detectionsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: #888;">Không phát hiện đối tượng nào.</td></tr>`;
    }
  }

  // ----------------------------------------------------
  // View Switch & Download
  // ----------------------------------------------------
  btnViewAnnotated.addEventListener('click', () => {
    if (!annotatedDataUrl) return;
    imageAnnotated.classList.remove('hidden');
    imageOriginal.classList.add('hidden');
    btnViewAnnotated.classList.add('active');
    btnViewOriginal.classList.remove('active');
  });

  btnViewOriginal.addEventListener('click', () => {
    if (!currentPreviewUrl) return;
    imageAnnotated.classList.add('hidden');
    imageOriginal.classList.remove('hidden');
    btnViewAnnotated.classList.remove('active');
    btnViewOriginal.classList.add('active');
  });

  btnDownload.addEventListener('click', () => {
    const targetUrl = annotatedDataUrl || currentPreviewUrl;
    if (!targetUrl) return;
    const a = document.createElement('a');
    a.href = targetUrl;
    a.download = `counted_${Date.now()}.jpg`;
    a.click();
  });

  btnFullscreen.addEventListener('click', () => {
    const targetUrl = annotatedDataUrl || currentPreviewUrl;
    if (!targetUrl) return;
    modalImg.src = targetUrl;
    modalZoom.classList.remove('hidden');
  });

  btnCloseModal.addEventListener('click', () => modalZoom.classList.add('hidden'));
  modalBackdrop.addEventListener('click', () => modalZoom.classList.add('hidden'));
});
