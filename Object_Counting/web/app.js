/**
 * Multi-Model Object & PC Case Counting Web Application
 * Supports YOLO11, LOCA, Grounding DINO, and AnomalyCLIP heatmaps.
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
  let lastHealth = null;
  let selectionVersion = 0;
  let lastResultModel = null;
  let activeInput = 'image';
  let currentVideoFile = null;
  let videoPreviewUrl = null;
  let videoJob = null;
  let videoSubmitting = false;
  let videoCancelling = false;
  let videoPollTimer = null;
  let videoPollErrors = 0;
  let maxVideoBytes = 128 * 1024 * 1024;
  let maxVideoSeconds = 300;
  let maxVideoSamples = 600;
  let selectedVideoDuration = null;

  let currentModel = 'yolo11'; // Counting models or 'anomalyclip'.
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
  const tabVideo = document.getElementById('tab-video');
  const uploadSection = document.getElementById('upload-section');
  const webcamSection = document.getElementById('webcam-section');
  const sampleSection = document.getElementById('sample-section');
  const videoSection = document.getElementById('video-section');
  const videoDropZone = document.getElementById('video-drop-zone');
  const videoFileInput = document.getElementById('video-file-input');
  const videoFormatsText = document.getElementById('video-formats-text');
  const videoPreviewBox = document.getElementById('video-preview-box');
  const videoPreview = document.getElementById('video-preview');
  const videoFilename = document.getElementById('video-filename');
  const videoFilesize = document.getElementById('video-filesize');
  const btnClearVideo = document.getElementById('btn-clear-video');
  const videoSampleFps = document.getElementById('video-sample-fps');
  const videoSamplingHelp = document.getElementById('video-sampling-help');
  const videoModelNote = document.getElementById('video-model-note');
  const videoResultPanel = document.getElementById('video-result-panel');
  const videoEmptyState = document.getElementById('video-empty-state');
  const videoJobProgress = document.getElementById('video-job-progress');
  const videoJobState = document.getElementById('video-job-state');
  const videoProgressBar = document.getElementById('video-progress-bar');
  const videoJobDetail = document.getElementById('video-job-detail');
  const videoLastMetric = document.getElementById('video-last-metric');
  const videoJobError = document.getElementById('video-job-error');
  const btnCancelVideo = document.getElementById('btn-cancel-video');
  const btnRetryVideoStatus = document.getElementById('btn-retry-video-status');
  const videoCompletedResult = document.getElementById('video-completed-result');
  const videoResult = document.getElementById('video-result');
  const videoResultSummary = document.getElementById('video-result-summary');
  const videoDownloadLink = document.getElementById('video-download-link');
  const videoCsvLink = document.getElementById('video-csv-link');

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
  const shelfPreprocessingGroup = document.getElementById('shelf-preprocessing-group');

  // Model Selector
  const btnModelYolo = document.getElementById('btn-model-yolo');
  const btnModelLoca = document.getElementById('btn-model-loca');
  const btnModelDino = document.getElementById('btn-model-dino');
  const btnModelAnomaly = document.getElementById('btn-model-anomalyclip');
  const yoloConfigGroup = document.getElementById('yolo-config-group');
  const locaConfigGroup = document.getElementById('loca-config-group');
  const dinoConfigGroup = document.getElementById('dino-config-group');
  const anomalyConfigGroup = document.getElementById('anomalyclip-config-group');
  const anomalyImgszSelect = document.getElementById('anomaly-imgsz-select');
  const anomalyOpacityRange = document.getElementById('anomaly-opacity-range');
  const anomalyOpacityVal = document.getElementById('anomaly-opacity-val');
  const anomalyStatus = document.getElementById('anomalyclip-status');

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
  const detectionsTableContainer = document.getElementById('detections-table-container');
  const anomalyResultInfo = document.getElementById('anomaly-result-info');

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
  function renderHealth(data) {
    const anomaly = data.anomalyclip;
    if (anomaly) {
      anomalyStatus.classList.toggle('is-unavailable', !anomaly.ready);
      anomalyStatus.textContent = anomaly.ready
        ? 'Đã có đủ trọng số AnomalyCLIP. Sẵn sàng phân tích ảnh.'
        : 'AnomalyCLIP chưa sẵn sàng. Hoàn tất bước cài đặt mô hình rồi tải lại trang.';
    } else {
      anomalyStatus.classList.add('is-unavailable');
      anomalyStatus.textContent = 'Máy chủ chưa cung cấp trạng thái AnomalyCLIP. Khởi động lại web sau khi cập nhật.';
    }
    if (currentModel === 'anomalyclip') {
      statusPill.className = `status-pill ${anomaly && anomaly.ready ? 'status-ready' : 'status-loading'}`;
      statusText.textContent = anomaly && anomaly.ready
        ? `AnomalyCLIP sẵn sàng • ${anomaly.device || data.device || 'CPU'}`
        : 'AnomalyCLIP chưa sẵn sàng';
    } else {
      statusPill.className = `status-pill ${data.ready ? 'status-ready' : 'status-loading'}`;
      statusText.textContent = data.ready
        ? `${data.model} sẵn sàng • ${data.device || 'CPU'}`
        : `${data.model || 'Mô hình'} chưa sẵn sàng`;
    }
    if (data.video) {
      if (Number.isFinite(data.video.max_upload_mb)) maxVideoBytes = data.video.max_upload_mb * 1024 * 1024;
      if (Number.isFinite(data.video.max_duration_seconds)) maxVideoSeconds = data.video.max_duration_seconds;
      if (Number.isFinite(data.video.max_sampled_frames)) maxVideoSamples = data.video.max_sampled_frames;
      videoFormatsText.textContent = `MP4, MOV, AVI, WEBM · tối đa ${Math.round(maxVideoBytes / 1024 / 1024)} MB và ${Math.round(maxVideoSeconds / 60)} phút`;
      videoSamplingHelp.textContent = `Chỉ phân tích frame lấy mẫu; tối đa ${maxVideoSamples} frame mỗi video. Video kết quả không có âm thanh. Với AnomalyCLIP trên CPU, bắt đầu bằng 0,5 hoặc 1 frame/giây.`;
    }
    updateDetectButton();
  }

  function actionLabel(model = currentModel) {
    if (activeInput === 'video') return `Xử lý video bằng ${model === 'anomalyclip' ? 'AnomalyCLIP' : model === 'yolo11' ? 'YOLO11' : model === 'loca' ? 'LOCA' : 'Grounding DINO'}`;
    if (model === 'anomalyclip') return 'Phân tích bất thường bằng AnomalyCLIP';
    if (model === 'loca') return 'Đếm mật độ bằng LOCA';
    if (model === 'grounding_dino') return 'Nhận diện & Đếm bằng Grounding DINO';
    return 'Nhận diện & Đếm bằng YOLO11';
  }

  function updateDetectButton() {
    const anomalyUnavailable = currentModel === 'anomalyclip'
      && !(lastHealth && lastHealth.anomalyclip && lastHealth.anomalyclip.ready === true);
    if (activeInput === 'video') {
      const issue = videoUnavailableMessage();
      videoModelNote.textContent = issue || 'Sẵn sàng xử lý video lấy mẫu. Bạn có thể dừng công việc trong khi đang chạy.';
      videoModelNote.classList.toggle('is-unavailable', Boolean(issue));
      btnDetect.disabled = isProcessing || isVideoActive() || !currentVideoFile || Boolean(issue);
    } else {
      btnDetect.disabled = isProcessing || isVideoActive() || !currentFile || anomalyUnavailable;
    }
    if (!isProcessing && !isVideoActive()) detectBtnText.textContent = actionLabel();
  }

  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Mất kết nối máy chủ');
      const data = await res.json();
      lastHealth = data;
      renderHealth(data);
      if (data.max_upload_mb) {
        maxUploadMb = data.max_upload_mb;
        maxUploadBytes = maxUploadMb * 1024 * 1024;
        dropFormatsText.textContent = `Hỗ trợ JPG, PNG, WEBP (Tối đa ${maxUploadMb} MB)`;
      }
    } catch (err) {
      lastHealth = null;
      statusPill.className = 'status-pill status-offline';
      statusText.textContent = 'Không thể kết nối Backend';
      anomalyStatus.classList.add('is-unavailable');
      anomalyStatus.textContent = 'Không thể kết nối máy chủ để kiểm tra AnomalyCLIP.';
      updateDetectButton();
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
  anomalyOpacityRange.addEventListener('input', e => {
    anomalyOpacityVal.textContent = Number(e.target.value).toFixed(2);
  });

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
    if (isVideoActive()) {
      showToast('Video đang được xử lý. Dừng hoặc đợi hoàn tất trước khi đổi mô hình.', 'info');
      return;
    }
    if (currentModel !== model) {
      selectionVersion += 1;
      annotatedDataUrl = null;
      lastResultModel = null;
      statsWrapper.classList.add('hidden');
      resultActions.classList.add('hidden');
      imageAnnotated.classList.add('hidden');
      imageOriginal.classList.remove('hidden');
    }
    currentModel = model;
    btnModelYolo.classList.toggle('active', model === 'yolo11');
    btnModelLoca.classList.toggle('active', model === 'loca');
    btnModelDino.classList.toggle('active', model === 'grounding_dino');
    btnModelAnomaly.classList.toggle('active', model === 'anomalyclip');

    yoloConfigGroup.classList.toggle('hidden', model !== 'yolo11');
    locaConfigGroup.classList.toggle('hidden', model !== 'loca');
    dinoConfigGroup.classList.toggle('hidden', model !== 'grounding_dino');
    anomalyConfigGroup.classList.toggle('hidden', model !== 'anomalyclip');
    shelfPreprocessingGroup.classList.toggle('hidden', model === 'anomalyclip');
    checkUseSplit.disabled = model === 'anomalyclip';
    checkUseRoi.disabled = model === 'anomalyclip';
    detectBtnText.textContent = isProcessing ? 'Đang xử lý yêu cầu trước...' : actionLabel(model);

    if (model === 'loca') {
      updateWorkspaceBanner();
    } else {
      workspaceBanner.classList.add('hidden');
    }
    if (lastHealth) renderHealth(lastHealth);
    if (model === 'anomalyclip') checkHealth();
    updateDetectButton();
    redrawMainCanvas();
  }

  btnModelYolo.addEventListener('click', () => switchModel('yolo11'));
  btnModelLoca.addEventListener('click', () => switchModel('loca'));
  btnModelDino.addEventListener('click', () => switchModel('grounding_dino'));
  btnModelAnomaly.addEventListener('click', () => switchModel('anomalyclip'));

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
    if (isVideoActive() || isProcessing) {
      showToast('Đợi xử lý hoàn tất hoặc dừng video trước khi đổi nguồn.', 'info');
      return false;
    }
    [tabUpload, tabWebcam, tabSample, tabVideo].forEach(t => t.classList.remove('active'));
    [uploadSection, webcamSection, sampleSection, videoSection].forEach(s => s.classList.add('hidden'));
    activeBtn.classList.add('active');
    activeSection.classList.remove('hidden');
    if (activeBtn !== tabWebcam && mediaStream) {
      stopWebcam();
    }
    activeInput = activeBtn === tabVideo ? 'video' : 'image';
    const isVideo = activeInput === 'video';
    videoResultPanel.classList.toggle('hidden', !isVideo);
    previewBox.classList.toggle('hidden', isVideo || !currentFile);
    resultContent.classList.toggle('hidden', isVideo || !currentFile);
    emptyState.classList.toggle('hidden', isVideo || Boolean(currentFile));
    resultActions.classList.toggle('hidden', isVideo || !annotatedDataUrl);
    if (!isVideo && videoPreview.pause) videoPreview.pause();
    if (!isVideo && videoResult.pause) videoResult.pause();
    updateDetectButton();
    return true;
  }

  tabUpload.addEventListener('click', () => activateTab(tabUpload, uploadSection));
  tabWebcam.addEventListener('click', () => {
    if (activateTab(tabWebcam, webcamSection)) startWebcam();
  });
  tabSample.addEventListener('click', () => activateTab(tabSample, sampleSection));
  tabVideo.addEventListener('click', () => activateTab(tabVideo, videoSection));

  // ----------------------------------------------------
  // File Loading & Previews
  // ----------------------------------------------------
  function handleFileSelect(file) {
    if (!file || isVideoActive()) return;
    if (file.size > maxUploadBytes) {
      showToast(`Kích thước file (${(file.size / 1024 / 1024).toFixed(1)} MB) vượt quá giới hạn ${maxUploadMb} MB.`, 'error');
      return;
    }

    selectionVersion += 1;
    annotatedDataUrl = null;
    lastResultModel = null;
    currentFile = file;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    previewImage.src = currentPreviewUrl;
    previewFilename.textContent = file.name || 'image.jpg';
    previewFilesize.textContent = `${(file.size / 1024).toFixed(0)} KB`;
    previewBox.classList.remove('hidden');
    updateDetectButton();

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
    if (isVideoActive()) return;
    selectionVersion += 1;
    annotatedDataUrl = null;
    lastResultModel = null;
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
    if (isVideoActive()) return;
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
  // Video upload, background-job status, and sampled output
  // ----------------------------------------------------
  function isVideoActive() {
    return videoSubmitting || Boolean(videoJob && ['queued', 'processing', 'encoding'].includes(videoJob.state));
  }

  function videoUnavailableMessage() {
    if (!['yolo11', 'anomalyclip'].includes(currentModel)) {
      return 'Video hiện hỗ trợ YOLO11 (đếm case) và AnomalyCLIP (phân tích bất thường). Chọn một trong hai mô hình để tiếp tục.';
    }
    if (!lastHealth || !lastHealth.video) return 'Máy chủ chưa cung cấp chế độ video. Khởi động lại web sau khi cập nhật rồi tải lại trang.';
    if (lastHealth.video.ready !== true) return 'Bộ xử lý video chưa sẵn sàng. Kiểm tra FFmpeg trên máy rồi tải lại trang.';
    if (currentModel === 'anomalyclip' && !(lastHealth.anomalyclip && lastHealth.anomalyclip.ready === true)) {
      return 'AnomalyCLIP chưa sẵn sàng. Hoàn tất cài đặt trọng số rồi tải lại trang.';
    }
    if (currentModel === 'yolo11' && lastHealth.ready !== true) return 'Chưa có trọng số YOLO11 để xử lý video.';
    if (selectedVideoDuration !== null && Math.ceil(selectedVideoDuration * Number(videoSampleFps.value)) > maxVideoSamples) {
      return `Video ở tốc độ lấy mẫu này vượt ${maxVideoSamples} frame. Giảm số frame mỗi giây hoặc chọn đoạn ngắn hơn.`;
    }
    return '';
  }

  function setVideoControlsLocked(locked) {
    const fixedControls = [tabUpload, tabVideo, tabWebcam, tabSample, btnModelYolo, btnModelLoca,
      btnModelDino, btnModelAnomaly, fileInput, btnRemoveImage, btnLoadSample,
      videoFileInput, btnClearVideo, videoSampleFps, checkUseSplit, checkUseRoi];
    fixedControls.forEach(control => { control.disabled = locked; });
    document.querySelectorAll('.model-config-group input, .model-config-group select, .model-config-group button')
      .forEach(control => { control.disabled = locked; });
    if (!locked) {
      checkUseSplit.disabled = currentModel === 'anomalyclip';
      checkUseRoi.disabled = currentModel === 'anomalyclip';
    }
    videoDropZone.classList.toggle('is-locked', locked);
    updateDetectButton();
  }

  function clearVideoSelection() {
    if (isVideoActive()) return;
    if (videoPollTimer !== null) clearTimeout(videoPollTimer);
    videoPollTimer = null;
    currentVideoFile = null;
    selectedVideoDuration = null;
    videoJob = null;
    videoCancelling = false;
    videoPollErrors = 0;
    videoFileInput.value = '';
    if (videoPreview.pause) videoPreview.pause();
    if (videoResult.pause) videoResult.pause();
    videoPreview.removeAttribute('src');
    videoResult.removeAttribute('src');
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    videoPreviewUrl = null;
    videoPreviewBox.classList.add('hidden');
    videoDropZone.classList.remove('hidden');
    videoEmptyState.classList.remove('hidden');
    videoJobProgress.classList.add('hidden');
    videoCompletedResult.classList.add('hidden');
    videoJobError.classList.add('hidden');
    btnRetryVideoStatus.classList.add('hidden');
    updateDetectButton();
  }

  function handleVideoSelect(file) {
    if (!file || isVideoActive() || isProcessing) return;
    if (!/\.(mp4|mov|avi|webm)$/i.test(file.name || '')) {
      showToast('Chọn video MP4, MOV, AVI hoặc WEBM.', 'error');
      return;
    }
    if (file.size < 1 || file.size > maxVideoBytes) {
      showToast(`Chọn video không rỗng và tối đa ${Math.round(maxVideoBytes / 1024 / 1024)} MB.`, 'error');
      return;
    }
    clearVideoSelection();
    currentVideoFile = file;
    videoPreviewUrl = URL.createObjectURL(file);
    videoPreview.src = videoPreviewUrl;
    videoFilename.textContent = file.name;
    videoFilesize.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
    videoPreviewBox.classList.remove('hidden');
    videoDropZone.classList.add('hidden');
    updateDetectButton();
  }

  videoFileInput.addEventListener('change', event => handleVideoSelect(event.target.files && event.target.files[0]));
  btnClearVideo.addEventListener('click', () => {
    if (isVideoActive()) return;
    clearVideoSelection();
    videoFileInput.click();
  });
  videoDropZone.addEventListener('keydown', event => {
    if (event.target === videoFileInput) return;
    if ((event.key === 'Enter' || event.key === ' ') && !isVideoActive()) {
      event.preventDefault();
      videoFileInput.click();
    }
  });
  ['dragenter', 'dragover'].forEach(name => videoDropZone.addEventListener(name, event => {
    event.preventDefault();
    if (!isVideoActive()) videoDropZone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(name => videoDropZone.addEventListener(name, event => {
    event.preventDefault();
    videoDropZone.classList.remove('dragover');
  }));
  videoDropZone.addEventListener('drop', event => handleVideoSelect(event.dataTransfer && event.dataTransfer.files[0]));
  videoPreview.addEventListener('loadedmetadata', () => {
    if (!currentVideoFile) return;
    if (Number.isFinite(videoPreview.duration) && videoPreview.duration > maxVideoSeconds) {
      clearVideoSelection();
      showToast(`Video dài quá ${Math.round(maxVideoSeconds / 60)} phút. Chọn một đoạn ngắn hơn.`, 'error', 6500);
      return;
    }
    if (Number.isFinite(videoPreview.duration)) {
      selectedVideoDuration = videoPreview.duration;
      videoFilesize.textContent = `${(currentVideoFile.size / 1024 / 1024).toFixed(1)} MB · ${videoPreview.duration.toFixed(1)} giây`;
    }
    updateDetectButton();
  });
  videoPreview.addEventListener('error', () => {
    if (currentVideoFile) showToast('Trình duyệt chưa phát được bản gốc; vẫn có thể xử lý video.', 'info', 6500);
  });
  videoSampleFps.addEventListener('change', updateDetectButton);

  function jobUrl(value, fallback) {
    const url = value || fallback;
    if (typeof url !== 'string' || !url.startsWith('/api/video/jobs/') || url.includes('..')) {
      throw new Error('Máy chủ trả về đường dẫn video không hợp lệ.');
    }
    return url;
  }

  function renderVideoJob(job) {
    const knownStates = ['queued', 'processing', 'encoding', 'done', 'error', 'cancelled'];
    if (!job || !job.id || !knownStates.includes(job.state)) throw new Error('Trạng thái video không hợp lệ.');
    if (videoJob && videoJob.id !== job.id) throw new Error('Trạng thái trả về không khớp video đang xử lý.');
    if (videoJob && ['done', 'error', 'cancelled'].includes(videoJob.state)
        && ['queued', 'processing', 'encoding'].includes(job.state)) return;
    if (videoJob && ['queued', 'processing', 'encoding'].includes(job.state)
        && Number(job.processed_frames) < Number(videoJob.processed_frames)) return;
    videoJob = { ...videoJob, ...job };
    const active = isVideoActive();
    const stateNames = { queued: 'Đang chuẩn bị video…', processing: 'Đang phân tích các frame…',
      encoding: 'Đang hoàn tất video MP4…', done: 'Video đã xử lý xong',
      error: 'Chưa xử lý được video', cancelled: 'Đã dừng xử lý video' };
    videoEmptyState.classList.add('hidden');
    videoJobProgress.classList.remove('hidden');
    videoJobState.textContent = stateNames[videoJob.state];
    const progress = videoJob.state === 'done' ? 100 : Math.min(100, Math.max(0, Number(videoJob.progress) || 0));
    videoProgressBar.value = progress;
    const processed = Number(videoJob.processed_frames) || 0;
    const total = Number(videoJob.total_frames) || 0;
    videoJobDetail.textContent = total > 0
      ? `${processed}/${total} frame lấy mẫu · ${Math.round(progress)}%`
      : 'Đang đọc thông tin video…';
    const metric = videoJob.last_metric;
    videoLastMetric.textContent = '';
    if (metric && Number.isFinite(metric.timestamp_seconds)) {
      const prefix = `Frame tại ${metric.timestamp_seconds.toFixed(2)} giây`;
      if (videoJob.model_type === 'anomalyclip' && Number.isFinite(metric.anomaly_score)) {
        videoLastMetric.textContent = `${prefix} · Điểm bất thường: ${metric.anomaly_score.toFixed(3)}`;
      } else if (Number.isFinite(metric.count)) {
        videoLastMetric.textContent = `${prefix} · Số PC nhận diện: ${metric.count}`;
      }
    }
    btnCancelVideo.classList.toggle('hidden', !active);
    btnCancelVideo.disabled = videoCancelling;
    btnCancelVideo.textContent = videoCancelling ? 'Đang dừng…' : 'Dừng xử lý';
    videoJobError.classList.toggle('hidden', videoJob.state !== 'error');
    if (videoJob.state === 'error') videoJobError.textContent = videoJob.error || 'Không xử lý được video. Vui lòng thử lại.';
    if (!active) {
      videoCancelling = false;
      btnRetryVideoStatus.classList.add('hidden');
    }
    if (videoJob.state === 'done') {
      videoResult.src = jobUrl(videoJob.result_url);
      videoDownloadLink.href = jobUrl(videoJob.download_url, videoJob.result_url);
      const modelName = videoJob.model_type === 'anomalyclip' ? 'AnomalyCLIP' : 'YOLO11';
      videoResultSummary.textContent = `${modelName} · ${processed} frame đã phân tích · ${videoJob.sample_fps} frame/giây · Không có âm thanh`;
      videoCompletedResult.classList.remove('hidden');
      videoCsvLink.classList.toggle('hidden', !videoJob.csv_url);
      if (videoJob.csv_url) videoCsvLink.href = jobUrl(videoJob.csv_url);
    }
    setVideoControlsLocked(active);
    detectSpinner.classList.toggle('hidden', !active);
    detectBtnText.textContent = active ? 'Đang xử lý video…' : actionLabel();
  }

  function scheduleVideoPoll() {
    if (videoPollTimer !== null) clearTimeout(videoPollTimer);
    if (isVideoActive() && !videoSubmitting) videoPollTimer = setTimeout(pollVideoJob, 1000);
  }

  async function pollVideoJob() {
    videoPollTimer = null;
    if (!videoJob || !isVideoActive() || videoSubmitting) return;
    try {
      const response = await fetch(jobUrl(videoJob.status_url, `/api/video/jobs/${videoJob.id}`));
      if (response.status === 404) {
        renderVideoJob({ ...videoJob, state: 'error', error: 'Không còn tìm thấy công việc này. Máy chủ có thể đã khởi động lại; hãy gửi lại video.' });
        return;
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Không nhận được trạng thái video.');
      videoPollErrors = 0;
      videoJobError.classList.add('hidden');
      btnRetryVideoStatus.classList.add('hidden');
      renderVideoJob(data);
      scheduleVideoPoll();
    } catch (error) {
      videoPollErrors += 1;
      if (videoPollErrors < 3) {
        scheduleVideoPoll();
      } else {
        videoJobState.textContent = 'Chưa nhận được trạng thái mới';
        videoJobError.textContent = 'Video có thể vẫn đang chạy trên máy. Kiểm tra lại kết nối rồi bấm kiểm tra trạng thái; không cần gửi lại ngay.';
        videoJobError.classList.remove('hidden');
        btnRetryVideoStatus.classList.remove('hidden');
      }
    }
  }

  btnRetryVideoStatus.addEventListener('click', () => {
    if (!videoJob || !isVideoActive()) return;
    videoPollErrors = 0;
    videoJobError.classList.add('hidden');
    btnRetryVideoStatus.classList.add('hidden');
    pollVideoJob();
  });

  btnCancelVideo.addEventListener('click', async () => {
    if (!videoJob || !isVideoActive() || videoCancelling) return;
    videoCancelling = true;
    btnCancelVideo.disabled = true;
    btnCancelVideo.textContent = 'Đang dừng…';
    try {
      const response = await fetch(`/api/video/jobs/${encodeURIComponent(videoJob.id)}/cancel`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Chưa dừng được video.');
      renderVideoJob(data);
      scheduleVideoPoll();
    } catch (error) {
      videoCancelling = false;
      btnCancelVideo.disabled = false;
      btnCancelVideo.textContent = 'Dừng xử lý';
      showToast(error.message || 'Chưa kết nối được để dừng video.', 'error', 6500);
    }
  });

  async function performVideoDetection() {
    if (isVideoActive() || isProcessing || !currentVideoFile) return;
    const issue = videoUnavailableMessage();
    if (issue) {
      showToast(issue, 'error', 6500);
      checkHealth();
      return;
    }
    if (videoPollTimer !== null) clearTimeout(videoPollTimer);
    videoPollTimer = null;
    videoJob = null;
    videoSubmitting = true;
    videoCancelling = false;
    videoPollErrors = 0;
    videoPreview.pause();
    videoResult.pause();
    videoCompletedResult.classList.add('hidden');
    videoEmptyState.classList.add('hidden');
    videoJobProgress.classList.remove('hidden');
    videoJobError.classList.add('hidden');
    btnRetryVideoStatus.classList.add('hidden');
    btnCancelVideo.classList.add('hidden');
    videoJobState.textContent = 'Đang tải video lên…';
    videoJobDetail.textContent = 'Đang gửi video đến máy chủ trên máy này.';
    videoLastMetric.textContent = '';
    videoProgressBar.value = 0;
    setVideoControlsLocked(true);
    detectSpinner.classList.remove('hidden');
    detectBtnText.textContent = 'Đang tải video lên…';

    const formData = new FormData();
    formData.append('video', currentVideoFile, currentVideoFile.name);
    formData.append('model_type', currentModel);
    formData.append('sample_fps', videoSampleFps.value);
    if (currentModel === 'anomalyclip') {
      formData.append('anomaly_imgsz', anomalyImgszSelect.value);
      formData.append('anomaly_opacity', anomalyOpacityRange.value);
    } else if (currentModel === 'grounding_dino') {
      formData.append('dino_prompt', (dinoPromptInput && dinoPromptInput.value.trim()) || 'pc case');
      formData.append('dino_box_thresh', dinoBoxThreshRange ? dinoBoxThreshRange.value : '0.35');
      formData.append('dino_text_thresh', '0.25');
    } else if (currentModel === 'loca') {
      formData.append('loca_mode', locaMode || 'zero_shot');
    } else {
      formData.append('confidence', confidenceRange.value);
      formData.append('imgsz', imgszSelect.value);
      formData.append('nms_iou', yoloNmsRange.value);
      formData.append('sliced', checkSliced.checked ? 'true' : 'false');
      formData.append('enhance', checkEnhance.checked ? 'true' : 'false');
      formData.append('use_split', checkUseSplit.checked ? 'true' : 'false');
      formData.append('use_roi', checkUseRoi.checked ? 'true' : 'false');
    }
    try {
      const response = await fetch('/api/video/jobs', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Không gửi được video để xử lý.');
      videoSubmitting = false;
      renderVideoJob(data);
      scheduleVideoPoll();
    } catch (error) {
      videoSubmitting = false;
      videoJob = null;
      setVideoControlsLocked(false);
      detectSpinner.classList.add('hidden');
      detectBtnText.textContent = actionLabel();
      videoJobState.textContent = 'Chưa gửi được video';
      videoJobError.textContent = error.message || 'Không kết nối được máy chủ video. Vui lòng thử lại.';
      videoJobError.classList.remove('hidden');
    }
  }

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
    mainDrawCanvas.classList.toggle('drawing-disabled', currentModel !== 'loca' || locaMode !== 'few_shot');

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
    if (currentModel === 'anomalyclip'
        && !(lastHealth && lastHealth.anomalyclip && lastHealth.anomalyclip.ready === true)) {
      showToast('AnomalyCLIP chưa sẵn sàng. Hoàn tất cài đặt hoặc khởi động lại máy chủ web rồi tải lại trang.', 'error', 6500);
      checkHealth();
      return;
    }
    const requestedModel = currentModel;
    const requestedFile = currentFile;
    const requestedVersion = selectionVersion;

    const isFixed = checkLocaFixedExemplars && checkLocaFixedExemplars.checked;
    if (currentModel === 'loca' && locaMode === 'few_shot' && !isFixed && exemplarBoxes.length === 0) {
      showToast('Vui lòng Click trên ảnh để chọn ít nhất 1 mẫu, hoặc bật "Dùng 3 khung mẫu chuẩn cố định"!', 'error', 4500);
      return;
    }

    isProcessing = true;
    btnDetect.disabled = true;
    detectSpinner.classList.remove('hidden');
    statsWrapper.classList.add('hidden');
    resultActions.classList.add('hidden');
    annotatedDataUrl = null;
    imageAnnotated.classList.add('hidden');
    imageOriginal.classList.remove('hidden');

    if (requestedModel === 'anomalyclip') {
      detectBtnText.textContent = 'AnomalyCLIP đang phân tích...';
    } else if (requestedModel === 'loca') {
      detectBtnText.textContent = 'LOCA đang tích phân mật độ...';
    } else if (requestedModel === 'grounding_dino') {
      detectBtnText.textContent = 'Grounding DINO đang nhận diện...';
    } else {
      detectBtnText.textContent = 'YOLO11 đang quét nhận diện...';
    }

    const formData = new FormData();
    formData.append('image', requestedFile, requestedFile.name || 'image.jpg');
    formData.append('model_type', requestedModel);

    // Universal shelf pre-processing
    if (requestedModel !== 'anomalyclip') {
      if (checkUseSplit) formData.append('use_split', checkUseSplit.checked ? 'true' : 'false');
      if (checkUseRoi) formData.append('use_roi', checkUseRoi.checked ? 'true' : 'false');
    }

    if (requestedModel === 'anomalyclip') {
      formData.append('anomaly_imgsz', anomalyImgszSelect.value);
      formData.append('anomaly_opacity', anomalyOpacityRange.value);
    } else if (requestedModel === 'loca') {
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
    } else if (requestedModel === 'grounding_dino') {
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
      if (requestedModel === 'anomalyclip' && data.model_type !== 'anomalyclip') {
        throw new Error('Máy chủ chưa hỗ trợ AnomalyCLIP. Khởi động lại máy chủ web sau khi cập nhật rồi tải lại trang.');
      }
      if (requestedVersion !== selectionVersion || requestedFile !== currentFile) return;

      renderResults(data);
      const countMsg = data.model_type === 'anomalyclip'
        ? `Điểm bất thường: ${Number(data.anomaly_score).toFixed(3)}`
        : data.model_type === 'loca'
          ? `Đếm mật độ: ${data.count} (Tổng: ${data.raw_count})`
          : `Tìm thấy: ${data.count} đối tượng`;
      showToast(`Hoàn tất (${data.model})! ${countMsg}.`, 'success');
    } catch (err) {
      if (requestedVersion === selectionVersion) {
        showToast(err.message || 'Lỗi khi gửi yêu cầu tới máy chủ.', 'error', 6500);
      }
    } finally {
      isProcessing = false;
      updateDetectButton();
      detectSpinner.classList.add('hidden');
      detectBtnText.textContent = actionLabel();
    }
  }

  btnDetect.addEventListener('click', () => activeInput === 'video' ? performVideoDetection() : performDetection());

  // ----------------------------------------------------
  // Render Results
  // ----------------------------------------------------
  function renderResults(data) {
    statsWrapper.classList.remove('hidden');
    resultActions.classList.remove('hidden');

    const isAnomaly = data.model_type === 'anomalyclip';
    lastResultModel = data.model_type;
    detectionsTableContainer.classList.toggle('hidden', isAnomaly);
    anomalyResultInfo.classList.toggle('hidden', !isAnomaly);
    statCount.textContent = isAnomaly ? Number(data.anomaly_score).toFixed(3) : data.count;
    statTime.textContent = `${data.inference_ms} ms`;
    statRes.textContent = `${data.device || 'CPU'} • ${data.width}×${data.height}`;

    if (isAnomaly) {
      statLabelCount.textContent = 'Điểm bất thường';
      statLabelConf.textContent = 'Ý nghĩa điểm';
      statConf.textContent = 'Tham khảo';
      btnViewAnnotated.textContent = 'Bản đồ bất thường';
      workspaceBanner.classList.add('hidden');
    } else if (data.model_type === 'loca') {
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
    if (!isAnomaly && data.detections && data.detections.length > 0) {
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
    } else if (!isAnomaly) {
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
    a.download = `${lastResultModel === 'anomalyclip' ? 'anomalyclip_heatmap' : 'counted'}_${Date.now()}.jpg`;
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
