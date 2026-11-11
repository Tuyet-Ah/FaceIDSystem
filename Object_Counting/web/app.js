/**
 * PC Case Detection & Counting Web Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentFile = null;
  let currentPreviewUrl = null;
  let annotatedDataUrl = null;
  let mediaStream = null;
  let isProcessing = false;
  let maxUploadBytes = 64 * 1024 * 1024;
  let maxUploadMb = 64;

  // DOM Elements
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');
  const dropFormatsText = document.getElementById('drop-formats-text');

  // Tabs
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

  // Webcam
  const cameraFeed = document.getElementById('camera-feed');
  const cameraOverlay = document.getElementById('camera-overlay');
  const cameraErrorText = document.getElementById('camera-error-text');
  const btnSnapCamera = document.getElementById('btn-snap-camera');
  const btnToggleCamera = document.getElementById('btn-toggle-camera');

  // Sample
  const btnLoadSample = document.getElementById('btn-load-sample');

  // Inference Parameters
  const imgszSelect = document.getElementById('imgsz-select');
  const confidenceRange = document.getElementById('confidence-range');
  const confidenceVal = document.getElementById('confidence-val');

  // Detect Action
  const btnDetect = document.getElementById('btn-detect');
  const detectSpinner = document.getElementById('detect-spinner');
  const detectBtnText = document.getElementById('detect-btn-text');

  // Results
  const emptyState = document.getElementById('empty-state');
  const resultContent = document.getElementById('result-content');
  const resultActions = document.getElementById('result-actions');
  const statCount = document.getElementById('stat-count');
  const statTime = document.getElementById('stat-time');
  const statRes = document.getElementById('stat-res');
  const statConf = document.getElementById('stat-conf');
  const imageAnnotated = document.getElementById('image-annotated');
  const imageOriginal = document.getElementById('image-original');
  const tableCount = document.getElementById('table-count');
  const detectionsTbody = document.getElementById('detections-tbody');
  const btnViewAnnotated = document.getElementById('btn-view-annotated');
  const btnViewOriginal = document.getElementById('btn-view-original');
  const btnDownload = document.getElementById('btn-download');
  const btnFullscreen = document.getElementById('btn-fullscreen');

  // Modal
  const modalZoom = document.getElementById('modal-zoom');
  const modalImg = document.getElementById('modal-img');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const modalBackdrop = document.getElementById('modal-backdrop');

  // Toast Container
  const toastContainer = document.getElementById('toast-container');

  // ----------------------------------------------------
  // Notifications
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
  // Health Check
  // ----------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('API server returned error');
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
        statusText.textContent = `Mô hình: ${data.model} (Sẵn sàng)`;
      } else {
        statusPill.className = 'status-pill status-error';
        statusText.textContent = 'Chưa nạp weights (best.pt)';
      }
    } catch (err) {
      statusPill.className = 'status-pill status-error';
      statusText.textContent = 'Mất kết nối server';
    }
  }

  // ----------------------------------------------------
  // Tab Management
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
  // File Selection & Preview Handling
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
      showToast(`Kích thước ảnh quá lớn. Vui lòng chọn ảnh dưới ${maxUploadMb} MB.`, 'error');
      return;
    }

    currentFile = file;

    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    previewImage.src = currentPreviewUrl;
    previewFilename.textContent = file.name || 'image.jpg';
    previewFilesize.textContent = formatBytes(file.size);

    previewBox.classList.remove('hidden');
    btnDetect.disabled = false;
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
  }

  btnRemoveImage.addEventListener('click', clearSelectedFile);

  // Drag & Drop
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
  // Confidence Slider
  // ----------------------------------------------------
  confidenceRange.addEventListener('input', e => {
    const val = parseFloat(e.target.value).toFixed(2);
    const pct = Math.round(parseFloat(e.target.value) * 100);
    confidenceVal.textContent = `${val} (${pct}%)`;
  });

  // ----------------------------------------------------
  // Detection Execution
  // ----------------------------------------------------
  async function performDetection() {
    if (!currentFile || isProcessing) return;

    isProcessing = true;
    btnDetect.disabled = true;
    detectSpinner.classList.remove('hidden');
    detectBtnText.textContent = 'Đang nhận diện...';

    const formData = new FormData();
    formData.append('image', currentFile, currentFile.name || 'image.jpg');
    formData.append('confidence', confidenceRange.value);
    if (imgszSelect) {
      formData.append('imgsz', imgszSelect.value);
    }

    try {
      const startTime = performance.now();
      const res = await fetch('/api/detect', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || 'Nhận diện thất bại.');
      }

      renderResults(data);
      showToast(`Hoàn tất! Tìm thấy ${data.count} case máy tính.`, 'success');
    } catch (err) {
      showToast(err.message || 'Lỗi kết nối tới máy chủ.', 'error');
    } finally {
      isProcessing = false;
      btnDetect.disabled = false;
      detectSpinner.classList.add('hidden');
      detectBtnText.textContent = 'Nhận diện & Đếm Case';
    }
  }

  btnDetect.addEventListener('click', performDetection);

  // ----------------------------------------------------
  // Render Results
  // ----------------------------------------------------
  function renderResults(data) {
    emptyState.classList.add('hidden');
    resultContent.classList.remove('hidden');
    resultActions.classList.remove('hidden');

    statCount.textContent = data.count;
    statTime.textContent = `${data.inference_ms} ms`;
    statRes.textContent = `${data.width} × ${data.height} (${data.imgsz || 960}p)`;

    let avgConf = 0;
    if (data.detections && data.detections.length > 0) {
      const totalConf = data.detections.reduce((sum, d) => sum + (d.confidence || 0), 0);
      avgConf = Math.round((totalConf / data.detections.length) * 100);
    }
    statConf.textContent = data.detections.length > 0 ? `${avgConf}%` : '0%';

    annotatedDataUrl = data.image;
    imageAnnotated.src = annotatedDataUrl;
    imageOriginal.src = currentPreviewUrl;

    // Default view: annotated
    setViewMode('annotated');

    // Populate Detections Table
    tableCount.textContent = data.count;
    detectionsTbody.innerHTML = '';

    if (!data.detections || data.detections.length === 0) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 4;
      cell.textContent = 'Không có case nào vượt qua ngưỡng tin cậy này. Thử kéo giảm ngưỡng.';
      cell.className = 'text-dim';
      row.appendChild(cell);
      detectionsTbody.appendChild(row);
      return;
    }

    data.detections.forEach((item, index) => {
      const row = document.createElement('tr');

      // Col 1: STT
      const tdIdx = document.createElement('td');
      tdIdx.textContent = index + 1;
      row.appendChild(tdIdx);

      // Col 2: Class
      const tdClass = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = 'badge-class';
      badge.textContent = item.class_name || 'pc_case';
      tdClass.appendChild(badge);
      row.appendChild(tdClass);

      // Col 3: Confidence
      const tdConf = document.createElement('td');
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
      row.appendChild(tdConf);

      // Col 4: Coordinates
      const tdCoords = document.createElement('td');
      const code = document.createElement('span');
      code.className = 'coords-code';
      const [x1, y1, x2, y2] = item.xyxy.map(v => Math.round(v));
      code.textContent = `[${x1}, ${y1}, ${x2}, ${y2}]`;
      tdCoords.appendChild(code);
      row.appendChild(tdCoords);

      detectionsTbody.appendChild(row);
    });
  }

  // ----------------------------------------------------
  // View Switch & Download & Modal
  // ----------------------------------------------------
  function setViewMode(mode) {
    if (mode === 'annotated') {
      btnViewAnnotated.classList.add('active');
      btnViewOriginal.classList.remove('active');
      imageAnnotated.classList.remove('hidden');
      imageOriginal.classList.add('hidden');
    } else {
      btnViewOriginal.classList.add('active');
      btnViewAnnotated.classList.remove('active');
      imageOriginal.classList.remove('hidden');
      imageAnnotated.classList.add('hidden');
    }
  }

  btnViewAnnotated.addEventListener('click', () => setViewMode('annotated'));
  btnViewOriginal.addEventListener('click', () => setViewMode('original'));

  btnDownload.addEventListener('click', () => {
    if (!annotatedDataUrl) return;
    const a = document.createElement('a');
    a.href = annotatedDataUrl;
    a.download = `detected_case_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  function openModal(imgSrc) {
    modalImg.src = imgSrc;
    modalZoom.classList.remove('hidden');
  }

  function closeModal() {
    modalZoom.classList.add('hidden');
    modalImg.src = '';
  }

  btnFullscreen.addEventListener('click', () => {
    const isOriginal = btnViewOriginal.classList.contains('active');
    const src = isOriginal ? currentPreviewUrl : annotatedDataUrl;
    if (src) openModal(src);
  });

  imageAnnotated.addEventListener('click', () => {
    if (annotatedDataUrl) openModal(annotatedDataUrl);
  });

  imageOriginal.addEventListener('click', () => {
    if (currentPreviewUrl) openModal(currentPreviewUrl);
  });

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
