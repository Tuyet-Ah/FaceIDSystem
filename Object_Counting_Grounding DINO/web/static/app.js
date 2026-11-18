document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const statusPill = document.getElementById("status-pill");
  const modelPill = document.getElementById("model-pill");
  const devicePill = document.getElementById("device-pill");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  const webcamVideo = document.getElementById("webcam-video");
  const btnStartCamera = document.getElementById("btn-start-camera");
  const btnCaptureCamera = document.getElementById("btn-capture-camera");
  let webcamStream = null;

  const samplesList = document.getElementById("samples-list");

  const promptInput = document.getElementById("prompt-input");
  const chips = document.querySelectorAll(".chip");

  const boxThresholdSlider = document.getElementById("box-threshold");
  const boxThresholdVal = document.getElementById("box-threshold-val");
  const textThresholdSlider = document.getElementById("text-threshold");
  const textThresholdVal = document.getElementById("text-threshold-val");
  const roiInput = document.getElementById("roi-input");

  const btnCount = document.getElementById("btn-count");
  const countSpinner = document.getElementById("count-spinner");
  const countBtnText = document.getElementById("count-btn-text");

  const metricCount = document.getElementById("metric-count");
  const metricTime = document.getElementById("metric-time");
  const placeholder = document.getElementById("placeholder");
  const resultImg = document.getElementById("result-img");
  const detailsSection = document.getElementById("details-section");
  const detectionsTbody = document.getElementById("detections-tbody");

  // State
  let currentFile = null;
  let currentBase64 = null;
  let currentSampleName = null;

  // 1. Health check & Initial info
  fetch("/api/health")
    .then(res => res.json())
    .then(data => {
      if (data.status === "ok") {
        modelPill.textContent = data.model_id;
        devicePill.textContent = (data.device || "CPU").toUpperCase();
        statusPill.textContent = "Sẵn sàng (Ready)";
        statusPill.className = "pill pill-status status-ready";
      }
    })
    .catch(() => {
      statusPill.textContent = "Mất kết nối";
      statusPill.className = "pill pill-status status-busy";
    });

  // 2. Fetch Sample Images
  fetch("/api/samples")
    .then(res => res.json())
    .then(samples => {
      samplesList.innerHTML = "";
      if (!samples || samples.length === 0) {
        samplesList.innerHTML = "<p class='subtext'>Không có ảnh mẫu nào</p>";
        return;
      }
      samples.forEach(sample => {
        const item = document.createElement("div");
        item.className = "sample-item";
        item.innerHTML = `
          <img src="${sample.url}" alt="${sample.name}">
          <span>${sample.name}</span>
        `;
        item.addEventListener("click", () => {
          document.querySelectorAll(".sample-item").forEach(el => el.classList.remove("selected"));
          item.classList.add("selected");
          currentSampleName = sample.name;
          currentFile = null;
          currentBase64 = null;
          btnCount.disabled = false;
          // Trigger count right away on sample select
          triggerCount();
        });
        samplesList.appendChild(item);
      });
    });

  // 3. Tab Switching
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${btn.dataset.tab}`);
      if (targetPane) targetPane.classList.add("active");

      if (btn.dataset.tab !== "camera" && webcamStream) {
        stopCamera();
      }
    });
  });

  // 4. File Upload (Drag & Drop & Browse)
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  function handleSelectedFile(file) {
    currentFile = file;
    currentBase64 = null;
    currentSampleName = null;
    btnCount.disabled = false;

    // Preview uploaded file
    const reader = new FileReader();
    reader.onload = (e) => {
      placeholder.classList.add("hidden");
      resultImg.classList.remove("hidden");
      resultImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // 5. Webcam Support
  btnStartCamera.addEventListener("click", async () => {
    if (webcamStream) {
      stopCamera();
      return;
    }
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      webcamVideo.srcObject = webcamStream;
      btnStartCamera.textContent = "Tắt Camera";
      btnCaptureCamera.disabled = false;
    } catch (err) {
      alert("Không thể mở Webcam: " + err.message);
    }
  });

  btnCaptureCamera.addEventListener("click", () => {
    if (!webcamStream) return;
    const canvas = document.createElement("canvas");
    canvas.width = webcamVideo.videoWidth || 640;
    canvas.height = webcamVideo.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

    currentBase64 = canvas.toDataURL("image/jpeg", 0.9);
    currentFile = null;
    currentSampleName = null;
    btnCount.disabled = false;

    // Show preview
    placeholder.classList.add("hidden");
    resultImg.classList.remove("hidden");
    resultImg.src = currentBase64;

    triggerCount();
  });

  function stopCamera() {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      webcamStream = null;
      webcamVideo.srcObject = null;
      btnStartCamera.textContent = "Bật Camera";
      btnCaptureCamera.disabled = true;
    }
  }

  // 6. Prompt Chips
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      promptInput.value = chip.dataset.prompt;
    });
  });

  // 7. Sliders
  boxThresholdSlider.addEventListener("input", (e) => {
    boxThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  textThresholdSlider.addEventListener("input", (e) => {
    textThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // 8. Submit Count
  btnCount.addEventListener("click", () => {
    triggerCount();
  });

  async function triggerCount() {
    if (!currentFile && !currentBase64 && !currentSampleName) {
      alert("Vui lòng tải ảnh, chọn ảnh mẫu hoặc chụp từ Webcam trước!");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("prompt", promptInput.value.trim());
    formData.append("box_threshold", boxThresholdSlider.value);
    formData.append("text_threshold", textThresholdSlider.value);
    formData.append("roi", roiInput.value.trim());

    if (currentFile) {
      formData.append("file", currentFile);
    } else if (currentBase64) {
      formData.append("image_base64", currentBase64);
    } else if (currentSampleName) {
      formData.append("sample_name", currentSampleName);
    }

    try {
      const resp = await fetch("/api/count", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();
      if (!resp.ok || data.error) {
        throw new Error(data.error || "Lỗi khi xử lý");
      }

      // Render results
      metricCount.textContent = data.count;
      metricTime.textContent = `${data.inference_time_ms} ms`;

      if (data.annotated_image) {
        placeholder.classList.add("hidden");
        resultImg.classList.remove("hidden");
        resultImg.src = data.annotated_image;
      }

      // Fill table
      detectionsTbody.innerHTML = "";
      if (data.detections && data.detections.length > 0) {
        detailsSection.classList.remove("hidden");
        data.detections.forEach((det, idx) => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td><strong>${idx + 1}</strong></td>
            <td><span class="badge-class">${escapeHtml(det.class_name)}</span></td>
            <td><strong>${(det.confidence * 100).toFixed(1)}%</strong></td>
            <td><code>[${det.xyxy.join(", ")}]</code></td>
          `;
          detectionsTbody.appendChild(row);
        });
      } else {
        detailsSection.classList.remove("hidden");
        detectionsTbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#94a3b8">Không tìm thấy đối tượng nào khớp với prompt</td></tr>`;
      }
    } catch (err) {
      alert("Đã xảy ra lỗi: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    if (loading) {
      btnCount.disabled = true;
      countSpinner.classList.remove("hidden");
      countBtnText.textContent = "Đang suy luận (Detecting)...";
      statusPill.textContent = "Đang xử lý...";
      statusPill.className = "pill pill-status status-busy";
    } else {
      btnCount.disabled = false;
      countSpinner.classList.add("hidden");
      countBtnText.textContent = "Bắt đầu đếm (Detect & Count)";
      statusPill.textContent = "Sẵn sàng (Ready)";
      statusPill.className = "pill pill-status status-ready";
    }
  }

  function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag));
  }
});
