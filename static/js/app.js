(function () {
  "use strict";

  /* ==========================================================================
   * 1. Animated neural-network canvas background
   *    - lightweight, GPU-friendly, honors prefers-reduced-motion
   * ========================================================================== */
  var NeuralBackground = (function () {
    var canvas = document.getElementById("neuralCanvas");
    if (!canvas) return;

    var ctx = canvas.getContext("2d");
    if (!ctx) return;

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var width = 0;
    var height = 0;
    var nodes = [];
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var rafId = null;

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      buildNodes();
    }

    function buildNodes() {
      var target = reduceMotion ? 0 : Math.max(20, Math.min(70, Math.floor((width * height) / 22000)));
      nodes = [];
      for (var i = 0; i < target; i++) {
        nodes.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.35,
          r: Math.random() * 1.6 + 0.6,
        });
      }
    }

    function step() {
      ctx.clearRect(0, 0, width, height);

      var i, j, a, b, dx, dy, dist;
      var linkDist = Math.min(150, Math.max(90, width / 9));

      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        a.x += a.vx;
        a.y += a.vy;
        if (a.x < -20) a.x = width + 20;
        if (a.x > width + 20) a.x = -20;
        if (a.y < -20) a.y = height + 20;
        if (a.y > height + 20) a.y = -20;

        ctx.beginPath();
        ctx.arc(a.x, a.y, a.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(120, 180, 255, 0.5)";
        ctx.fill();
      }

      for (i = 0; i < nodes.length; i++) {
        a = nodes[i];
        for (j = i + 1; j < nodes.length; j++) {
          b = nodes[j];
          dx = a.x - b.x;
          dy = a.y - b.y;
          dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < linkDist) {
            var alpha = (1 - dist / linkDist) * 0.28;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = "rgba(99, 140, 255, " + alpha + ")";
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      rafId = window.requestAnimationFrame(step);
    }

    function start() {
      if (reduceMotion) {
        resize();
        for (var i = 0; i < nodes.length; i++) {
          var n = nodes[i];
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(120, 180, 255, 0.35)";
          ctx.fill();
        }
        return;
      }
      step();
    }

    window.addEventListener("resize", function () {
      if (rafId) window.cancelAnimationFrame(rafId);
      resize();
      if (!reduceMotion) step();
    });

    resize();
    start();
  })();

  /* ==========================================================================
   * 2. Recognition workspace logic (webcam primary, image upload secondary)
   * ========================================================================== */
  var fileInput = document.getElementById("fileInput");
  var dropZone = document.getElementById("dropZone");
  var previewWrap = document.getElementById("previewWrap");
  var preview = document.getElementById("preview");
  var clearBtn = document.getElementById("clearBtn");
  var chooseBtn = document.getElementById("chooseBtn");
  var resetBtn = document.getElementById("resetBtn");

  var startCameraBtn = document.getElementById("startCameraBtn");
  var pauseCameraBtn = document.getElementById("pauseCameraBtn");
  var stopCameraBtn = document.getElementById("stopCameraBtn");
  var cameraStage = document.getElementById("cameraStage");
  var cameraVideo = document.getElementById("cameraVideo");
  var captureCanvas = document.getElementById("captureCanvas");
  var cameraPlaceholder = document.getElementById("cameraPlaceholder");
  var cameraMessage = document.getElementById("cameraMessage");
  var cameraStatus = document.getElementById("cameraStatus");
  var cameraStatusText = document.getElementById("cameraStatusText");
  var detectionStatus = document.getElementById("detectionStatus");

  var predictBtn = document.getElementById("predictBtn");
  var predictSpinner = document.getElementById("predictSpinner");
  var predictLabel = document.getElementById("predictLabel");
  var predictHint = document.getElementById("predictHint");

  var resultPlaceholder = document.getElementById("resultPlaceholder");
  var resultContent = document.getElementById("resultContent");
  var predictionLetter = document.getElementById("predictionLetter");
  var predictionConfidence = document.getElementById("predictionConfidence");
  var confidenceFill = document.getElementById("confidenceFill");
  var predictionLanguage = document.getElementById("predictionLanguage");
  var topK = document.getElementById("topK");
  var errorMessage = document.getElementById("errorMessage");

  var languageHint = document.getElementById("languageHint");
  var languageButtons = document.querySelectorAll(".segmented__btn");

  var MAX_SIZE = 8 * 1024 * 1024; // matches Flask MAX_CONTENT_LENGTH
  var ALLOWED_TYPES = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
  var selectedFile = null;
  var currentLanguage = "ASL";
  var stream = null;
  var predicting = false;
  var cameraActive = false;
  var recognitionPaused = false;

  var LANGUAGE_HINTS = {
    ASL: "ASL uses one-hand MediaPipe landmarks + Random Forest (A–Z).",
    ISL: "ISL uses one/two-hand MediaPipe landmarks + Random Forest (A–Z).",
  };

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
  }

  function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
  }

  function showCameraMessage(message, isError) {
    cameraMessage.textContent = message;
    cameraMessage.classList.remove("hidden");
    cameraMessage.classList.toggle("camera-message--error", !!isError);
  }

  function clearCameraMessage() {
    cameraMessage.textContent = "";
    cameraMessage.classList.add("hidden");
    cameraMessage.classList.remove("camera-message--error");
  }

  function setLanguage(language) {
    if (currentLanguage !== language) {
      // Reset any in-flight/history-sensitive state when the mode changes.
      resetResult();
      if (detectionStatus) hideDetectionStatus();
    }
    currentLanguage = language;
    languageButtons.forEach(function (btn) {
      var active = btn.getAttribute("data-language") === language;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-checked", active ? "true" : "false");
    });
    if (languageHint) {
      languageHint.textContent = LANGUAGE_HINTS[language] || "";
    }
    if (predictionLanguage) {
      predictionLanguage.textContent = "Mode: " + language;
    }
  }

  languageButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      setLanguage(btn.getAttribute("data-language"));
    });
    btn.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        var buttons = Array.prototype.slice.call(languageButtons);
        var idx = buttons.indexOf(btn);
        var next =
          e.key === "ArrowRight"
            ? (idx + 1) % buttons.length
            : (idx - 1 + buttons.length) % buttons.length;
        buttons[next].focus();
        setLanguage(buttons[next].getAttribute("data-language"));
      }
    });
  });

  /* --------------------------------------------------------------------------
   * Webcam
   * ------------------------------------------------------------------------ */
  function updateDetectionStatus(message, kind) {
    if (!detectionStatus) return;
    if (!message) {
      detectionStatus.classList.add("hidden");
      detectionStatus.classList.remove("is-detected", "is-missing");
      detectionStatus.textContent = "";
      return;
    }
    detectionStatus.classList.remove("hidden");
    detectionStatus.classList.toggle("is-detected", kind === "detected");
    detectionStatus.classList.toggle("is-missing", kind === "missing");
    detectionStatus.textContent = message;
  }

  function hideDetectionStatus() {
    updateDetectionStatus(null);
  }

  function updateCameraStatus() {
    if (!cameraStatus || !cameraStatusText) return;
    if (!cameraActive) {
      cameraStatus.classList.add("hidden");
      return;
    }
    cameraStatus.classList.remove("hidden");
    if (recognitionPaused) {
      cameraStatus.classList.add("is-paused");
      cameraStatusText.textContent = "Camera live — recognition paused";
    } else {
      cameraStatus.classList.remove("is-paused");
      cameraStatusText.textContent = "Camera live — recognition active";
    }
  }

  function updateCameraUI(active) {
    cameraActive = active;
    startCameraBtn.classList.toggle("hidden", active);
    pauseCameraBtn.classList.toggle("hidden", !active);
    stopCameraBtn.classList.toggle("hidden", !active);
    cameraPlaceholder.classList.toggle("hidden", active);
    cameraVideo.classList.toggle("hidden", !active);
    cameraStage.classList.toggle("is-active", active);
    predictBtn.disabled = predicting || recognitionPaused || (!active && !selectedFile);
    predictHint.textContent = active
      ? (recognitionPaused
          ? "Recognition is paused. Resume to predict signs."
          : "Camera is live — show a sign, then capture and predict.")
      : "Start recognition to capture and predict a sign.";
    updateCameraStatus();
  }

  function updatePauseUI() {
    pauseCameraBtn.textContent = recognitionPaused ? "Resume Recognition" : "Pause Recognition";
    predictBtn.disabled = predicting || recognitionPaused || (!cameraActive && !selectedFile);
    predictHint.textContent = cameraActive
      ? (recognitionPaused
          ? "Recognition is paused. Resume to predict signs."
          : "Camera is live — show a sign, then capture and predict.")
      : "Start recognition to capture and predict a sign.";
    updateCameraStatus();
  }

  async function startCamera() {
    clearCameraMessage();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showCameraMessage(
        "Your browser does not support camera access. Use a modern browser such as Chrome, Edge, or Firefox.",
        true
      );
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
    } catch (err) {
      var msg = "Unable to access the camera.";
      if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
        msg = "Camera permission was denied. Allow camera access in your browser and try again.";
      } else if (err && err.name === "NotFoundError") {
        msg = "No camera was found on this device.";
      } else if (err && err.name === "NotReadableError") {
        msg = "The camera is already in use by another application.";
      }
      showCameraMessage(msg, true);
      return;
    }

    cameraVideo.srcObject = stream;
    recognitionPaused = false;
    resetResult();
    hideDetectionStatus();
    updateCameraUI(true);
    updatePauseUI();
  }

  function pauseRecognition() {
    if (!cameraActive) return;
    recognitionPaused = !recognitionPaused;
    if (recognitionPaused) {
      // Dropping history: next prediction starts fresh when resumed.
      resetResult();
      hideDetectionStatus();
    }
    updatePauseUI();
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(function (track) {
        track.stop();
      });
      stream = null;
    }
    cameraVideo.srcObject = null;
    recognitionPaused = false;
    resetResult();
    hideDetectionStatus();
    updateCameraUI(false);
    clearCameraMessage();
  }

  var WEBCAM_CAPTURE_COUNT = 5; // matches Flask temporal smoothing window
  var WEBCAM_CAPTURE_INTERVAL_MS = 120;

  function captureFrameToBlob() {
    return new Promise(function (resolve, reject) {
      if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
        reject(new Error("The camera has not produced a frame yet."));
        return;
      }
      captureCanvas.width = cameraVideo.videoWidth;
      captureCanvas.height = cameraVideo.videoHeight;
      var ctx = captureCanvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas is not supported in this browser."));
        return;
      }
      ctx.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCanvas.toBlob(function (blob) {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error("Could not capture the video frame."));
        }
      }, "image/jpeg", 0.92);
    });
  }

  // Capture a short burst of frames so the server can apply the same 5-frame
  // majority-vote smoothing as recognition/real_time_landmark.py.
  function captureFrameBurst() {
    var blobs = [];
    var timer = null;

    return new Promise(function (resolve, reject) {
      timer = window.setInterval(function () {
        captureFrameToBlob().then(function (blob) {
          blobs.push(blob);
          if (blobs.length === WEBCAM_CAPTURE_COUNT) {
            window.clearInterval(timer);
            resolve(blobs);
          }
        }).catch(function (err) {
          window.clearInterval(timer);
          reject(err);
        });
      }, WEBCAM_CAPTURE_INTERVAL_MS);
    });
  }

  /* --------------------------------------------------------------------------
   * Validation + preview (image upload secondary)
   * ------------------------------------------------------------------------ */
  function validateFile(file) {
    if (!file) return "Please choose an image file.";
    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
      return "Unsupported file type. Use PNG, JPG, BMP or WebP.";
    }
    if (file.size > MAX_SIZE) {
      return "File is too large. Maximum size is 8 MB.";
    }
    return null;
  }

  function updatePreview(file) {
    if (!file) {
      previewWrap.classList.add("hidden");
      preview.src = "";
      resetBtn.disabled = true;
      predictBtn.disabled = predicting || !stream;
      return;
    }
    var url = URL.createObjectURL(file);
    preview.onload = function () {
      URL.revokeObjectURL(url);
    };
    preview.src = url;
    previewWrap.classList.remove("hidden");
    resetBtn.disabled = false;
    predictBtn.disabled = predicting;
  }

  function selectFile(file) {
    var err = validateFile(file);
    if (err) {
      showError(err);
      clearSelection();
      return;
    }
    selectedFile = file;
    clearError();
    resetResult();
    updatePreview(file);
  }

  function clearSelection() {
    selectedFile = null;
    fileInput.value = "";
    updatePreview(null);
  }

  function resetResult() {
    resultPlaceholder.classList.remove("hidden");
    resultContent.classList.add("hidden");
    predictionLetter.textContent = "—";
    predictionConfidence.textContent = "Confidence 0%";
    confidenceFill.style.width = "0%";
    topK.innerHTML = "";
    clearError();
  }

  function setResult(data) {
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");
    predictionLanguage.textContent = "Mode: " + (data.language || currentLanguage);

    if (data.prediction) {
      predictionLetter.textContent = data.prediction;
      var confidence = data.confidence != null ? data.confidence : 0;
      predictionConfidence.textContent =
        "Confidence " + (confidence * 100).toFixed(1) + "%";
      confidenceFill.style.width = Math.max(0, Math.min(100, confidence * 100)) + "%";
      clearError();
      updateDetectionStatus("Hand detected", "detected");
    } else {
      predictionLetter.textContent = "?";
      predictionConfidence.textContent = "Confidence —";
      confidenceFill.style.width = "0%";
      if (data.error) showError(data.error);
      updateDetectionStatus(
        data.error || "No hand detected. Show your sign inside the camera view.",
        "missing"
      );
    }

    topK.innerHTML = "";
    if (Array.isArray(data.top_k)) {
      data.top_k.forEach(function (item) {
        var el = document.createElement("div");
        el.className = "top-k__item";

        var letter = document.createElement("span");
        letter.className = "top-k__letter";
        letter.textContent = item.letter;

        var pct = document.createElement("span");
        pct.className = "top-k__pct";
        pct.textContent = (item.probability * 100).toFixed(1) + "%";

        el.appendChild(letter);
        el.appendChild(pct);
        topK.appendChild(el);
      });
    }
  }

  function setLoading(loading) {
    predicting = loading;
    predictBtn.disabled = loading || (!stream && !selectedFile);
    predictSpinner.classList.toggle("hidden", !loading);
    predictLabel.textContent = loading ? "Predicting…" : "Predict Sign";
  }

  async function predict() {
    if (predicting) return;
    if (recognitionPaused) {
      showError("Recognition is paused. Resume recognition to predict.");
      return;
    }

    clearError();
    setLoading(true);

    var formData = new FormData();
    formData.append("language", currentLanguage);

    if (stream) {
      // Webcam: capture a burst so Flask can apply the working script's
      // 5-frame majority-vote smoothing.
      var blobs = null;
      try {
        blobs = await captureFrameBurst();
      } catch (err) {
        setLoading(false);
        showError(err.message);
        return;
      }

      blobs.forEach(function (blob, index) {
        formData.append("frames", blob, "frame_" + index + ".jpg");
      });
    } else if (selectedFile) {
      formData.append("image", selectedFile, "capture.jpg");
    } else {
      setLoading(false);
      showError("Enable the camera or choose an image first.");
      return;
    }

    fetch("/api/predict", {
      method: "POST",
      body: formData,
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        setLoading(false);

        if (!result.ok) {
          showError(result.data.error || "Prediction failed.");
          predictionLanguage.textContent = "Mode: " + currentLanguage;
          return;
        }

        setResult(result.data);
      })
      .catch(function (err) {
        setLoading(false);
        showError("Network error: " + err.message);
      });
  }

  /* --------------------------------------------------------------------------
   * Event wiring
   * ------------------------------------------------------------------------ */
  startCameraBtn.addEventListener("click", startCamera);
  if (pauseCameraBtn) pauseCameraBtn.addEventListener("click", pauseRecognition);
  stopCameraBtn.addEventListener("click", stopCamera);
  predictBtn.addEventListener("click", predict);

  chooseBtn.addEventListener("click", function () {
    fileInput.click();
  });

  dropZone.addEventListener("click", function (e) {
    if (e.target === clearBtn) return;
    fileInput.click();
  });

  dropZone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) selectFile(fileInput.files[0]);
  });

  clearBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    clearSelection();
  });

  resetBtn.addEventListener("click", function () {
    clearSelection();
    resetResult();
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    dropZone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropZone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (evt) {
    dropZone.addEventListener(evt, function (e) {
      e.preventDefault();
      dropZone.classList.remove("is-dragover");
    });
  });

  dropZone.addEventListener("drop", function (e) {
    if (e.dataTransfer.files.length) {
      selectFile(e.dataTransfer.files[0]);
    }
  });

  // Stop the camera when the user navigates away.
  window.addEventListener("beforeunload", stopCamera);
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) stopCamera();
  });

  /* ==========================================================================
   * 3. Mobile navigation toggle
   * ========================================================================== */
  var navToggle = document.getElementById("navToggle");
  var navMenu = document.getElementById("navMenu");

  if (navToggle && navMenu) {
    navToggle.addEventListener("click", function () {
      var open = navMenu.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });

    navMenu.addEventListener("click", function (e) {
      if (e.target.classList.contains("nav__link")) {
        navMenu.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* ==========================================================================
   * 4. Active navigation highlighting on scroll
   * ========================================================================== */
  var navLinks = Array.prototype.slice.call(
    document.querySelectorAll(".nav__link")
  );

  function sectionIdFor(href) {
    var hash = href.split("#")[1];
    return hash ? document.getElementById(hash) : null;
  }

  function updateActiveNav() {
    var current = null;
    navLinks.forEach(function (link) {
      var section = sectionIdFor(link.getAttribute("href"));
      if (!section) return;
      var rect = section.getBoundingClientRect();
      if (rect.top <= 120) {
        current = link;
      }
    });

    navLinks.forEach(function (link) {
      link.classList.toggle("is-active", link === current);
    });
  }

  var scrollTicking = false;
  window.addEventListener("scroll", function () {
    if (!scrollTicking) {
      window.requestAnimationFrame(function () {
        updateActiveNav();
        scrollTicking = false;
      });
      scrollTicking = true;
    }
  });
  updateActiveNav();
})();
