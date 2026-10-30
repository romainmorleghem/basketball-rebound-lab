
document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // ANGLE EXPLORER
  // ============================================================

  const angleButtons = document.querySelectorAll("[data-angle]");

  const angleVideo = document.getElementById("angle-video");
  const angleVideoSource = document.getElementById("angle-video-source");
  const selectedAngle = document.getElementById("selected-angle");
  const angleVideoTitle = document.getElementById("angle-video-title");
  const angleVideoTag = document.getElementById("angle-video-tag");
  const angleVideoCaption = document.getElementById("angle-video-caption");
  const angleStatus = document.getElementById("angle-status");

  const angleVideos = {
    0: "0deg.mp4",
    15: "15deg.mp4",
    30: "30deg.mp4",
    45: "45deg.mp4",
    60: "60deg.mp4",
    75: "75deg.mp4"
  };

  function selectAngle(angle) {
    const videoFilename = angleVideos[angle];

    if (!videoFilename || !angleVideo || !angleVideoSource) {
      return;
    }

    const angleLabel = `${angle}°`;
    const videoPath = `assets/videos/${videoFilename}`;

    // Update the selected angle button.
    angleButtons.forEach((button) => {
      const isSelected = button.dataset.angle === String(angle);

      button.classList.toggle("active", isSelected);
      button.setAttribute("aria-pressed", String(isSelected));
    });

    // Update angle labels.
    if (selectedAngle) {
      selectedAngle.textContent = angleLabel;
    }
    if (angleVideoTitle) {
      angleVideoTitle.textContent = `Trajectories at ${angleLabel}`;
    }

    if (angleVideoTag) {
      angleVideoTag.textContent = angleLabel;
    }

    if (angleVideoCaption) {
      angleVideoCaption.textContent =
        `Manim simulation for a ${angleLabel} shot angle.`;
    }

    // Reset status and remove any previous error state.
    if (angleStatus) {
      angleStatus.textContent = `Selected angle: ${angleLabel}`;
      angleStatus.classList.remove("error");
    }

    // Avoid restarting the video if the selected source is already loaded.
    if (angleVideoSource.getAttribute("src") === videoPath) {
      return;
    }

    // Change the video source.
    angleVideo.pause();
    angleVideoSource.setAttribute("src", videoPath);
    angleVideo.load();

    // Report a missing video if the browser cannot load it.
    angleVideo.onerror = () => {
      if (angleStatus) {
        angleStatus.textContent =
          `Could not load ${videoFilename}. Check that the file exists in assets/videos/.`;
        angleStatus.classList.add("error");
      }
    };
  }

  // Attach the angle selection behaviour.
  angleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      selectAngle(button.dataset.angle);
    });
  });

  // Initialise the page with the 0-degree video.
  selectAngle("0");


  // ============================================================
  // PYTHON SOURCE CODE VIEWER
  // ============================================================

  const pythonSource = document.getElementById("python-source");
  const copyCodeButton = document.getElementById("copy-code");
  const codeStatus = document.getElementById("code-status");
  const pythonFile = "assets/BasketBall_Rebounds_MonteCarlo.py";
  let loadedPythonCode = "";

  async function loadPythonSource() {
    if (!pythonSource) return;

    try {
      const response = await fetch(pythonFile);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      loadedPythonCode = await response.text();
      pythonSource.textContent = loadedPythonCode;

      if (window.Prism) {
        window.Prism.highlightElement(pythonSource);
      }

      if (codeStatus) {
        codeStatus.textContent = "Source code loaded successfully.";
        codeStatus.classList.remove("error");
      }
    } catch (error) {
      console.error("Could not load Python source:", error);

      pythonSource.textContent =
        "Unable to load the Python source code. Check that the .py file is in the assets folder.";

      if (codeStatus) {
        codeStatus.textContent =
          "Could not load the source file. Check the file path and run the site through a local HTTP server.";
        codeStatus.classList.add("error");
      }

      if (copyCodeButton) {
        copyCodeButton.disabled = true;
      }
    }
  }

  if (copyCodeButton) {
    copyCodeButton.addEventListener("click", async () => {
      if (!loadedPythonCode) return;

      try {
        await navigator.clipboard.writeText(loadedPythonCode);

        copyCodeButton.textContent = "Copied!";

        if (codeStatus) {
          codeStatus.textContent = "Python source copied to clipboard.";
          codeStatus.classList.remove("error");
        }

        window.setTimeout(() => {
          copyCodeButton.textContent = "Copy code";
        }, 1800);
      } catch (error) {
        console.error("Could not copy Python source:", error);

        if (codeStatus) {
          codeStatus.textContent =
            "Clipboard access was blocked. You can still select and copy the code manually.";
          codeStatus.classList.add("error");
        }
      }
    });
  }

  loadPythonSource();
});
