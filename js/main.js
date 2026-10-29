document.addEventListener("DOMContentLoaded", () => {
  const angleButtons = document.querySelectorAll("[data-angle]");

  const angleVideo = document.getElementById("angle-video");
  const angleVideoSource = document.getElementById("angle-video-source");

  const selectedAngle = document.getElementById("selected-angle");
  const angleVideoTitle = document.getElementById("angle-video-title");
  const angleVideoTag = document.getElementById("angle-video-tag");
  const angleVideoCaption = document.getElementById("angle-video-caption");
  const angleStatus = document.getElementById("angle-status");

  const angleVideos = {
    0: "BasketballRebounds0Deg.mp4",
    10: "BasketballRebounds10Deg.mp4",
    20: "BasketballRebounds20Deg.mp4",
    30: "BasketballRebounds30Deg.mp4",
    45: "BasketballRebounds45Deg.mp4",
    60: "BasketballRebounds60Deg.mp4",
    70: "BasketballRebounds70Deg.mp4",
    80: "BasketballRebounds80Deg.mp4"
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
});
