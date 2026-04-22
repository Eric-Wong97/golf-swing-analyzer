/**
 * main.js — App entry point and render loop orchestrator.
 */

import { calculateAngle } from './geometry.js';
import { initPoseLandmarker, detectPose, isReady, PoseLandmarker, DrawingUtils } from './mediapipe.js';
import { initDrawing, resetDrawing } from './drawing.js';
import { resetPerformanceData, recordFrame, updateReport, exportReport } from './performance.js';

// ─── DOM References ───────────────────────────────────────────────────────────
const uploadZone      = document.getElementById('upload-zone');
const videoUploadEl   = document.getElementById('video-upload');
const fileBadge       = document.getElementById('file-badge');
const fileNameText    = document.getElementById('file-name-text');
const loadingEl       = document.getElementById('loading-indicator');
const workspaceEl     = document.getElementById('workspace');

const videoEl         = document.getElementById('video-element');
const outputCanvas    = document.getElementById('output-canvas');
const lineCanvas      = document.getElementById('line-canvas');
const outputCtx       = outputCanvas.getContext('2d');

const armAngleEl      = document.getElementById('arm-angle');
const armRecEl        = document.getElementById('arm-rec');
const armBarEl        = document.getElementById('arm-bar');
const kneeAngleEl     = document.getElementById('knee-angle');
const kneeRecEl       = document.getElementById('knee-rec');
const kneeBarEl       = document.getElementById('knee-bar');

const exportBtn       = document.getElementById('export-report');

// ─── State ────────────────────────────────────────────────────────────────────
let lastVideoTime = -1;

// ─── Drawing init ─────────────────────────────────────────────────────────────
initDrawing({
  lineCanvas,
  colorPicker: document.getElementById('line-color'),
  toggleBtn:   document.getElementById('toggle-drawing'),
  undoBtn:     document.getElementById('undo-drawing'),
  redoBtn:     document.getElementById('redo-drawing'),
  clearBtn:    document.getElementById('clear-drawing'),
  modeBadge:   document.getElementById('mode-badge'),
});

// ─── Upload zone ──────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => videoUploadEl.click());
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) loadVideo(file);
});

videoUploadEl.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) loadVideo(file);
});

async function loadVideo(file) {
  fileNameText.textContent = file.name;
  fileBadge.classList.remove('hidden');

  const url = URL.createObjectURL(file);
  videoEl.src = url;

  resetPerformanceData();
  resetDrawing();
  document.getElementById('performance-report').classList.add('hidden');
  lastVideoTime = -1;

  workspaceEl.classList.remove('hidden');
  workspaceEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

  if (!isReady()) {
    await initPoseLandmarker((loading) => {
      loadingEl.classList.toggle('hidden', !loading);
    });
  }
}

// ─── Canvas resize on video metadata ─────────────────────────────────────────
videoEl.addEventListener('loadedmetadata', () => {
  outputCanvas.width  = videoEl.videoWidth;
  outputCanvas.height = videoEl.videoHeight;
  lineCanvas.width    = videoEl.videoWidth;
  lineCanvas.height   = videoEl.videoHeight;
});

// ─── Export ───────────────────────────────────────────────────────────────────
exportBtn.addEventListener('click', () => exportReport(videoEl));

// ─── Dashboard helpers ────────────────────────────────────────────────────────
function setAngleDisplay(valueEl, recEl, barEl, angle, isGood, goodText, badText) {
  valueEl.textContent = `${Math.round(angle)}°`;
  recEl.textContent   = isGood ? goodText : badText;
  recEl.className     = `metric-rec ${isGood ? 'rec-good' : 'rec-bad'}`;
  const pct = Math.min(Math.max((angle / 180) * 100, 0), 100);
  barEl.style.width = `${pct}%`;
  barEl.className   = `metric-bar ${isGood ? 'bar-good' : 'bar-bad'}`;
}

function setNoDetection() {
  [armAngleEl, kneeAngleEl].forEach((el) => (el.textContent = '--°'));
  [armRecEl, kneeRecEl].forEach((el) => {
    el.textContent = 'No person detected';
    el.className = 'metric-rec';
  });
  [armBarEl, kneeBarEl].forEach((el) => (el.style.width = '0%'));
}

// ─── Render Loop ──────────────────────────────────────────────────────────────
function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (!isReady() || videoEl.readyState < 2 || videoEl.videoWidth === 0) return;
  if (lastVideoTime === videoEl.currentTime) return;

  lastVideoTime = videoEl.currentTime;
  const timestampMs = performance.now();

  detectPose(videoEl, timestampMs, (result) => {
    outputCtx.save();
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    if (result.landmarks?.length > 0) {
      const landmarks = result.landmarks[0];

      // Draw skeleton
      const du = new DrawingUtils(outputCtx);
      du.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: 'rgba(0,255,120,0.85)',
        lineWidth: 2,
      });
      du.drawLandmarks(landmarks, {
        color: '#ff3b6b',
        lineWidth: 1,
        radius: 3,
      });

      // Extract key landmarks
      const lShoulder = landmarks[11];
      const lElbow    = landmarks[13];
      const lWrist    = landmarks[15];
      const lHip      = landmarks[23];
      const lKnee     = landmarks[25];
      const lAnkle    = landmarks[27];

      if (lShoulder && lElbow && lWrist && lHip && lKnee && lAnkle) {
        const armAngle  = calculateAngle(lShoulder, lElbow, lWrist);
        const kneeAngle = calculateAngle(lHip, lKnee, lAnkle);

        const isArmGood  = armAngle >= 160;
        const isKneeGood = kneeAngle >= 130 && kneeAngle <= 170;
        let kneeText = 'Knee posture: GOOD';
        if (kneeAngle > 170) kneeText = 'Add knee bend!';
        else if (kneeAngle < 130) kneeText = 'Less knee bend!';

        setAngleDisplay(armAngleEl, armRecEl, armBarEl, armAngle, isArmGood, 'Lead arm straight ✓', 'Keep lead arm straight!');
        setAngleDisplay(kneeAngleEl, kneeRecEl, kneeBarEl, kneeAngle, isKneeGood, 'Knee posture: GOOD ✓', kneeText);

        // Angle overlay on canvas
        outputCtx.font         = `bold 22px 'JetBrains Mono', monospace`;
        outputCtx.lineWidth    = 3;
        outputCtx.strokeStyle  = 'rgba(0,0,0,0.7)';
        outputCtx.fillStyle    = isArmGood ? '#00ff78' : '#ff3b6b';

        const ex = lElbow.x * outputCanvas.width;
        const ey = lElbow.y * outputCanvas.height;
        outputCtx.strokeText(`${Math.round(armAngle)}°`, ex + 14, ey - 4);
        outputCtx.fillText(`${Math.round(armAngle)}°`, ex + 14, ey - 4);

        outputCtx.fillStyle = isKneeGood ? '#00ff78' : '#ff3b6b';
        const kx = lKnee.x * outputCanvas.width;
        const ky = lKnee.y * outputCanvas.height;
        outputCtx.strokeText(`${Math.round(kneeAngle)}°`, kx + 14, ky - 4);
        outputCtx.fillText(`${Math.round(kneeAngle)}°`, kx + 14, ky - 4);

        recordFrame({ armAngle, kneeAngle, detected: true });
        updateReport(videoEl);
      }
    } else {
      setNoDetection();
      recordFrame({ armAngle: 0, kneeAngle: 0, detected: false });
    }

    outputCtx.restore();
  });
}

requestAnimationFrame(renderLoop);
