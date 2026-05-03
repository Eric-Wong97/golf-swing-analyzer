/**
 * main.js — App entry point and render loop orchestrator.
 */

console.log('main.js: Script loading started');

import { initPoseLandmarker, detectPose, initRefPoseLandmarker, detectRefPose, isReady, getConstants } from './mediapipe.js';
import { initDrawing, resetDrawing } from './drawing.js';

// Lazy load puppet.js to keep initial load light
let updatePuppet = null;
let updateReferencePuppet = null;
let initPuppet = null;

console.log('main.js: Essential modules imported');

// ─── DOM References ───────────────────────────────────────────────────────────
const uploadZone      = document.getElementById('upload-zone');
const videoUploadEl   = document.getElementById('video-upload');
const fileBadge       = document.getElementById('file-badge');
const fileNameText    = document.getElementById('file-name-text');

const refVideoUploadEl = document.getElementById('ref-video-upload');
const refFileBadge     = document.getElementById('ref-file-badge');
const refFileNameText  = document.getElementById('ref-file-name-text');

const loadingEl       = document.getElementById('loading-indicator');
const workspaceEl     = document.getElementById('workspace');

const videoEl         = document.getElementById('video-element');
const refVideoEl      = document.getElementById('ref-video-element');
const outputCanvas    = document.getElementById('output-canvas');
const lineCanvas      = document.getElementById('line-canvas');
const outputCtx       = outputCanvas.getContext('2d');

const puppetContainer = document.getElementById('puppet-container');

// ─── State ────────────────────────────────────────────────────────────────────
let lastVideoTime = -1;
let puppetInitialized = false;

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
uploadZone.addEventListener('click', () => {
  console.log('main.js: Upload zone clicked');
  videoUploadEl.click();
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) {
    loadVideo(file);
  }
});

videoUploadEl.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) loadVideo(file);
});

refVideoUploadEl.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) {
    refFileNameText.textContent = file.name;
    refFileBadge.classList.remove('hidden');
    
    const url = URL.createObjectURL(file);
    refVideoEl.src = url;
    refVideoEl.classList.remove('hidden');
    
    // Initialize secondary AI model
    await initRefPoseLandmarker();
    refVideoEl.play();
  }
});

async function loadVideo(file) {
  fileNameText.textContent = file.name;
  fileBadge.classList.remove('hidden');

  const url = URL.createObjectURL(file);
  videoEl.src = url;

  resetDrawing();
  lastVideoTime = -1;

  workspaceEl.classList.remove('hidden');
  workspaceEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Start heavy initialization only after upload
  if (!puppetInitialized) {
    console.log('main.js: Lazy loading puppet.js');
    try {
      const puppetModule = await import('./puppet.js');
      initPuppet = puppetModule.initPuppet;
      updatePuppet = puppetModule.updatePuppet;
      updateReferencePuppet = puppetModule.updateReferencePuppet;
      
      initPuppet(puppetContainer);
      puppetInitialized = true;
      console.log('main.js: Puppet initialized');
    } catch (err) {
      console.error('main.js: Failed to lazy load puppet.js', err);
    }
  }

  if (!isReady()) {
    console.log('main.js: Starting lazy pose landmarker init');
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

// ─── Render Loop ──────────────────────────────────────────────────────────────
function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (!isReady() || videoEl.readyState < 2 || videoEl.videoWidth === 0) return;
  if (lastVideoTime === videoEl.currentTime) return;

  lastVideoTime = videoEl.currentTime;
  const timestampMs = performance.now();

  const { PoseLandmarker, DrawingUtils } = getConstants();

  detectPose(videoEl, timestampMs, (result) => {
    outputCtx.save();
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    if (result.landmarks?.length > 0 && DrawingUtils && PoseLandmarker) {
      const landmarks = result.landmarks[0];
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
    }

    if (result.worldLandmarks?.length > 0 && updatePuppet) {
      updatePuppet(result.worldLandmarks[0]);
    }

    outputCtx.restore();
  });

  if (refVideoEl && refVideoEl.readyState >= 2 && refVideoEl.videoWidth > 0 && !refVideoEl.paused) {
    detectRefPose(refVideoEl, performance.now(), (result) => {
      if (result.worldLandmarks?.length > 0 && updateReferencePuppet) {
        updateReferencePuppet(result.worldLandmarks[0]);
      }
    });
  }
}

console.log('main.js: Initial execution completed');
requestAnimationFrame(renderLoop);
