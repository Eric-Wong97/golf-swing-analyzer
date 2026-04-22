/**
 * mediapipe.js
 * Handles loading and running the MediaPipe PoseLandmarker.
 */

let poseLandmarker = null;
let isLoading = false;
let DrawingUtils = null;
let PoseLandmarker = null;

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm';
const MODEL_URL = '/pose_landmarker_lite.task';

export { PoseLandmarker, DrawingUtils };

/**
 * Load the pose landmarker model (idempotent — safe to call multiple times).
 */
export async function initPoseLandmarker(onLoadingChange) {
  if (poseLandmarker || isLoading) return;
  isLoading = true;
  onLoadingChange(true);

  try {
    console.log('mediapipe.js: Loading MediaPipe tasks-vision from CDN...');
    const visionModule = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0');
    PoseLandmarker = visionModule.PoseLandmarker;
    DrawingUtils = visionModule.DrawingUtils;
    const FilesetResolver = visionModule.FilesetResolver;

    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    console.log('mediapipe.js: MediaPipe initialized successfully');
  } catch (err) {
    console.error('mediapipe.js: Failed to initialize MediaPipe', err);
  } finally {
    isLoading = false;
    onLoadingChange(false);
  }
}

export function detectPose(video, timestampMs, callback) {
  if (!poseLandmarker) return;
  poseLandmarker.detectForVideo(video, timestampMs, callback);
}

export function isReady() {
  return poseLandmarker !== null;
}

export function getConstants() {
  return { PoseLandmarker, DrawingUtils };
}
