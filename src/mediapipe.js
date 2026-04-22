/**
 * mediapipe.js
 * Handles loading and running the MediaPipe PoseLandmarker.
 */

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0';

let poseLandmarker = null;
let isLoading = false;

const WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export { PoseLandmarker, DrawingUtils };

/**
 * Load the pose landmarker model (idempotent — safe to call multiple times).
 * @param {(loading: boolean) => void} onLoadingChange
 */
export async function initPoseLandmarker(onLoadingChange) {
  if (poseLandmarker || isLoading) return;
  isLoading = true;
  onLoadingChange(true);

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });

  isLoading = false;
  onLoadingChange(false);
}

/**
 * Run pose detection on the current video frame.
 * @param {HTMLVideoElement} video
 * @param {number} timestampMs
 * @param {(result: import('@mediapipe/tasks-vision').PoseLandmarkerResult) => void} callback
 */
export function detectPose(video, timestampMs, callback) {
  if (!poseLandmarker) return;
  poseLandmarker.detectForVideo(video, timestampMs, callback);
}

export function isReady() {
  return poseLandmarker !== null;
}
