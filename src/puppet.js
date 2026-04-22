/**
 * puppet.js — Handles the 3D puppet visualization using Three.js.
 */

import * as THREE from 'https://esm.sh/three@0.160.0';
import { OrbitControls } from 'https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let joints = [];
let bones = [];
let targetPositions = []; // Latest data from AI
let lerpFactor = 0.2;     // Smoothing factor (0 to 1, lower = smoother but more lag)

const POSE_CONNECTIONS = [
  [0, 11], [0, 12],                               // Neck/Head
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper body
  [11, 23], [12, 24], [23, 24],                   // Torso
  [23, 25], [25, 27], [24, 26], [26, 28],         // Legs
  [27, 31], [28, 32], [27, 29], [28, 30]          // Feet
];

/**
 * Initialize the 3D scene in the provided container.
 */
export function initPuppet(container) {
  console.log('puppet.js: Initializing 3D scene...');
  
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c10);

  const aspect = container.clientWidth / container.clientHeight;
  camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
  camera.position.set(0, 0, 3);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.innerHTML = ''; 
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 5, 5);
  scene.add(directionalLight);

  const gridHelper = new THREE.GridHelper(10, 10, 0x333333, 0x222222);
  gridHelper.position.y = -1;
  scene.add(gridHelper);

  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  const jointGeo = new THREE.SphereGeometry(0.04, 16, 16);
  const jointMat = new THREE.MeshPhongMaterial({ color: 0x22d36e });
  
  joints = [];
  targetPositions = [];
  for (let i = 0; i < 33; i++) {
    const joint = new THREE.Mesh(jointGeo, jointMat);
    joint.visible = false;
    scene.add(joint);
    joints.push(joint);
    targetPositions.push(new THREE.Vector3());
  }

  bones = [];
  const boneMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
  for (let i = 0; i < POSE_CONNECTIONS.length; i++) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3()
    ]);
    const line = new THREE.Line(geometry, boneMat);
    scene.add(line);
    bones.push(line);
  }

  window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();

  // ─── Smoothing Logic ───────────────────────────────────────────────────────
  // Lerp actual joint positions towards the targets from MediaPipe
  for (let i = 0; i < joints.length; i++) {
    if (joints[i].visible) {
      joints[i].position.lerp(targetPositions[i], lerpFactor);
    }
  }

  // Update Bone lines to follow the smoothed joints
  POSE_CONNECTIONS.forEach((conn, i) => {
    const start = joints[conn[0]];
    const end = joints[conn[1]];
    if (start && end && start.visible && end.visible) {
      const posAttr = bones[i].geometry.attributes.position;
      const arr = posAttr.array;
      arr[0] = start.position.x;
      arr[1] = start.position.y;
      arr[2] = start.position.z;
      arr[3] = end.position.x;
      arr[4] = end.position.y;
      arr[5] = end.position.z;
      posAttr.needsUpdate = true;
      bones[i].visible = true;
    } else {
      bones[i].visible = false;
    }
  });

  if (renderer && scene && camera) renderer.render(scene, camera);
}

/**
 * Update the target positions from MediaPipe landmarks.
 */
export function updatePuppet(worldLandmarks) {
  if (!worldLandmarks || worldLandmarks.length === 0) {
    joints.forEach(j => j.visible = false);
    return;
  }

  const scale = 1.0;
  worldLandmarks.forEach((lm, i) => {
    if (joints[i]) {
      // Set the TARGET position (will be lerped to in animate loop)
      // Unmirrored: Standard world coordinates
      targetPositions[i].set(lm.x * scale, -lm.y * scale, -lm.z * scale);
      
      // If it was just hidden, snap to position instead of lerping from (0,0,0)
      if (!joints[i].visible) {
        joints[i].position.copy(targetPositions[i]);
        joints[i].visible = true;
      }
    }
  });
}
