/**
 * puppet.js — Handles the 3D puppet visualization using Three.js.
 */

import * as THREE from 'https://esm.sh/three@0.160.0';
import { OrbitControls } from 'https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let joints = [];
let bones = [];
let targetPositions = []; // Latest data from AI

let refJoints = [];
let refBones = [];
let refTargetPositions = []; // Latest data from AI for reference

let torso, head;
let refTorso, refHead;

let lerpFactor = 0.2;     // Smoothing factor (0 to 1, lower = smoother but more lag)

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper body
  [11, 23], [12, 24], [23, 24],                   // Torso
  [23, 25], [25, 27], [24, 26], [26, 28],         // Legs
  [27, 31], [28, 32], [27, 29], [28, 30]          // Feet
];

// Define fixed bone lengths for realism (in meters)
const BONE_DEFS = [
  { parent: 11, child: 13, len: 0.28 }, // L Upper Arm
  { parent: 13, child: 15, len: 0.25 }, // L Forearm
  { parent: 12, child: 14, len: 0.28 }, // R Upper Arm
  { parent: 14, child: 16, len: 0.25 }, // R Forearm
  { parent: 23, child: 25, len: 0.42 }, // L Thigh
  { parent: 25, child: 27, len: 0.40 }, // L Shin
  { parent: 24, child: 26, len: 0.42 }, // R Thigh
  { parent: 26, child: 28, len: 0.40 }, // R Shin
  { parent: 27, child: 31, len: 0.15 }, // L Foot
  { parent: 28, child: 32, len: 0.15 }, // R Foot
];

// Hierarchy for FK (Root is Hips)
const HIERARCHY = {
  11: 23, 12: 24, // Shoulders from Hips
  13: 11, 14: 12, // Elbows from Shoulders
  15: 13, 16: 14, // Wrists from Elbows
  25: 23, 26: 24, // Knees from Hips
  27: 25, 28: 26, // Ankles from Knees
  31: 27, 32: 28, // Toes from Ankles
  0: 11, // Head from L Shoulder (anchor point)
};

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
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  const axesHelper = new THREE.AxesHelper(1);
  scene.add(axesHelper);

  const jointGeo = new THREE.SphereGeometry(0.04, 16, 16);
  const jointMat = new THREE.MeshPhongMaterial({ color: 0x22d36e });
  const refJointMat = new THREE.MeshPhongMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.4 });
  
  joints = [];
  targetPositions = [];
  refJoints = [];
  refTargetPositions = [];

  for (let i = 0; i < 33; i++) {
    const joint = new THREE.Mesh(jointGeo, jointMat);
    joint.visible = false;
    scene.add(joint);
    joints.push(joint);
    targetPositions.push(new THREE.Vector3());

    const refJoint = new THREE.Mesh(jointGeo, refJointMat);
    refJoint.visible = false;
    scene.add(refJoint);
    refJoints.push(refJoint);
    refTargetPositions.push(new THREE.Vector3());
  }

  // Create "Bones" as Capsules for better visuals
  bones = [];
  refBones = [];
  const boneMat = new THREE.MeshPhongMaterial({ color: 0x2d3436 });
  const refBoneMat = new THREE.MeshPhongMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.2 });

  POSE_CONNECTIONS.forEach(() => {
    // We use simple cylinders scaled/rotated in animate
    const boneGeo = new THREE.CylinderGeometry(0.025, 0.025, 1, 8);
    const bone = new THREE.Mesh(boneGeo, boneMat);
    bone.visible = false;
    scene.add(bone);
    bones.push(bone);

    const refBone = new THREE.Mesh(boneGeo, refBoneMat);
    refBone.visible = false;
    scene.add(refBone);
    refBones.push(refBone);
  });

  // Torso and Head
  const torsoGeo = new THREE.BoxGeometry(0.3, 0.45, 0.15);
  const torsoMat = new THREE.MeshPhongMaterial({ color: 0x2d3436 });
  torso = new THREE.Mesh(torsoGeo, torsoMat);
  scene.add(torso);

  const headGeo = new THREE.SphereGeometry(0.12, 16, 16);
  head = new THREE.Mesh(headGeo, torsoMat);
  scene.add(head);

  const refTorsoMat = new THREE.MeshPhongMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.2 });
  refTorso = new THREE.Mesh(torsoGeo, refTorsoMat);
  scene.add(refTorso);

  refHead = new THREE.Mesh(headGeo, refTorsoMat);
  scene.add(refHead);

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
    if (refJoints[i].visible) {
      refJoints[i].position.lerp(refTargetPositions[i], lerpFactor);
    }
  }

  // Update Body Parts
  updateBodyParts(torso, head, joints);
  updateBodyParts(refTorso, refHead, refJoints);

  // Update Bone meshes to connect smoothed joints
  POSE_CONNECTIONS.forEach((conn, i) => {
    updateBoneMesh(bones[i], joints[conn[0]], joints[conn[1]]);
    updateBoneMesh(refBones[i], refJoints[conn[0]], refJoints[conn[1]]);
  });

  if (renderer && scene && camera) renderer.render(scene, camera);
}

function updateBodyParts(torsoMesh, headMesh, jointList) {
  const sL = jointList[11];
  const sR = jointList[12];
  const hL = jointList[23];
  const hR = jointList[24];
  const nose = jointList[0];

  if (sL?.visible && sR?.visible && hL?.visible && hR?.visible) {
    const shoulderMid = sL.position.clone().add(sR.position).multiplyScalar(0.5);
    const hipMid = hL.position.clone().add(hR.position).multiplyScalar(0.5);
    
    torsoMesh.position.copy(shoulderMid).add(hipMid).multiplyScalar(0.5);
    torsoMesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      shoulderMid.clone().sub(hipMid).normalize()
    );
    torsoMesh.visible = true;
  } else {
    torsoMesh.visible = false;
  }

  if (nose?.visible && sL?.visible && sR?.visible) {
    const shoulderMid = sL.position.clone().add(sR.position).multiplyScalar(0.5);
    headMesh.position.copy(nose.position);
    headMesh.visible = true;
  } else {
    headMesh.visible = false;
  }
}

/**
 * Positions and scales a cylinder between two points.
 */
function updateBoneMesh(bone, start, end) {
  if (start && end && start.visible && end.visible) {
    const startPos = start.position;
    const endPos = end.position;
    const distance = startPos.distanceTo(endPos);
    
    bone.scale.set(1, distance, 1);
    bone.position.copy(startPos).add(endPos).multiplyScalar(0.5);
    bone.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      endPos.clone().sub(startPos).normalize()
    );
    bone.visible = true;
  } else {
    bone.visible = false;
  }
}

/**
 * Applies anatomical constraints to the raw landmarks.
 * Root is the midpoint of the hips (23, 24).
 */
function applyAnatomicalConstraints(landmarks, targets) {
  const root = new THREE.Vector3(
    (landmarks[23].x + landmarks[24].x) * 0.5,
    (landmarks[23].y + landmarks[24].y) * 0.5,
    (landmarks[23].z + landmarks[24].z) * 0.5
  );

  // We'll calculate the lowest Y to ground it later, 
  // but first we need to build the skeleton relative to root.
  const constrained = new Array(33).fill(null);
  
  // Initialize hips
  constrained[23] = new THREE.Vector3(landmarks[23].x, landmarks[23].y, landmarks[23].z);
  constrained[24] = new THREE.Vector3(landmarks[24].x, landmarks[24].y, landmarks[24].z);

  // Traverse hierarchy to fix bone lengths
  const queue = [23, 24];
  const visited = new Set([23, 24]);

  // Find connections for the queue
  const connections = {};
  POSE_CONNECTIONS.forEach(([p, c]) => {
    if (!connections[p]) connections[p] = [];
    if (!connections[c]) connections[c] = [];
    connections[p].push(c);
    connections[c].push(p);
  });

  while (queue.length > 0) {
    const pIdx = queue.shift();
    const children = connections[pIdx] || [];
    
    children.forEach(cIdx => {
      if (visited.has(cIdx)) return;
      
      const parentPos = constrained[pIdx];
      const rawDir = new THREE.Vector3(
        landmarks[cIdx].x - landmarks[pIdx].x,
        landmarks[cIdx].y - landmarks[pIdx].y,
        landmarks[cIdx].z - landmarks[pIdx].z
      ).normalize();

      // Find bone length
      const def = BONE_DEFS.find(d => (d.parent === pIdx && d.child === cIdx) || (d.parent === cIdx && d.child === pIdx));
      let length = def ? def.len : parentPos.distanceTo(new THREE.Vector3(landmarks[cIdx].x, landmarks[cIdx].y, landmarks[cIdx].z));

      // Extra Z-stabilization: dampening deep Z movements
      const zDiff = landmarks[cIdx].z - landmarks[pIdx].z;
      const dampedZ = zDiff * 0.7; // Dampen Z depth by 30% to reduce flickering contortions
      
      const stabilizedDir = new THREE.Vector3(
        landmarks[cIdx].x - landmarks[pIdx].x,
        landmarks[cIdx].y - landmarks[pIdx].y,
        dampedZ
      ).normalize();

      constrained[cIdx] = parentPos.clone().add(stabilizedDir.multiplyScalar(length));
      visited.add(cIdx);
      queue.push(cIdx);
    });
  }

  // Calculate lowest Y for grounding
  const feetIndices = [27, 28, 29, 30, 31, 32];
  let lowestY = -Infinity;
  constrained.forEach((pos, idx) => {
    if (pos && feetIndices.includes(idx)) {
      if (pos.y > lowestY) lowestY = pos.y;
    }
  });

  // Apply to targets with grounding and scale
  constrained.forEach((pos, i) => {
    if (pos && targets[i]) {
      targets[i].set(pos.x, (lowestY - pos.y), -pos.z);
    }
  });
}

/**
 * Update the target positions from MediaPipe landmarks.
 */
export function updatePuppet(worldLandmarks) {
  if (!worldLandmarks || worldLandmarks.length === 0) {
    joints.forEach(j => j.visible = false);
    return;
  }

  applyAnatomicalConstraints(worldLandmarks, targetPositions);
  
  joints.forEach((j, i) => {
    if (!j.visible) {
      j.position.copy(targetPositions[i]);
      j.visible = true;
    }
  });
}

/**
 * Update the target positions for the reference skeleton from MediaPipe landmarks.
 */
export function updateReferencePuppet(worldLandmarks) {
  if (!worldLandmarks || worldLandmarks.length === 0) {
    refJoints.forEach(j => j.visible = false);
    return;
  }

  applyAnatomicalConstraints(worldLandmarks, refTargetPositions);
  
  refJoints.forEach((j, i) => {
    if (!j.visible) {
      j.position.copy(refTargetPositions[i]);
      j.visible = true;
    }
  });
}
