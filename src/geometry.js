/**
 * geometry.js
 * Pure math utilities for angle calculations.
 */

/**
 * Calculate the angle at vertex B formed by points A-B-C.
 * @param {{x:number, y:number}} a
 * @param {{x:number, y:number}} b  – the vertex
 * @param {{x:number, y:number}} c
 * @returns {number} angle in degrees [0, 180]
 */
export function calculateAngle(a, b, c) {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

/**
 * Clamp a value between min and max.
 */
export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
