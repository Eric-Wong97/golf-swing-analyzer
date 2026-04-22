/**
 * drawing.js
 * Manages the two-point line drawing tool, undo/redo, and color picker on line-canvas.
 */

/** @type {Array<{start:{x,y}, end:{x,y}, color:string}>} */
let drawnLines = [];
/** @type {Array<{start:{x,y}, end:{x,y}, color:string}>} */
let redoLines = [];
let isDrawingMode = false;
let pendingStart = null;
let pointerPreview = null;

/** @type {HTMLCanvasElement} */
let canvas;
/** @type {CanvasRenderingContext2D} */
let ctx;
/** @type {HTMLInputElement} */
let colorPicker;

// UI elements
let toggleBtn, undoBtn, redoBtn, clearBtn, modeBadge;

const LINE_WIDTH = 3;

export function initDrawing(elements) {
  canvas = elements.lineCanvas;
  ctx = canvas.getContext('2d');
  colorPicker = elements.colorPicker;
  toggleBtn = elements.toggleBtn;
  undoBtn = elements.undoBtn;
  redoBtn = elements.redoBtn;
  clearBtn = elements.clearBtn;
  modeBadge = elements.modeBadge;

  toggleBtn.addEventListener('click', toggleMode);
  undoBtn.addEventListener('click', undo);
  redoBtn.addEventListener('click', redo);
  clearBtn.addEventListener('click', clearAll);

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerleave', onPointerLeave);
}

function toggleMode() {
  isDrawingMode = !isDrawingMode;
  toggleBtn.classList.toggle('active', isDrawingMode);
  canvas.style.pointerEvents = isDrawingMode ? 'auto' : 'none';
  canvas.style.cursor = isDrawingMode ? 'crosshair' : 'default';
  modeBadge.classList.toggle('hidden', !isDrawingMode);
  if (!isDrawingMode) {
    pendingStart = null;
    pointerPreview = null;
    redraw();
  }
  syncButtons();
}

function undo() {
  if (!drawnLines.length) return;
  redoLines.push(drawnLines.pop());
  pendingStart = null;
  pointerPreview = null;
  syncButtons();
  redraw();
}

function redo() {
  if (!redoLines.length) return;
  drawnLines.push(redoLines.pop());
  pendingStart = null;
  pointerPreview = null;
  syncButtons();
  redraw();
}

function clearAll() {
  drawnLines = [];
  redoLines = [];
  pendingStart = null;
  pointerPreview = null;
  syncButtons();
  redraw();
}

function syncButtons() {
  undoBtn.disabled = drawnLines.length === 0;
  redoBtn.disabled = redoLines.length === 0;
}

/**
 * Map a pointer clientX/Y to canvas pixel space, respecting object-fit: contain.
 */
function canvasPoint(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const canvasAspect = canvas.width / canvas.height;
  const rectAspect = rect.width / rect.height;
  let renderW, renderH, offX, offY;

  if (rectAspect > canvasAspect) {
    renderH = rect.height;
    renderW = renderH * canvasAspect;
    offX = (rect.width - renderW) / 2;
    offY = 0;
  } else {
    renderW = rect.width;
    renderH = renderW / canvasAspect;
    offX = 0;
    offY = (rect.height - renderH) / 2;
  }

  const bx = Math.min(Math.max(clientX - rect.left - offX, 0), renderW);
  const by = Math.min(Math.max(clientY - rect.top - offY, 0), renderH);
  return { x: (bx / renderW) * canvas.width, y: (by / renderH) * canvas.height };
}

function onPointerDown(e) {
  if (!isDrawingMode) return;
  e.preventDefault();
  const pt = canvasPoint(e.clientX, e.clientY);
  if (!pendingStart) {
    pendingStart = pt;
    pointerPreview = pt;
  } else {
    drawnLines.push({ start: pendingStart, end: pt, color: colorPicker.value });
    redoLines = [];
    pendingStart = null;
    pointerPreview = null;
    syncButtons();
  }
  redraw();
}

function onPointerMove(e) {
  if (!isDrawingMode || !pendingStart) return;
  pointerPreview = canvasPoint(e.clientX, e.clientY);
  redraw();
}

function onPointerLeave() {
  if (!isDrawingMode || !pendingStart) return;
  pointerPreview = null;
  redraw();
}

export function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const line of drawnLines) {
    drawLine(line.start, line.end, line.color, false);
  }

  if (isDrawingMode && pendingStart) {
    // Draw a dot at the first click point
    ctx.beginPath();
    ctx.arc(pendingStart.x, pendingStart.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = colorPicker.value;
    ctx.fill();

    if (pointerPreview) {
      drawLine(pendingStart, pointerPreview, colorPicker.value, true);
    }
  }
}

function drawLine(start, end, color, isDashed) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = LINE_WIDTH;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (isDashed) ctx.setLineDash([8, 6]);
  ctx.shadowColor = 'rgba(0,0,0,0.4)';
  ctx.shadowBlur = 4;
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  ctx.restore();
}

/**
 * Reset drawing state when a new video is loaded.
 */
export function resetDrawing() {
  drawnLines = [];
  redoLines = [];
  pendingStart = null;
  pointerPreview = null;
  syncButtons();
  redraw();
}
