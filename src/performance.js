/**
 * performance.js
 * Tracks per-frame angle data and generates the performance report + export.
 */

/** @type {{ armAngles: number[], kneeAngles: number[], totalFrames: number, detectedFrames: number }} */
let data = makeEmpty();

function makeEmpty() {
  return { armAngles: [], kneeAngles: [], totalFrames: 0, detectedFrames: 0 };
}

export function resetPerformanceData() {
  data = makeEmpty();
}

export function recordFrame({ armAngle, kneeAngle, detected }) {
  data.totalFrames++;
  if (detected) {
    data.detectedFrames++;
    data.armAngles.push(armAngle);
    data.kneeAngles.push(kneeAngle);
  }
}

export function hasData() {
  return data.armAngles.length > 0;
}

function avg(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Populate the report DOM elements.
 * @param {HTMLVideoElement} video
 */
export function updateReport(video) {
  if (!hasData()) return;

  const { armAngles, kneeAngles, totalFrames, detectedFrames } = data;
  const armAvg = avg(armAngles);
  const armMin = Math.min(...armAngles);
  const armMax = Math.max(...armAngles);
  const kneeAvg = avg(kneeAngles);
  const kneeMin = Math.min(...kneeAngles);
  const kneeMax = Math.max(...kneeAngles);

  // Session
  set('report-duration', video.duration ? `${video.duration.toFixed(1)}s` : '--');
  set('report-frames', totalFrames);
  set('report-detection', `${((detectedFrames / totalFrames) * 100).toFixed(1)}%`);

  // Arm
  set('report-arm-current', `${Math.round(armAngles.at(-1))}°`);
  set('report-arm-avg', `${Math.round(armAvg)}°`);
  set('report-arm-min', `${Math.round(armMin)}°`);
  set('report-arm-max', `${Math.round(armMax)}°`);
  const armGoodPct = (armAngles.filter((a) => a >= 160).length / armAngles.length) * 100;
  set(
    'report-arm-assessment',
    armAvg >= 160
      ? `Excellent — ${armGoodPct.toFixed(0)}% of frames in good range`
      : `Needs improvement — only ${armGoodPct.toFixed(0)}% of frames in good range`
  );

  // Knee
  set('report-knee-current', `${Math.round(kneeAngles.at(-1))}°`);
  set('report-knee-avg', `${Math.round(kneeAvg)}°`);
  set('report-knee-min', `${Math.round(kneeMin)}°`);
  set('report-knee-max', `${Math.round(kneeMax)}°`);
  const kneeGoodPct =
    (kneeAngles.filter((k) => k >= 130 && k <= 170).length / kneeAngles.length) * 100;
  set(
    'report-knee-assessment',
    kneeAvg >= 130 && kneeAvg <= 170
      ? `Excellent — ${kneeGoodPct.toFixed(0)}% of frames in optimal range`
      : `Needs adjustment — only ${kneeGoodPct.toFixed(0)}% of frames in optimal range`
  );

  // Recommendations
  const recs = buildRecommendations(armAngles, armAvg, armMax, armMin, kneeAvg);
  const recList = document.getElementById('recommendations-list');
  recList.innerHTML = recs.map((r) => `<li class="${r.type}">${r.text}</li>`).join('');

  document.getElementById('performance-report').classList.remove('hidden');
}

function buildRecommendations(armAngles, armAvg, armMax, armMin, kneeAvg) {
  const recs = [];
  if (armAvg < 160) {
    recs.push({
      text: `Focus on keeping your lead arm straighter. Average angle was ${Math.round(armAvg)}°, target is ≥ 160°.`,
      type: 'warning',
    });
  } else {
    recs.push({
      text: `Great job maintaining lead arm straightness! Average ${Math.round(armAvg)}° is in the optimal range.`,
      type: 'success',
    });
  }
  if (kneeAvg < 130) {
    recs.push({
      text: `Your knees are too bent. Average was ${Math.round(kneeAvg)}°, optimal is 130–170°.`,
      type: 'warning',
    });
  } else if (kneeAvg > 170) {
    recs.push({
      text: `Add more knee flex for better athletic posture. Average was ${Math.round(kneeAvg)}°, optimal is 130–170°.`,
      type: 'warning',
    });
  } else {
    recs.push({
      text: `Perfect knee posture! Average ${Math.round(kneeAvg)}° is in the optimal athletic range.`,
      type: 'success',
    });
  }
  const armRange = armMax - armMin;
  if (armRange > 30) {
    recs.push({
      text: `Work on consistency — your lead arm angle varied by ${Math.round(armRange)}° throughout the swing.`,
      type: 'warning',
    });
  }
  return recs;
}

function set(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

/**
 * Export the current report as a plain-text file download.
 */
export function exportReport(video) {
  if (!hasData()) {
    alert('No data to export yet. Please analyze a video first.');
    return;
  }
  const { armAngles, kneeAngles, totalFrames, detectedFrames } = data;
  const armAvg = avg(armAngles);
  const kneeAvg = avg(kneeAngles);

  const text = `
GOLF SWING ANALYZER — PERFORMANCE REPORT
Generated: ${new Date().toLocaleString()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SESSION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Video Duration : ${video.duration ? video.duration.toFixed(1) : '--'} seconds
Frames Analyzed: ${totalFrames}
Detection Rate : ${((detectedFrames / totalFrames) * 100).toFixed(1)}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEAD ARM ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current : ${Math.round(armAngles.at(-1))}°
Average : ${Math.round(armAvg)}°
Min     : ${Math.round(Math.min(...armAngles))}°
Max     : ${Math.round(Math.max(...armAngles))}°
Threshold : ≥ 160° (Good)
Good Frames: ${((armAngles.filter((a) => a >= 160).length / armAngles.length) * 100).toFixed(0)}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNEE POSTURE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current : ${Math.round(kneeAngles.at(-1))}°
Average : ${Math.round(kneeAvg)}°
Min     : ${Math.round(Math.min(...kneeAngles))}°
Max     : ${Math.round(Math.max(...kneeAngles))}°
Optimal Range : 130° – 170°
Good Frames: ${((kneeAngles.filter((k) => k >= 130 && k <= 170).length / kneeAngles.length) * 100).toFixed(0)}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${document.getElementById('recommendations-list')?.innerText ?? ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by Golf Swing Analyzer — https://github.com/Eric-Wong97/golf-swing-analyzer
`;

  const blob = new Blob([text.trim()], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), {
    href: url,
    download: `golf-swing-report-${new Date().toISOString().split('T')[0]}.txt`,
  });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
