# Golf Swing Analyzer

A client-side interactive web application that uses computer vision to analyze a golf swing from prerecorded videos. This tool leverages Google's MediaPipe Pose Landmarker to extract real-time joint positions and calculate critical geometric angles to provide immediate posture recommendations.

## Features
- **Browser-Based Video Processing**: Analyze your `.mp4`, `.mov`, or `.webm` files entirely locally in your browser. No server uploads required, guaranteeing 100% privacy.
- **Pose Estimation**: Extracts body landmarks using the lightweight MediaPipe Tasks Vision API for extremely fast and accurate tracking.
- **Live Analysis**: Evaluates core golf mechanics:
  - **Lead Arm Extension**: Calculates the angle between your lead shoulder, elbow, and wrist.
  - **Athletic Posture (Knee Flexion)**: Calculates the angle between your lead hip, knee, and ankle.
- **Drawing Tools**: Draw lines and annotations directly on the video while paused to analyze angles and planes.
- **Performance Report**: Get a summary of your swing metrics including minimum/maximum angles and automated feedback.

## Prerequisites
- Node.js (v18+)
- npm or yarn

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Eric-Wong97/golf-swing-analyzer.git
   cd golf-swing-analyzer
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Model Setup**:
   The web app uses MediaPipe's Task Vision API. The lightweight pose landmarker model (`pose_landmarker_lite.task`) is already included in the root directory and will be served locally.

## Usage

Start the local Vite development server:

```bash
npm run dev
```

Then open your browser to the local URL provided (usually `http://localhost:5173`).

### Controls
Once the video player is open:
- Use the built-in video controls to play, pause, or seek.
- While paused, you can use the Drawing Mode (pencil icon) to annotate the frame.
- Click "Export" in the Performance Report section to download a summary.

## Limitations
- **Right-Handed Bias**: The current logic calculates angles based on the left shoulder, elbow, and knee. This makes it a perfect lead-side analyzer for right-handed golfers, but it currently tracks the trailing side for left-handed golfers.
- **Body Tracking Only**: Google's MediaPipe is highly optimized for tracking the human body. As a result, this app does not currently track the golf club shaft, the clubface angle, or the golf ball itself.
