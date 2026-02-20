import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, c.
    Returns angle in degrees.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # We will just rely on our manual cv2.circle drawing in the main loop
    # to avoid needing the missing mp.solutions module entirely
    return annotated_image

def analyze_swing(video_path):
    if not video_path:
        print("Error: A video file path must be provided.")
        return

    # Initialize MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    
    # Use IMAGE mode to support seeking forwards and backwards without timestamp errors
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE)
        
    landmarker = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Fallback if FPS cannot be read
        
    frame_index = 0
    paused = False
    saved_frame_count = 0
    step_frame = 0
    annotated_image = None
    
    while True:
        if not paused or step_frame != 0:
            if step_frame == -1:
                # To go backward 1 frame, set position to frame_index - 2, 
                # next read() will advance to frame_index - 1
                frame_index = max(0, frame_index - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = cap.read()
            if not ret:
                if step_frame == 0:
                    print("Video analysis complete.")
                    break
                else:
                    step_frame = 0
                    continue
                    
            frame_index += 1
            step_frame = 0
            
            h, w, _ = frame.shape
            # Recolor image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Make detection using standard image detect mode
            detection_result = landmarker.detect(mp_image)
            
            # Draw basic landmarks
            annotated_image = frame.copy()
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                # MediaPipe Pose Landmarks: 11 (left shoulder), 13 (left elbow), 15 (left wrist)
                # 23 (left hip), 25 (left knee), 27 (left ankle)
                
                left_shoulder = [landmarks[11].x, landmarks[11].y]
                left_elbow = [landmarks[13].x, landmarks[13].y]
                left_wrist = [landmarks[15].x, landmarks[15].y]
                
                lead_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                left_hip = [landmarks[23].x, landmarks[23].y]
                left_knee = [landmarks[25].x, landmarks[25].y]
                left_ankle = [landmarks[27].x, landmarks[27].y]
                
                lead_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                # Visualize angles dynamically at the joint
                cv2.putText(annotated_image, f"{int(lead_arm_angle)} deg", 
                           tuple(np.multiply(left_elbow, [w, h]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                           
                cv2.putText(annotated_image, f"{int(lead_knee_angle)} deg", 
                           tuple(np.multiply(left_knee, [w, h]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                           
                # Draw circles on major landmarks
                for idx in [11, 13, 15, 23, 25, 27]:
                    pos = tuple(np.multiply([landmarks[idx].x, landmarks[idx].y], [w, h]).astype(int))
                    cv2.circle(annotated_image, pos, 5, (0, 255, 0), -1)
                    
            # Recommendations UI panel
            cv2.rectangle(annotated_image, (0, 0), (w, 140), (0, 0, 0), -1)
            cv2.putText(annotated_image, 'GOLF SWING ANALYSIS MVP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            y_offset = 60
            # Guard these recommendations against frames with 0 landmarks
            if detection_result.pose_landmarks:
                if lead_arm_angle < 160:
                    cv2.putText(annotated_image, "Recommendation: Keep lead arm straight", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    y_offset += 25
                else:
                    cv2.putText(annotated_image, "Lead arm straight: GOOD", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 25
                    
                if lead_knee_angle > 170:
                    cv2.putText(annotated_image, "Recommendation: Add knee bend", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                elif lead_knee_angle < 130:
                    cv2.putText(annotated_image, "Recommendation: Less knee bend", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(annotated_image, "Knee posture: GOOD", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
            # Controls UI
            y_offset += 30
            controls_text = "Controls: [SPACE] Play/Pause | [,] Prev | [.] Next | [s] Save | [q] Quit"
            cv2.putText(annotated_image, controls_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if annotated_image is not None:
            cv2.imshow('Golf Swing Analyzer', annotated_image)

        # Handle keyboard events
        # Wait 30ms to yield control to the GUI and roughly match 30fps playback
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to pause/resume
            paused = not paused
        elif key == ord('s'):  # 's' to save screenshot
            if annotated_image is not None:
                saved_frame_count += 1
                filename = f"keyframe_{saved_frame_count}.png"
                cv2.imwrite(filename, annotated_image)
                print(f"Saved snapshot to: {filename}")
        elif key == ord(','):
            step_frame = -1
            paused = True
        elif key == ord('.'):
            step_frame = 1
            paused = True

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_swing.py <path_to_video>")
        print("Example: python analyze_swing.py swing.mp4")
        sys.exit(1)
        
    video_file = sys.argv[1]
    
    print(f"Starting Golf Swing Analyzer on {video_file}...")
    print("Press 'q' in the video window to exit early.")
    analyze_swing(video_file)
