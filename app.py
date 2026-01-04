import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
from ultralytics import YOLO
import numpy as np
from directkeys import (PressKey, ReleaseKey, up_arrow_pressed, 
                        down_arrow_pressed, left_arrow_pressed, 
                        right_arrow_pressed, space_pressed)
import time
import torch
import mediapipe as mp
from flask import Flask, Response

app = Flask(__name__)

# --- CẤU HÌNH YOLO POSE (GIỮ NGUYÊN TỪ FILE GỐC) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Tự động dùng CUDA nếu có để tránh lag
model = YOLO('yolov8n-pose.pt').to("cuda" if torch.cuda.is_available() else "cpu")

origin_x, origin_y = 320, 360
neutral_radius = 50
update_interval = 0.6
last_update_time = time.time()
current_key_pressed = set()

# --- CẤU HÌNH HAND TRACKING (GIỮ NGUYÊN TỪ MAIN.PY) ---
detector = HandDetector(detectionCon=0.8, maxHands=1)
move_threshold = 100
upper_threshold = 200
lower_threshold = 300

def press_key_if_needed(key, action):
    global current_key_pressed
    if action == 'press' and key not in current_key_pressed:
        PressKey(key)
        current_key_pressed.add(key)
    elif action == 'release' and key in current_key_pressed:
        ReleaseKey(key)
        current_key_pressed.remove(key)

def update_keys_pose(new_keys):
    global current_key_pressed
    keys_to_press = set(new_keys)
    for key in current_key_pressed - keys_to_press:
        ReleaseKey(key)
    for key in keys_to_press - current_key_pressed:
        PressKey(key)
    current_key_pressed = keys_to_press

def gen_yolo_pose_frames():
    global origin_x, origin_y, last_update_time
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.resize(frame, (640, 720))
        results = model(frame, verbose=False)[0]
        blank_image = np.zeros((720, 640, 3), dtype=np.uint8)
        annotated_frame = results.plot() # Vẽ skeleton mặc định của YOLO
        
        position = "NEUTRAL"
        keys_to_press = []

        if results.keypoints is not None and len(results.keypoints.data) > 0:
            for kp_data in results.keypoints.data:
                kp = kp_data.cpu().numpy()
                center_x, center_y, conf = kp[0] # Nose
                
                if conf > 0.5:
                    if time.time() - last_update_time > update_interval:
                        origin_x, origin_y = int(center_x), int(center_y)
                        last_update_time = time.time()

                    dx, dy = int(center_x) - origin_x, origin_y - int(center_y)
                    if not (abs(dx) <= neutral_radius and abs(dy) <= neutral_radius):
                        if abs(dy) > abs(dx):
                            if dy > 0: position = "UP"; keys_to_press.append(up_arrow_pressed)
                            else: position = "DOWN"; keys_to_press.append(down_arrow_pressed)
                        else:
                            if dx > 0: position = "RIGHT"; keys_to_press.append(right_arrow_pressed)
                            else: position = "LEFT"; keys_to_press.append(left_arrow_pressed)

                    update_keys_pose(keys_to_press)
                    
                    # Giữ nguyên phần vẽ nhãn và khung trung tâm từ code cũ
                    cv2.putText(blank_image, position, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.line(blank_image, (origin_x, 0), (origin_x, 720), (0, 255, 0), 1)
                    cv2.line(blank_image, (0, origin_y), (640, origin_y), (0, 255, 0), 1)
                    cv2.rectangle(blank_image, (origin_x - 50, origin_y - 50), (origin_x + 50, origin_y + 50), (255, 255, 0), 2)

        # Gộp hình ảnh theo tỉ lệ cũ
        output = cvzone.stackImages([annotated_frame, blank_image], cols=2, scale=0.8)
        _, buffer = cv2.imencode('.jpg', output)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def gen_hand_tracking_frames():
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Giữ nguyên logic vẽ và nhãn từ main.py
        hands, img = detector.findHands(frame)
        h, w, _ = frame.shape
        center_x = w // 2

        if hands:
            lmList = hands[0]['lmList']
            fingers = detector.fingersUp(hands[0])
            hand_x, hand_y = lmList[9][0], lmList[9][1]
            
            # Label "Skill Activated" và các phím nhấn giữ nguyên logic cũ
            if fingers == [0, 0, 0, 0, 0]:
                cv2.putText(frame, 'Skill Activated', (380, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                press_key_if_needed(space_pressed, 'press')
            elif hand_y < upper_threshold:
                cv2.putText(frame, 'Jumping', (420, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                press_key_if_needed(up_arrow_pressed, 'press')
                press_key_if_needed(down_arrow_pressed, 'release')
            elif hand_y > lower_threshold:
                cv2.putText(frame, 'Crouching', (420, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                press_key_if_needed(down_arrow_pressed, 'press')
                press_key_if_needed(up_arrow_pressed, 'release')
            elif hand_x > center_x + move_threshold:
                cv2.putText(frame, 'Moving Left', (420, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                press_key_if_needed(left_arrow_pressed, 'press')
                press_key_if_needed(right_arrow_pressed, 'release')
            elif hand_x < center_x - move_threshold:
                cv2.putText(frame, 'Moving Right', (420, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                press_key_if_needed(right_arrow_pressed, 'press')
                press_key_if_needed(left_arrow_pressed, 'release')
            else:
                cv2.putText(frame, 'Neutral Position', (350, 460), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                for key in [up_arrow_pressed, down_arrow_pressed, left_arrow_pressed, right_arrow_pressed, space_pressed]:
                    press_key_if_needed(key, 'release')

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_pose')
def video_yolo():
    return Response(gen_yolo_pose_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_hand')
def video_hand():
    return Response(gen_hand_tracking_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)