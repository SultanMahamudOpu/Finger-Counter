import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def count_fingers(hand_landmarks, hand_label='Right'):

    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    
    if hand_label == 'Right':
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    
    for tip_id in finger_tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Face & Finger Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (350, 100), (50, 50, 50), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    total_fingers = 0
    right_count = 0
    left_count = 0
    hand_texts = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            count = count_fingers(hand_landmarks, hand_label)
            total_fingers += count
            if hand_label == "Right":
                right_count = count
            elif hand_label == "Left":
                left_count = count
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if len(results.multi_hand_landmarks) == 1:
            hand_texts.append(f"{hand_label} Hand: {count} Fingers")
        else:
            hand_texts.append(f"Right: {right_count}  Left: {left_count}")
            hand_texts.append(f"Total: {total_fingers}")
    else:
        hand_texts.append("Raise your hand(s) to count fingers")
    
    for i, text in enumerate(hand_texts):
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face & Finger Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
