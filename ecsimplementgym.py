import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("gym_pose_model")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5)

smoothing_factor = 0.2

prev_prediction = np.zeros(6)

def extract_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks is not None:
        keypoints = np.zeros((len(results.pose_landmarks.landmark), 2))
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y]
        return keypoints, results.pose_landmarks
    else:
        return None, None

def predict_pose(keypoints):
    keypoints = keypoints / 640 
    prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
    return prediction

def all_keypoints_visible(keypoints):
    return keypoints is not None and len(keypoints) == 33

def smooth_predictions(current_prediction):
    global prev_prediction
    if prev_prediction.shape != current_prediction.shape:
        prev_prediction = np.zeros_like(current_prediction)
    smoothed_prediction = smoothing_factor * current_prediction + (1 - smoothing_factor) * prev_prediction
    prev_prediction = smoothed_prediction
    return smoothed_prediction

def detect_and_predict_pose(image):
    keypoints, landmarks = extract_pose(image)
    if keypoints is None:
        error_text = "Error: Person out of frame"
        cv2.putText(image, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return image
    
    prediction = predict_pose(keypoints)
    smoothed_prediction = smooth_predictions(prediction)
    max_confidence = np.max(smoothed_prediction)
    for i, confidence in enumerate(smoothed_prediction):
        accuracy = (confidence / max_confidence) * 100
        accuracy_text = f"{i}: {round(accuracy, 2)}%"
        if accuracy > 90:
            pass
        cv2.putText(image, accuracy_text, (10, 30 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)
    return image

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_and_predict_pose(frame)
        cv2.imshow('gym Pose Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

