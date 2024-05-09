import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks is not None:
        keypoints = np.zeros((len(results.pose_landmarks.landmark), 2))
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y]
        return keypoints
    else:
        return None
    
def process_images():
    poses = []
    labels = []
    for i in range(1, 3360):
        image_path = f"C:/Users/dipto/Desktop/gym pose detection/dataset/{str(i).zfill(3)}"
        
        image_formats = ["jpg", "png", "webp", "jpeg", "PNG"]
        for ext in image_formats:
            full_path = f"{image_path}.{ext}"
            print("Trying to load:", full_path)
            image = cv2.imread(full_path)
            if image is not None:
                break
        
        if image is None:
            print("Error loading image:", image_path)
            continue
            
        keypoints = extract_pose(image)
        if keypoints is not None:
            poses.append(keypoints)
            
            if 1 <= i <= 784:
                labels.append(0)  # Bench Extended
            elif 785 <= i <= 1554:
                labels.append(1)  # Bench Compressed
            elif 1555 <= i <= 2030:
                labels.append(2)  # Shoulder Extended
            elif 2031 <= i <= 2520:
                labels.append(3)  # Shoulder Relaxed
            elif 2521 <= i <= 2940:
                labels.append(4)  # Squat Extended
            elif 2941 <= i <= 3360:
                labels.append(5)  # Squat Relaxed
    return np.array(poses), np.array(labels)

poses, labels = process_images()

poses = poses / 640 #Normalization

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((33, 2, 1), input_shape=(33, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(poses, labels, epochs=100, batch_size=32)

model.save("gym_pose_model")
