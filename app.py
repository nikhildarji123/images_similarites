# Streamlit app.py
import streamlit as st
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import tempfile
import os

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to extract facial features
def get_facial_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, None, None

    face = faces[0]
    landmarks = predictor(gray, face)

    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    features = {
        "jawline": points[0:17],
        "right_eyebrow": points[17:22],
        "left_eyebrow": points[22:27],
        "nose": points[27:36],
        "right_eye": points[36:42],
        "left_eye": points[42:48],
        "mouth": points[48:68],
    }

    return img, features, face

# Calculate feature ratios
def calculate_feature_ratios(features, face):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_width = x2 - x1
    face_height = y2 - y1

    ratios = {}
    for feature, points in features.items():
        if feature in ["right_eye", "left_eye", "mouth"]:
            area = cv2.contourArea(np.array(points, dtype=np.int32).reshape(-1, 1, 2))
        else:
            x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
            area = w * h

        ratios[feature] = abs(area) / (face_width * face_height) if (face_width * face_height) != 0 else 0

    return ratios

# Compare feature ratios
def compare_feature_ratios(ratios1, ratios2):
    similarity_scores = {}
    for feature in ratios1.keys():
        ratio1 = ratios1[feature]
        ratio2 = ratios2[feature]

        if ratio1 is not None and ratio2 is not None:
            max_ratio = max(ratio1, ratio2)
            min_ratio = min(ratio1, ratio2)
            similarity = (1 - abs(max_ratio - min_ratio)/max_ratio)*100 if max_ratio !=0 else 100
            similarity_scores[feature] = round(similarity, 2)
        else:
            similarity_scores[feature] = None

    return similarity_scores

# Visualize faces with detected features
def visualize_faces(img1, img2, features1, features2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, img, features in zip(axes, [img1, img2], [features1, features2]):
        for feature, points in features.items():
            points = np.array(points, dtype=np.int32)
            if feature in ["right_eye", "left_eye", "mouth"]:
                cv2.fillPoly(img, [points], (0, 255, 0))
            else:
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    cv2.line(img, tuple(start), tuple(end), (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

    return fig

# Main Streamlit app
st.title("Face Comparison Analysis")
st.write("Upload two face images to compare using traditional feature analysis and deep learning")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    image1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    if image1 is not None:
        st.image(image1, caption="Uploaded Image 1", use_column_width=True)

with col2:
    image2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])
    if image2 is not None:
        st.image(image2, caption="Uploaded Image 2", use_column_width=True)

if st.button("Compare Faces"):
    if image1 is None or image2 is None:
        st.error("Please upload both images")
    else:
        # Save images to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
            tmp1.write(image1.getvalue())
            image1_path = tmp1.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
            tmp2.write(image2.getvalue())
            image2_path = tmp2.name

        # Process with traditional method
        st.header("Facial Feature Analysis")
        img1, features1, face1 = get_facial_features(image1_path)
        img2, features2, face2 = get_facial_features(image2_path)
        
        if img1 is None or img2 is None:
            st.error("Error loading images. Please ensure valid image files are uploaded.")
        elif features1 is None or features2 is None:
            st.error("No faces detected in one or both images")
        else:
            ratios1 = calculate_feature_ratios(features1, face1)
            ratios2 = calculate_feature_ratios(features2, face2)
            similarity_scores = compare_feature_ratios(ratios1, ratios2)
            
            avg_similarity = np.mean([score for score in similarity_scores.values() if score is not None])
            
            st.write("### Facial Feature Similarity Scores:")
            for feature, score in similarity_scores.items():
                st.write(f"- {feature.capitalize()}: {score}%")
                
                # Display images with facial landmarks
            st.write("#### Analyzed Images with Detected Features")
            fig = visualize_faces(img1.copy(), img2.copy(), features1, features2)
                
        
        # Process with DeepFace
        st.header("Deep Learning Analysis")
        try:
            result = DeepFace.verify(image1_path, image2_path, 
                                    model_name="Facenet512", 
                                    enforce_detection=False)
            
            similarity = round(100 - (result['distance'] * 100), 2)
            threshold = round(result['threshold'] * 100, 2)
            
            if similarity > 50:
                st.write(f"Similarity Score: {similarity}%")
                st.write(f"Threshold: {threshold}%")
                
                # Display images in results
                st.write("#### Uploaded Images")
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.image(image1, caption="Image 1", use_column_width=True)
                with col_result2:
                    st.image(image2, caption="Image 2", use_column_width=True)
                
                if similarity >= threshold:
                    st.success("✅ Faces are verified to be the same person!")
                else:
                    st.error("❌ Faces are verified to be different people")
            else:
                st.warning("Deep learning similarity score is below 50%. No further details displayed.")
                
        except Exception as e:
            st.error(f"DeepFace analysis failed: {str(e)}")
        
        # Cleanup temporary files
        os.unlink(image1_path)
        os.unlink(image2_path)