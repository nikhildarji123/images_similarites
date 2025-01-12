from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import face_recognition
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Standardize and resize image while maintaining aspect ratio
def resize_and_align(image):
    height, width = image.shape[:2]
    max_size = 600  # Define maximum size for the longest side
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size)
    return image

# Check if the image is valid (8-bit grayscale or RGB)
def is_valid_image(image):
    if image is None or image.dtype != np.uint8:
        return False
    if len(image.shape) not in [2, 3]:
        return False
    if len(image.shape) == 3 and image.shape[2] not in [3, 4]:  # RGB or RGBA
        return False
    return True

# Compute feature-wise similarity using Euclidean distance with normalization
def compute_feature_similarity(feature1, feature2):
    feature1_normalized = feature1 / np.linalg.norm(feature1) if np.linalg.norm(feature1) != 0 else feature1
    feature2_normalized = feature2 / np.linalg.norm(feature2) if np.linalg.norm(feature2) != 0 else feature2
    return np.linalg.norm(feature1_normalized - feature2_normalized)

# Visualize features by drawing rectangles around detected faces
def visualize_features(image, face_locations, output_path):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)

# Extract facial features based on landmarks (using face_recognition)
def extract_features(face_landmarks):
    features = {
        'eyes': np.array([face_landmarks['left_eye'], face_landmarks['right_eye']]),
        'nose': np.array(face_landmarks['nose_bridge']),
        'mouth': np.array(face_landmarks['top_lip'] + face_landmarks['bottom_lip']),
        'jawline': np.array(face_landmarks['chin']),
        'eyebrows': np.array([face_landmarks['left_eyebrow'], face_landmarks['right_eyebrow']])
    }
    return features

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Get uploaded files
        file1 = request.files["image1"]
        file2 = request.files["image2"]

        # Save files temporarily
        file1_path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file2_path = os.path.join(app.config["UPLOAD_FOLDER"], file2.filename)
        file1.save(file1_path)
        file2.save(file2_path)

        # Read and process images
        img1 = face_recognition.load_image_file(file1_path)
        img2 = face_recognition.load_image_file(file2_path)

        # Resize images for consistency if needed
        img1_resized = resize_and_align(img1)
        img2_resized = resize_and_align(img2)

        # Validate images before processing further
        if not is_valid_image(img1_resized) or not is_valid_image(img2_resized):
            raise ValueError("One or both uploaded files are not valid images.")

        # Get face encodings and landmarks
        encodings1 = face_recognition.face_encodings(img1_resized)
        encodings2 = face_recognition.face_encodings(img2_resized)

        landmarks1 = face_recognition.face_landmarks(img1_resized)
        landmarks2 = face_recognition.face_landmarks(img2_resized)

        if len(encodings1) == 0 or len(encodings2) == 0:
            raise ValueError("No faces found in one or both images.")

        # Extract facial features from landmarks (assuming one face per image)
        features1 = extract_features(landmarks1[0])
        features2 = extract_features(landmarks2[0])

        # Compute weighted feature-wise similarities
        weights = {
            'eyes': 0.4,
            'nose': 0.3,
            'mouth': 0.2,
            'jawline': 0.1,
            'eyebrows': 0.05,
        }

        weighted_similarity_score = sum(weights[feature] * compute_feature_similarity(features1[feature], features2[feature]) for feature in weights)

        # Overall similarity based on encodings
        overall_similarity = compute_feature_similarity(encodings1[0], encodings2[0])

        # Visualize features (bounding boxes around detected faces)
        face_locations1 = face_recognition.face_locations(img1_resized)
        face_locations2 = face_recognition.face_locations(img2_resized)

        output1_path = os.path.join(app.config["OUTPUT_FOLDER"], "annotated1.jpg")
        output2_path = os.path.join(app.config["OUTPUT_FOLDER"], "annotated2.jpg")

        visualize_features(img1_resized.copy(), face_locations1, output1_path)
        visualize_features(img2_resized.copy(), face_locations2, output2_path)

        similarity_score_percentage = max(0, 100 * (1 - (overall_similarity / 0.6)))  # Adjust threshold as needed

        # Determine similarity result based on overall score
        if similarity_score_percentage >= 85:
            similarity_result = "Faces are highly similar."
        elif similarity_score_percentage >= 50:
            similarity_result = "Faces are similar."
        elif similarity_score_percentage >= 30:
            similarity_result = "Faces are moderately similar."
        else:
            similarity_result = "Faces are not similar."

        # Prepare response with detailed results
        response = {
            "similarity_result": similarity_result,
            "annotated1": "annotated1.jpg",
            "annotated2": "annotated2.jpg",
            "accuracy": round(similarity_score_percentage, 2),
            "overall_similarity": round(overall_similarity, 4),
            "weighted_similarity": round(weighted_similarity_score * 100, 4),  # Convert to percentage for readability
            "feature_similarities": {feature: round(compute_feature_similarity(features1[feature], features2[feature]), 4) for feature in weights},
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in upload route: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route("/outputs/<filename>")
def outputs(filename):
    try:
       return send_from_directory(app.config["OUTPUT_FOLDER"], filename)
    except Exception as e:
       print(f"Error in outputs route: {str(e)}")
       return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
   app.run(debug=True)
