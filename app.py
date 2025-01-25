from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
import numpy as np
import os
from fastapi import Request


# Initialize FastAPI app
app = FastAPI()
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files for outputs
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

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

# Normalize feature similarity to range [0, 1]
def normalize_similarity(computed_distance, min_distance, max_distance):
    if computed_distance < min_distance:
        return 1.0
    elif computed_distance > max_distance:
        return 0.0
    else:
        return (max_distance - computed_distance) / (max_distance - min_distance)

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        # Save files temporarily
        file1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
        file2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
        
        with open(file1_path, "wb") as buffer:
            buffer.write(await image1.read())
        
        with open(file2_path, "wb") as buffer:
            buffer.write(await image2.read())

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

        # Compute weighted feature-wise similarities with normalization
        weights = {
            'eyes': 0.4,
            'nose': 0.3,
            'mouth': 0.2,
            'jawline': 0.1,
            'eyebrows': 0.05,
        }

        feature_similarity_scores = {}
        
        for feature in weights:
            computed_distance = compute_feature_similarity(features1[feature], features2[feature])
            min_distance = 0.0
            max_distance = 0.5
            
            normalized_score = normalize_similarity(computed_distance, min_distance, max_distance)
            feature_similarity_scores[feature] = round(normalized_score, 4)

        # Overall similarity based on encodings (not normalized here)
        overall_similarity = compute_feature_similarity(encodings1[0], encodings2[0])

        # Visualize features (bounding boxes around detected faces)
        face_locations1 = face_recognition.face_locations(img1_resized)
        face_locations2 = face_recognition.face_locations(img2_resized)

        output1_path = os.path.join(OUTPUT_FOLDER, "annotated1.jpg")
        output2_path = os.path.join(OUTPUT_FOLDER, "annotated2.jpg")

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

         # Prepare response with detailed results including normalized feature similarities
        response_data = {
            "similarity_result": similarity_result,
            "annotated1": "annotated1.jpg",
            "annotated2": "annotated2.jpg",
            "accuracy": round(similarity_score_percentage, 2),
            "overall_similarity": round(overall_similarity, 4),
            "feature_similarities": feature_similarity_scores,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        print(f"Error in upload route: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
