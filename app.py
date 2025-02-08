from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
import numpy as np
import os
from deepface import DeepFace  # Optional: Adds DeepFace for better verification


# Initialize FastAPI app
app = FastAPI()
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files for output images
app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")


# Resize and standardize images (preserve aspect ratio)
def resize_image(image, max_size=600):
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    if scale < 1:  # Resize only if larger than max_size
        return cv2.resize(image, (int(width * scale), int(height * scale)))
    return image


# Compute similarity using Euclidean distance with normalization
def compute_similarity(feature1, feature2):
    if np.linalg.norm(feature1) == 0 or np.linalg.norm(feature2) == 0:
        return 1  # Maximum dissimilarity
    feature1, feature2 = feature1 / np.linalg.norm(feature1), feature2 / np.linalg.norm(feature2)
    return np.linalg.norm(feature1 - feature2)


# Normalize similarity scores (higher score means more similarity)
def normalize_score(distance, min_val=0, max_val=0.6):
    return max(0, min(1, (max_val - distance) / (max_val - min_val)))


# Draw bounding boxes around detected faces
def annotate_faces(image, face_locations, output_path):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imwrite(output_path, image)


# Extract facial feature points
def extract_features(landmarks):
    return {
        "eyes": np.array([landmarks["left_eye"], landmarks["right_eye"]]),
        "nose": np.array(landmarks["nose_bridge"]),
        "mouth": np.array(landmarks["top_lip"] + landmarks["bottom_lip"]),
        "jawline": np.array(landmarks["chin"]),
        "eyebrows": np.array([landmarks["left_eyebrow"], landmarks["right_eyebrow"]]),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        # Save images temporarily
        file1_path, file2_path = os.path.join(UPLOAD_FOLDER, image1.filename), os.path.join(UPLOAD_FOLDER, image2.filename)
        with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
            f1.write(await image1.read())
            f2.write(await image2.read())

        # Load and resize images
        img1, img2 = face_recognition.load_image_file(file1_path), face_recognition.load_image_file(file2_path)
        img1, img2 = resize_image(img1), resize_image(img2)

        # Get face encodings
        encodings1, encodings2 = face_recognition.face_encodings(img1), face_recognition.face_encodings(img2)
        landmarks1, landmarks2 = face_recognition.face_landmarks(img1), face_recognition.face_landmarks(img2)

        if not encodings1 or not encodings2:
            raise HTTPException(status_code=400, detail="No faces detected in one or both images.")

        # Extract facial features
        features1, features2 = extract_features(landmarks1[0]), extract_features(landmarks2[0])

        # Weighted feature similarity
        weights = {"eyes": 0.4, "nose": 0.3, "mouth": 0.2, "jawline": 0.1, "eyebrows": 0.05}
        feature_similarities = {feat: normalize_score(compute_similarity(features1[feat], features2[feat])) for feat in weights}

        # Compute overall similarity
        overall_sim = compute_similarity(encodings1[0], encodings2[0])
        similarity_percentage = round(100 * (1 - (overall_sim / 0.6)), 2)  # Adjusted similarity score
        similarity_percentage = max(0, min(100, similarity_percentage))  # Ensure within range

        # Perform DeepFace analysis for extra verification (optional)
        try:
            deepface_result = DeepFace.verify(file1_path, file2_path, model_name="VGG-Face", enforce_detection=False)
            deepface_match = deepface_result["verified"]
        except Exception:
            deepface_match = "Error in DeepFace"

        # Annotate face bounding boxes
        face_locations1, face_locations2 = face_recognition.face_locations(img1), face_recognition.face_locations(img2)
        output1_path, output2_path = os.path.join(OUTPUT_FOLDER, "annotated1.jpg"), os.path.join(OUTPUT_FOLDER, "annotated2.jpg")
        annotate_faces(img1.copy(), face_locations1, output1_path)
        annotate_faces(img2.copy(), face_locations2, output2_path)

        # Construct full URLs for output images
        image1_url = f"/outputs/annotated1.jpg"
        image2_url = f"/outputs/annotated2.jpg"

        # Determine similarity result
        similarity_result = "Highly Similar" if similarity_percentage >= 85 else (
            "Similar" if similarity_percentage >= 50 else "Moderately Similar" if similarity_percentage >=35
            else "Not Similar"
        )

        return JSONResponse({
            "similarity_result": similarity_result,
            "similarity_percentage": similarity_percentage,
            "overall_similarity_score": round(overall_sim, 4),
            "feature_similarities": feature_similarities,
            "deepface_match": deepface_match,
            "annotated_images": {
                "image1_url": image1_url,
                "image2_url": image2_url,
            },
        })

    except Exception as e:
        print(f"Error in upload route: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
