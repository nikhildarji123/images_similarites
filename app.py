from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import face_recognition
import numpy as np
import os
from deepface import DeepFace

app = FastAPI()
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")

app.mount("/outputs", StaticFiles(directory=OUTPUT_FOLDER), name="outputs")

def resize_image(image, max_size=600):
    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    return cv2.resize(image, (int(width * scale), int(height * scale))) if scale < 1 else image

def compute_similarity(feature1, feature2):
    if np.linalg.norm(feature1) == 0 or np.linalg.norm(feature2) == 0:
        return 1  
    return np.linalg.norm(feature1 / np.linalg.norm(feature1) - feature2 / np.linalg.norm(feature2))

def normalize_score(distance, min_val=0, max_val=0.6):
    return max(0, min(1, (max_val - distance) / (max_val - min_val)))

def draw_colored_arrows(image, landmarks, output_path):
    thickness = 5
    arrow_features = {
        "eye_line": (landmarks["left_eye"][0], landmarks["right_eye"][-1], (255, 0, 0)),
        "nose_mouth": (landmarks["nose_bridge"][-1], landmarks["top_lip"][0], (0, 255, 0)),
        "eyebrows": (landmarks["left_eyebrow"][-1], landmarks["right_eyebrow"][0], (0, 0, 255)),
    }

    for key, (start, end, color) in arrow_features.items():
        if start is not None and end is not None:
            cv2.arrowedLine(image, tuple(start), tuple(end), color, thickness, tipLength=0.2)

    cv2.imwrite(output_path, image)

def extract_features(landmarks):
    return {
        "eyes": np.array([landmarks["left_eye"], landmarks["right_eye"]]),
        "nose": np.array(landmarks["nose_bridge"]),
        "mouth": np.array(landmarks["top_lip"] + landmarks["bottom_lip"]),
        "jawline": np.array(landmarks["chin"]),
        "eyebrows": np.array([landmarks["left_eyebrow"], landmarks["right_eyebrow"]]),
    }

def is_side_angle_face(landmarks):
    nose_bridge = landmarks["nose_bridge"]
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    if nose_bridge and left_eye and right_eye:
        nose_tip = nose_bridge[-1]
        left_eye_x = np.mean([point[0] for point in left_eye])
        right_eye_x = np.mean([point[0] for point in right_eye])

        if nose_tip[0] < left_eye_x or nose_tip[0] > right_eye_x:
            return True
    return False

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        file1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
        file2_path = os.path.join(UPLOAD_FOLDER, image2.filename)
        with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
            f1.write(await image1.read())
            f2.write(await image2.read())
        img1 = face_recognition.load_image_file(file1_path)
        img2 = face_recognition.load_image_file(file2_path)
        img1, img2 = resize_image(img1), resize_image(img2)

        encodings1, encodings2 = face_recognition.face_encodings(img1), face_recognition.face_encodings(img2)
        landmarks1, landmarks2 = face_recognition.face_landmarks(img1), face_recognition.face_landmarks(img2)

        if not encodings1 or not encodings2:
            raise HTTPException(status_code=400, detail="No faces detected in one or both images.")

        if is_side_angle_face(landmarks1[0]) or is_side_angle_face(landmarks2[0]):
            raise HTTPException(status_code=400, detail="Side-angle faces detected. Please use front-facing images for better accuracy.")

        features1, features2 = extract_features(landmarks1[0]), extract_features(landmarks2[0])

        feature_similarities = {
            feat: normalize_score(compute_similarity(features1[feat], features2[feat]))
            for feat in features1
        }

        similarity_percentage = 0

        overall_sim = compute_similarity(encodings1[0], encodings2[0])
        similarity_percentage = round(100 * (1 - (overall_sim / 0.6)), 2)
        similarity_percentage = max(0, min(100, similarity_percentage))

        models = ["VGG-Face", "Facenet", "ArcFace"]
        deepface_results = {}
        for model in models:
            try:
                result = DeepFace.verify(file1_path, file2_path, model_name=model, enforce_detection=False)
                deepface_results[model] = result["verified"]
            except Exception as e:
                deepface_results[model] = f"Error: {str(e)}"

        output1_path = os.path.join(OUTPUT_FOLDER, "annotated1.jpg")
        output2_path = os.path.join(OUTPUT_FOLDER, "annotated2.jpg")
        draw_colored_arrows(img1.copy(), landmarks1[0], output1_path)
        draw_colored_arrows(img2.copy(), landmarks2[0], output2_path)

        image1_url = f"/outputs/annotated1.jpg"
        image2_url = f"/outputs/annotated2.jpg"

        similarity_result = "Highly Similar" if similarity_percentage >= 85 else (
            "Similar" if similarity_percentage >= 50 
            else "Moderately Similar" if similarity_percentage >= 35 else "Not Similar" 
        )

        return JSONResponse({
            "similarity_result": similarity_result,
            "similarity_percentage": similarity_percentage,
            
            "feature_similarities": feature_similarities,
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
