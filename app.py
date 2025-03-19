from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os

# Load the trained model
MODEL_PATH = "my_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define class labels
classes = {
    4: ("nv", "Melanocytic Nevi"),
    6: ("mel", "Melanoma"),
    2: ("bkl", "Benign Keratosis-Like Lesions"),
    1: ("bcc", "Basal Cell Carcinoma"),
    5: ("vasc", "Pyogenic Granulomas and Hemorrhage"),
    0: ("akiec", "Actinic Keratoses and Intraepithelial Carcinomae"),
    3: ("df", "Dermatofibroma"),
}

# Preprocess input image (Match local script)
def preprocess_image(image: Image.Image, target_size=(28, 28)):
    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert grayscale to RGB if needed
    if image.ndim == 2:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert from RGB (PIL) to BGR (OpenCV) to match cv2.imread()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize to target shape
    image = cv2.resize(image, target_size)

    # Standardize (matching local script)
    mean, std = np.mean(image), np.std(image)
    image = (image - mean) / std  

    # Ensure correct shape and dtype
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Shape: (1, 28, 28, 3)

    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file extension
        allowed_extensions = {"jpg", "jpeg", "png"}
        if file.filename.split(".")[-1].lower() not in allowed_extensions:
            return {"error": "Invalid file format. Please upload a JPG, JPEG, or PNG image."}
        
        # Read and preprocess image
        image = Image.open(BytesIO(await file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)  # Returns an array of probabilities
        
        # Check for NaN values
        if np.any(np.isnan(prediction)):
            return {"error": "Model output contains NaN values. Check model integrity."}

        # Extract the class index with highest probability
        predicted_class = int(np.argmax(prediction))

        # Get class label
        class_name, description = classes.get(predicted_class, ("unknown", "Unknown condition"))

        return {
            "prediction": predicted_class,
            "class_name": class_name,
            "description": description,
            "confidence": float(np.max(prediction))  # Return confidence score
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/use-sample")
async def use_sample():
    try:
        # Path to the sample image
        sample_image_path = "assets/sample.jpg"  # Ensure this image exists in the assets folder

        # Check if the file exists
        if not os.path.exists(sample_image_path):
            return {"error": "Sample image not found. Please check the assets folder."}

        # Load the sample image
        image = Image.open(sample_image_path)
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)  

        # Check for NaN values
        if np.any(np.isnan(prediction)):
            return {"error": "Model output contains NaN values. Check model integrity."}

        # Extract the class index with highest probability
        predicted_class = int(np.argmax(prediction))

        # Get class label
        class_name, description = classes.get(predicted_class, ("unknown", "Unknown condition"))

        return {
            "prediction": predicted_class,
            "class_name": class_name,
            "description": description,
            "confidence": float(np.max(prediction))  # Return confidence score
        }
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
