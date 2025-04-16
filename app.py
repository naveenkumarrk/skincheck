from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
from typing import Optional
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

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

# Together API configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY","121fc570d5c53716122134db6a816ce5c9cdbdcb70bbc131584410c8327c7dcd")
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"

def generate_together_response(prompt, temperature=0.7, max_tokens=500):
    """Generate a response using the Together API"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", # You can change this to your preferred model
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40
    }
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error calling Together API: {str(e)}")
        return f"I'm sorry, I encountered an error processing your request. Please try again later."

def preprocess_image(image: Image.Image, target_size=(28, 28)):
    image = np.array(image)

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

# Chatbot responses for skin conditions
chatbot_responses = {
    "general": [
        "Regular skin checks are crucial for early detection of skin cancer. Look for changes in size, color, shape, or texture of moles.",
        "Always use sunscreen with SPF 30+ when outdoors, even on cloudy days.",
        "If you notice any unusual skin changes, consult a dermatologist immediately.",
        "Monthly self-examinations can help catch skin cancer early when it's most treatable.",
        "The ABCDE rule for melanoma: Asymmetry, Border irregularity, Color variations, Diameter >6mm, Evolving features."
    ],
    "nv": [
        "Melanocytic nevi (moles) are usually harmless but should be monitored for changes.",
        "Take photos of your moles periodically to track any changes over time.",
        "Most people have between 10-40 moles by adulthood. This is normal.",
        "Consider having a dermatologist check any moles that concern you."
    ],
    "mel": [
        "Melanoma is serious but highly treatable when caught early.",
        "Regular skin checks and prompt medical attention are essential for melanoma management.",
        "Reduce risk by limiting sun exposure, using sunscreen, and avoiding tanning beds.",
        "Schedule an immediate appointment with a dermatologist for proper diagnosis and treatment."
    ],
    "bkl": [
        "Benign keratosis-like lesions are non-cancerous growths that appear with age.",
        "While typically harmless, have a dermatologist evaluate any concerning lesions.",
        "Protect affected areas from sun exposure with clothing or sunscreen.",
        "Treatment options include cryotherapy or topical medications if the lesions are bothersome."
    ],
    "bcc": [
        "Basal cell carcinoma is the most common form of skin cancer and rarely spreads.",
        "Early treatment is important to prevent damage to surrounding tissue.",
        "After treatment, follow-up regularly with your dermatologist.",
        "Protect yourself with sun-protective clothing and regular sunscreen use."
    ],
    "vasc": [
        "Pyogenic granulomas are benign vascular growths that can bleed easily.",
        "They often appear after injury and may need surgical removal if problematic.",
        "Apply pressure if bleeding occurs and seek medical attention.",
        "These growths are not cancerous but should be evaluated by a dermatologist."
    ],
    "akiec": [
        "Actinic keratoses are pre-cancerous lesions caused by sun damage.",
        "Early treatment can prevent progression to squamous cell carcinoma.",
        "Topical treatments or procedures like cryotherapy are common management approaches.",
        "Use sun protection daily to prevent new lesions from forming."
    ],
    "df": [
        "Dermatofibromas are benign, firm nodules that often appear on the legs.",
        "They typically don't require treatment unless they cause discomfort.",
        "These growths may darken after sun exposure.",
        "If a lesion changes in appearance, consult with a dermatologist."
    ]
}

# Basic chatbot endpoint with static responses
@app.post("/chat")
async def chat(message: str = Form(...), condition: Optional[str] = Form(None)):
    """
    Process user messages and provide dermatological advice
    
    Parameters:
    - message: User's input message
    - condition: Optional skin condition detected from previous analysis
    """
    message = message.lower()
    
    # Check for specific skin condition queries
    if condition and condition in chatbot_responses:
        # Provide specific advice for the detected condition
        responses = chatbot_responses[condition]
        return {"response": responses[0], "additional_info": responses[1:3]}
    
    # Handle common queries
    if any(word in message for word in ["protect", "prevention", "sunscreen", "sun"]):
        return {
            "response": "Sun protection is critical in preventing skin cancer.",
            "additional_info": [
                "Use broad-spectrum SPF 30+ sunscreen daily, even when cloudy.",
                "Wear protective clothing, hats, and seek shade between 10am-4pm."
            ]
        }
    
    elif any(word in message for word in ["check", "self-exam", "examine", "look"]):
        return {
            "response": "Regular self-exams are important for early detection.",
            "additional_info": [
                "Use the ABCDE method: check for Asymmetry, Border irregularity, Color changes, Diameter >6mm, and Evolution.",
                "Examine your skin monthly and document any changes with photos."
            ]
        }
    
    elif any(word in message for word in ["doctor", "dermatologist", "when", "visit"]):
        return {
            "response": "See a dermatologist immediately if you notice:",
            "additional_info": [
                "New or changing moles, sores that don't heal, or unusual skin growths.",
                "Annual skin checks are recommended, especially for those with risk factors."
            ]
        }
    
    elif any(word in message for word in ["melanoma", "cancer", "serious"]):
        return {
            "response": "Early detection of melanoma significantly improves treatment outcomes.",
            "additional_info": [
                "Melanoma can be life-threatening but is highly treatable when caught early.",
                "Risk factors include fair skin, history of sunburns, and family history of skin cancer."
            ]
        }
    
    elif any(word in message for word in ["treat", "treatment", "options"]):
        return {
            "response": "Treatment depends on the specific diagnosis and stage.",
            "additional_info": [
                "Options may include surgical removal, topical treatments, radiation, or immunotherapy.",
                "Always follow your dermatologist's recommended treatment plan."
            ]
        }
    
    # Default response for other queries
    else:
        return {
            "response": "I'm your dermatology assistant focused on skin cancer awareness.",
            "additional_info": [
                "I can provide information about skin cancer types, prevention, and detection.",
                "For proper diagnosis, always consult with a qualified healthcare professional."
            ]
        }

# Advanced dynamic chatbot endpoint using the Together API
@app.post("/ask-dermatologist")
async def ask_dermatologist(question: str = Form(...), condition: Optional[str] = Form(None)):
    """
    Process user questions using the Together API to provide dynamic, personalized dermatological advice
    
    Parameters:
    - question: User's question about skin health
    - condition: Optional skin condition detected from previous analysis
    """
    try:
        # Format prompt based on whether a condition was detected
        condition_context = ""
        if condition and condition in chatbot_responses:
            condition_name = next((desc for code, (abbr, desc) in classes.items() if abbr == condition), "Unknown")
            condition_context = f"The user has uploaded an image that was classified as {condition_name} ({condition})."
        
        # Create the prompt for the Together API
        prompt = f"""You are an expert dermatologist specializing in skin cancer and skin conditions. 
Provide a helpful, concise response to the user's question. Keep your answer short (3-5 sentences maximum), 
factually accurate, and focused on evidence-based medicine. If appropriate, organize key points in a bullet list format.

{condition_context}

User question: {question}

Remember:
- Always recommend consulting a healthcare professional for diagnosis and treatment
- Be clear, concise, and educational
- Provide actionable advice where appropriate
- Focus on scientifically validated information

Your response:"""

        # Call the Together API
        response_text = generate_together_response(prompt, temperature=0.7, max_tokens=500)
        
        # Extract key points (if any) from the response
        lines = response_text.split('\n')
        main_response = lines[0] if lines else response_text
        additional_info = []
        
        for line in lines[1:]:
            # Look for bullet points or numbered lists
            clean_line = line.strip()
            if clean_line and (clean_line.startswith('-') or clean_line.startswith('•') or 
                              (len(clean_line) > 1 and clean_line[0].isdigit() and clean_line[1] == '.')):
                additional_info.append(clean_line.lstrip('- •0123456789. '))
        
        # If no bullet points were found but we have multiple paragraphs, use those
        if not additional_info and len(lines) > 1:
            main_response = lines[0]
            additional_info = [line.strip() for line in lines[1:] if line.strip()]
        
        return {
            "response": main_response,
            "additional_info": additional_info[:3]  # Limit to 3 additional points
        }
    
    except Exception as e:
        return {
            "response": "I'm sorry, I encountered an error while processing your question.",
            "additional_info": [
                "This may be due to a temporary issue with our service.",
                "Please try again later or rephrase your question."
            ],
            "error": str(e)
        }

# Generate more detailed condition advice report
@app.post("/detailed-advice")
async def detailed_advice(condition: str = Form(...)):
    """
    Generate detailed advice for specific skin conditions
    
    Parameters:
    - condition: The skin condition code (e.g., "mel", "bcc")
    """
    # Check if condition is valid
    if condition not in chatbot_responses:
        return {
            "error": "Unknown condition code. Please provide a valid condition."
        }
    
    # Get condition name
    condition_name = next((desc for code, (abbr, desc) in classes.items() if abbr == condition), "Unknown")
    
    # Use Together API for generating detailed advice
    prompt = f"""As an expert dermatologist, create a concise but comprehensive report about {condition_name}.
Structure your response with these sections:
1. What it is: Brief explanation in simple terms
2. Risk factors: Key risk factors in a few points
3. Management: Brief overview of treatment approaches
4. When to see a doctor: Clear indicators for seeking medical attention

Keep each section to 1-2 sentences maximum. Be factually accurate and educational.
"""
    
    try:
        # Generate the detailed response
        response_text = generate_together_response(prompt, temperature=0.7, max_tokens=1000)
        
        # Parse the response into sections
        sections = {}
        current_section = None
        
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if "what it is" in lower_line:
                current_section = "what_it_is"
                sections[current_section] = ""
            elif "risk factor" in lower_line:
                current_section = "risk_factors"
                sections[current_section] = ""
            elif "management" in lower_line:
                current_section = "management"
                sections[current_section] = ""
            elif "when to see" in lower_line or "doctor" in lower_line and "when" in lower_line:
                current_section = "when_to_see_doctor"
                sections[current_section] = ""
            elif current_section:
                # Add content to current section, removing any numbering or bullets
                clean_line = line.lstrip('- •0123456789. ')
                if sections[current_section]:
                    sections[current_section] += " " + clean_line
                else:
                    sections[current_section] = clean_line
        
        # Prepare the response
        response = {
            "condition_name": condition_name,
            "condition_code": condition,
            "what_it_is": sections.get("what_it_is", "Information not available"),
            "risk_factors": sections.get("risk_factors", "Information not available"),
            "management": sections.get("management", "Information not available"),
            "when_to_see_doctor": sections.get("when_to_see_doctor", "When in doubt, consult a dermatologist.")
        }
        
        return response
        
    except Exception as e:
        # Fallback to static responses if API fails
        response = {
            "condition_name": condition_name,
            "condition_code": condition,
            "what_it_is": "Information not available due to service error.",
            "risk_factors": "Please try again later.",
            "management": "In the meantime, consult a healthcare professional for advice.",
            "when_to_see_doctor": "If you're concerned about your skin, see a dermatologist.",
            "error": str(e)
        }
        
        return response

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)