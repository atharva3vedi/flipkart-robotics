# This script processes an image for OCR and quality analysis, resizes it if necessary, and sends it for vision-based assessment using Groq's API.

import cv2
import base64
from groq import Groq
import os

# Function to resize the image and ensure its file size is within the given limits
def resize_image(image_path, max_size=800, max_file_size=4 * 1024 * 1024):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    scaling_factor = max_size / float(max(height, width))
    resized_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    resized_image_path = "resized_image.jpg"
    cv2.imwrite(resized_image_path, resized_img)
    
    while os.path.getsize(resized_image_path) > max_file_size:
        scaling_factor *= 0.9
        resized_img = cv2.resize(resized_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        cv2.imwrite(resized_image_path, resized_img)

    return resized_image_path

# Function to preprocess the image for OCR, enhancing contrast and applying thresholding
def preprocess_image(image_path, max_file_size=4 * 1024 * 1024):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    contrast_enhanced = cv2.equalizeHist(blurred)
    edges = cv2.Canny(contrast_enhanced, 50, 150)
    combined = cv2.bitwise_or(contrast_enhanced, edges)
    thresh = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, thresh)
    
    if os.path.getsize(processed_image_path) > max_file_size:
        processed_image_path = resize_image(processed_image_path)

    return processed_image_path

# Function to encode the image in base64 format for transmission
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

original_image_path = ''
groq_key=os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_key)

# Initial analysis to detect if the image contains fruits or vegetables
initial_analysis = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Does this image contain fruits or vegetables?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(original_image_path)}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

contains_fruits_or_vegetables = "fruits" in initial_analysis.choices[0].message.content.lower() or "vegetables" in initial_analysis.choices[0].message.content.lower()

# Based on the initial analysis, either resize or preprocess the image
if not contains_fruits_or_vegetables:
    processed_image_path = resize_image(original_image_path)
else:
    processed_image_path = preprocess_image(original_image_path)

# Get the base64 string of the processed image
base64_image = encode_image(processed_image_path)

# Create a chat completion request with the processed image and specific queries about its content
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the image and extract detailed information from the product packaging, including the expiration date or use-by date if available. If no such date is present, do not mention it. Assess the freshness of the item if it is a fruit or vegetable; if it is not, refrain from discussing its freshness. For fruits and vegetables, provide an estimation of how long they will remain edible, considering factors such as quality, defects, discoloration, or irregular shapes. Be rigorous in your evaluation and present your findings in clear, concise points, focusing on packaging details as well as any visible product quality issues."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-90b-vision-preview",
)

print(chat_completion.choices[0].message.content)
