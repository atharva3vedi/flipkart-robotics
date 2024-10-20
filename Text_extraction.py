import cv2
import base64
from groq import Groq
import os

def resize_image(image_path, max_size=800, max_file_size=4 * 1024 * 1024):
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image to reduce size
    height, width = img.shape[:2]
    scaling_factor = max_size / float(max(height, width))
    
    # Perform resizing
    resized_img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Save the resized image
    resized_image_path = "resized_image.jpg"
    cv2.imwrite(resized_image_path, resized_img)

    # Check file size and resize further if needed
    while os.path.getsize(resized_image_path) > max_file_size:
        scaling_factor *= 0.9  # Reduce size by 10%
        resized_img = cv2.resize(resized_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        cv2.imwrite(resized_image_path, resized_img)

    return resized_image_path

# Function to preprocess the image for OCR, ensuring file size is <4MB
def preprocess_image(image_path, max_file_size=4 * 1024 * 1024):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply contrast enhancement for better text visibility
    contrast_enhanced = cv2.equalizeHist(blurred)

    # Apply edge detection to sharpen the image and enhance clarity
    edges = cv2.Canny(contrast_enhanced, 50, 150)

    # Combine edges and the contrast-enhanced image
    combined = cv2.bitwise_or(contrast_enhanced, edges)

    # Apply adaptive thresholding to highlight text
    thresh = cv2.adaptiveThreshold(
        combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save the processed image temporarily
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, thresh)

    # Check the file size, and if it exceeds 4MB, resize the image
    if os.path.getsize(processed_image_path) > max_file_size:
        processed_image_path = resize_image(processed_image_path)

    return processed_image_path

# Function to encode the image in base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your original image
original_image_path = "rottenapple.jpg"

# Initialize the Groq client
client = Groq(api_key='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')

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

# Analyze the content of the response to check for fruits or vegetables
contains_fruits_or_vegetables = "fruits" in initial_analysis.choices[0].message.content.lower() or "vegetables" in initial_analysis.choices[0].message.content.lower()


# If the image contains fruits or vegetables, only resize it
if not contains_fruits_or_vegetables:
    processed_image_path = resize_image(original_image_path)
else:
    # If not, apply OCR-specific preprocessing
    processed_image_path = preprocess_image(original_image_path)

# Get the base64 string of the processed image
base64_image = encode_image(processed_image_path)

# Create a chat completion request with the processed image
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Also tell me the exact expiration date/use by date of the product if it's there if its not dont mention it. How fresh the produce is if it is a fruit or vegetable if it is not a fruit or vegetable dont say anything about it being a fruit or vegetable and how long do you think it will remain edible? Automatically assess the quality of fruits and vegetables by detecting defects, discoloration, or irregular shapes and be harsh in your assesment. Go into as much detail as you can and be very thorough and present your findings in digestable points."},
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

# Print the response from Groq's vision model
print(chat_completion.choices[0].message.content)
