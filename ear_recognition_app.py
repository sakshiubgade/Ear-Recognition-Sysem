# Import libraries
import streamlit as st
from PIL import Image
import re  # For regular expressions
import tensorflow as tf  # For model loading
import numpy as np  # For array operations

# Path to your processed images folder (modify if needed)
data_dir = "C:/Users/PRANAY/OneDrive/Desktop/Ear Recoginition System_Mini Project_6sem/ear/raw"

# Function to extract label (user ID) from filename
def get_label(filename):
  """Extracts the label (user ID) using regular expressions."""
  match = re.search(r"(\d+)_", filename)  # Match digits followed by an underscore
  if match:
    return match.group(1)
  else:
    return None

# Title and description
st.title("Ear Recognition Model")
st.write("Upload an image for prediction. (Supports common image formats)")

# Image upload and prediction
uploaded_file = st.file_uploader("Choose an image:")
if uploaded_file is not None:
  # Define allowed extensions within the code block where it's used
  allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
  filename = uploaded_file.name
  if not any(filename.endswith(ext) for ext in allowed_extensions):
    st.error("Unsupported file format. Please upload a JPG, JPEG, PNG, or BMP image.")
  else:
    # Load the image using PIL
    try:
      img = Image.open(uploaded_file)
      img = img.convert('RGB')  # Convert to RGB if grayscale
      img = img.resize((224, 224))  # Resize to match model input size (modify if needed)
      img_array = np.array(img)
      img_array = img_array / 255.0  # Normalize pixel values

      # Load your trained model
      model_path = "C:/Users/PRANAY/OneDrive/Desktop/Ear Recoginition System_Mini Project_6sem/94.61_ear_recognition_model.h5"
      model = tf.keras.models.load_model(model_path)

      # Prediction logic using your model
      img_array = np.expand_dims(img_array, axis=0)  # Add a new dimension for batch size

      prediction = model.predict(img_array)[0]
      predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability

      # Get label from filename
      label = get_label(filename)

      # Display results
      if label:
        st.write(f"Predicted Label (User ID): {label}")
      else:
        st.warning("Could not extract label and features from image. The image is not an ear image !!!!!")
      st.image(img, caption="Uploaded Image")
      #st.write(f"Predicted Class: {predicted_class}")  # Display predicted class

    except Exception as e:
      st.error(f"An error occurred while processing the image: {e}")
