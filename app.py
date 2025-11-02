# Import required libraries
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set up the Streamlit page with a title
st.title("Object Movement Direction Detector")
st.write("Upload an image to detect if an object is moving up or down!")

# Create file uploader widget
uploaded_file = st.file_uploader("Choose an image.", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read and convert the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Split image into top and bottom halves
    top_half = blurred[:height//2, :]
    bottom_half = blurred[height//2:, :]
    
    # Calculate average intensity for each half
    top_avg = np.mean(top_half)
    bottom_avg = np.mean(bottom_half)
    
    # Display the original image
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Determine movement direction based on intensity difference
    if abs(top_avg - bottom_avg) > 5:  # threshold for significant difference
        if top_avg > bottom_avg:
            st.success("The object appears to be moving UP!")
            st.write("The top half is brighter, suggesting upward movement.")
        else:
            st.success("The object appears to be moving DOWN!")
            st.write("The bottom half is brighter, suggesting downward movement.")
    else:
        st.info("Cannot determine clear movement direction.")
        st.write("The brightness difference is not significant enough.")
    
    # Display technical details if user wants to see them
    with st.expander("Show Technical Details"):
        st.write(f"Top half average brightness: {top_avg:.2f}")
        st.write(f"Bottom half average brightness: {bottom_avg:.2f}")
        st.write(f"Brightness difference: {abs(top_avg - bottom_avg):.2f}")
else:
    # Show instructions when no file is uploaded
    st.info("Please upload an image to get started!")