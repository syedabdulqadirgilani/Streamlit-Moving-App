import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analyze_graph_trend(image_path_or_bytes):
    """
    Analyzes a graph image to determine if the last part is trending up or down.
    Returns the analyzed image and the trend direction string.
    """
    # Load image
    if isinstance(image_path_or_bytes, str):
        img = cv2.imread(image_path_or_bytes)
    else: # Assuming bytes from Streamlit uploader
        file_bytes = np.asarray(bytearray(image_path_or_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

    if img is None:
        return None, "Error: Could not load image."

    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image (isolate the graph line)
    # Adjust these values based on typical graph line colors and background
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) 

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is our main graph line
    if not contours:
        return original_img, "No clear graph line detected."
        
    main_contour = max(contours, key=cv2.contourArea)

    # Get points of the contour
    points = [p[0] for p in main_contour]
    
    # Sort points by X-coordinate to represent left-to-right progression
    points.sort(key=lambda p: p[0])

    if len(points) < 2:
        return original_img, "Not enough points to determine trend."

    # Focus on the last N% of the graph for trend analysis
    # You can adjust this percentage (e.g., 0.1 for 10%, 0.2 for 20%)
    percentage_for_trend = 0.15 # Analyze the last 15% of the graph
    start_index = int(len(points) * (1 - percentage_for_trend))
    
    if start_index >= len(points) - 1: # Ensure there are at least two points for a segment
        start_index = len(points) - 2
        if start_index < 0:
            return original_img, "Not enough data points in the last segment."

    trend_points = points[start_index:]

    # Find the leftmost and rightmost points in the trend_points segment
    x_min_trend, y_at_x_min_trend = trend_points[0][0], trend_points[0][1]
    x_max_trend, y_at_x_max_trend = trend_points[-1][0], trend_points[-1][1]

    # Calculate slope (y-axis is inverted in images: lower y-value is higher on screen)
    # A negative change in y_coordinate (y_at_x_max_trend - y_at_x_min_trend) means the line went up.
    # A positive change in y_coordinate means the line went down.
    
    y_diff = y_at_x_max_trend - y_at_x_min_trend
    x_diff = x_max_trend - x_min_trend

    trend = "Undetermined"
    if x_diff > 0: # Ensure there's horizontal movement
        if y_diff < -5: # Graph went up (y-coordinate decreased)
            trend = "Upward Trend"
        elif y_diff > 5: # Graph went down (y-coordinate increased)
            trend = "Downward Trend"
        else:
            trend = "Relatively Flat"
    else:
        trend = "Insufficient horizontal movement in last segment."


    # Draw the trend line and arrow on the original image
    if x_diff > 0: # Only draw if there's horizontal movement for trend
        # Define line start and end for visualization based on the trend segment
        start_point_viz = (x_min_trend, y_at_x_min_trend)
        end_point_viz = (x_max_trend, y_at_x_max_trend)

        color = (0, 255, 0) if "Upward" in trend else ((0, 0, 255) if "Downward" in trend else (255, 255, 0)) # Green for up, Red for down, Yellow for flat
        cv2.arrowedLine(original_img, start_point_viz, end_point_viz, color, 3, tipLength=0.5)

    return original_img, trend


# Streamlit app layout
st.title("Graph Trend Analyzer")
st.write("Upload a screenshot of a graph to determine if its last part is trending upward or downward.")

uploaded_file = st.file_uploader("Choose a graph image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Graph", use_column_width=True)
    st.write("")
    st.write("Analyzing...")

    analyzed_img, trend_direction = analyze_graph_trend(uploaded_file)
    
    if analyzed_img is not None:
        st.subheader("Analysis Result:")
        st.write(f"The last part of the graph shows an: **{trend_direction}**")
        
        # Convert OpenCV image (BGR) to RGB for Streamlit display
        analyzed_img_rgb = cv2.cvtColor(analyzed_img, cv2.COLOR_BGR2RGB)
        st.image(analyzed_img_rgb, caption="Analyzed Graph with Trend Indicator", use_column_width=True)
    else:
        st.error(trend_direction)
