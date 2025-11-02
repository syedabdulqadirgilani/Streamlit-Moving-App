import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analyze_graph_trend_robust(image_bytes):
    """
    Analyzes a graph image to determine if the last part is trending up or down.
    Returns the analyzed image and the trend direction string.
    """
    # Load image from bytes
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        return None, "Error: Could not load image."

    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection or adaptive thresholding for better line isolation
    # Edge detection is often better for finding lines of varying intensity
    edges = cv2.Canny(blurred, 50, 150) # You might need to adjust these thresholds

    # Dilate edges to make the line thicker and more continuous
    kernel = np.ones((2,2),np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find potential graph lines (e.g., based on area or aspect ratio)
    potential_lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100: # Filter out very small noise contours, adjust as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Consider contours that are wide relative to their height (like a horizontal line)
            # or have a significant number of points
            if w > h * 0.5 and len(contour) > 20: 
                potential_lines.append(contour)

    if not potential_lines:
        return original_img, "No significant graph line detected."
        
    # Combine potential lines into one for robust analysis, or pick the largest
    # For a single main graph, the largest might be sufficient
    main_contour = max(potential_lines, key=cv2.contourArea)

    # Get points of the contour
    # Approximate the contour to simplify it and remove noise
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    approx_points = cv2.approxPolyDP(main_contour, epsilon, False)
    
    points = [p[0] for p in approx_points] # Extract (x,y) tuples
    
    # Sort points by X-coordinate to ensure left-to-right progression
    points.sort(key=lambda p: p[0])

    if len(points) < 2:
        return original_img, "Not enough distinct points on the graph line to determine trend."

    # Focus on the last N% of the graph for trend analysis
    percentage_for_trend = 0.20 # Analyze the last 20% of the graph for better stability
    start_index = int(len(points) * (1 - percentage_for_trend))
    
    # Ensure start_index is valid and leaves at least 2 points for calculation
    if start_index >= len(points) - 1:
        start_index = max(0, len(points) - 2) # Ensure at least two points
    
    trend_points = points[start_index:]

    if len(trend_points) < 2:
        return original_img, "Not enough data points in the last segment for trend analysis."

    # Find the leftmost and rightmost points in the trend_points segment
    x_min_trend, y_at_x_min_trend = trend_points[0][0], trend_points[0][1]
    x_max_trend, y_at_x_max_trend = trend_points[-1][0], trend_points[-1][1]

    # Calculate slope
    # Remember y-axis is inverted in images: lower y-value is higher on screen.
    # A negative change in y_coordinate (y_at_x_max_trend - y_at_x_min_trend) means the line went up.
    # A positive change in y_coordinate means the line went down.
    
    y_diff = y_at_x_max_trend - y_at_x_min_trend
    x_diff = x_max_trend - x_min_trend

    trend = "Undetermined"
    # A threshold for "flatness" to avoid minor fluctuations being called up/down
    slope_threshold = 0.05 * img.shape[0] # 5% of image height as a significance threshold

    if x_diff > 10: # Ensure significant horizontal movement in the segment (at least 10 pixels wide)
        if y_diff < -slope_threshold: # Graph went up (y-coordinate decreased significantly)
            trend = "Upward Trend"
        elif y_diff > slope_threshold: # Graph went down (y-coordinate increased significantly)
            trend = "Downward Trend"
        else:
            trend = "Relatively Flat"
    elif abs(y_diff) > slope_threshold: # If little horizontal movement but significant vertical
        if y_diff < 0:
            trend = "Upward Trend (steep vertical)"
        else:
            trend = "Downward Trend (steep vertical)"
    else:
        trend = "Relatively Flat (insufficient movement)"


    # Draw the trend line and arrow on the original image
    # Use the most representative start and end points of the detected trend segment
    trend_start_point = (x_min_trend, y_at_x_min_trend)
    trend_end_point = (x_max_trend, y_at_x_max_trend)

    color = (0, 255, 0) # Green for up
    if "Downward" in trend:
        color = (0, 0, 255) # Red for down
    elif "Flat" in trend or "Undetermined" in trend:
        color = (255, 255, 0) # Yellow for flat/undetermined

    cv2.arrowedLine(original_img, trend_start_point, trend_end_point, color, 3, tipLength=0.5)

    return original_img, trend

# Streamlit app layout
st.set_page_config(page_title="Graph Trend Analyzer", layout="centered")
st.title("Graph Trend Analyzer")
st.write("Upload a screenshot of a graph to determine if its last part is trending upward or downward.")
st.markdown("---")

uploaded_file = st.file_uploader("Choose a graph image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Graph", use_column_width=True)
    st.write("")
    st.info("Analyzing... Please wait.")

    analyzed_img_bgr, trend_direction = analyze_graph_trend_robust(uploaded_file)
    
    if analyzed_img_bgr is not None:
        st.subheader("Analysis Result:")
        st.markdown(f"The last part of the graph shows an: **<span style='color: {'green' if 'Upward' in trend_direction else ('red' if 'Downward' in trend_direction else 'orange')}; font-size: 20px;'>{trend_direction}</span>**", unsafe_allow_html=True)
        
        # Convert OpenCV image (BGR) to RGB for Streamlit display
        analyzed_img_rgb = cv2.cvtColor(analyzed_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(analyzed_img_rgb, caption="Analyzed Graph with Trend Indicator", use_column_width=True)
    else:
        st.error(trend_direction)

