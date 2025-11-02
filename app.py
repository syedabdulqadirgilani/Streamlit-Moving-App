import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analyze_graph_trend_enhanced(image_bytes):
    """
    Analyzes a graph image to determine if the last part is trending up or down.
    This version includes more robust line detection.
    Returns the analyzed image and the trend direction string.
    """
    # Load image from bytes
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    if img is None:
        return None, "Error: Could not load image."

    original_img = img.copy()
    h, w, _ = img.shape
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try multiple methods to find the graph line
    
    # Method 1: Canny Edge Detection (good for distinct lines)
    edges = cv2.Canny(gray, 50, 150)
    
    # Method 2: Adaptive Thresholding (good for lines on varied backgrounds)
    # This works well if the line is significantly darker or lighter than its immediate surroundings
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    
    # Method 3: Color-based segmentation (if graph line has a distinct color, e.g., green on black/white)
    # Assuming common graph line colors like green, blue, red.
    # We'll check for green lines, as your example showed a green line.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a range for green color in HSV (adjust these values if your graph lines are different colors)
    # Lower bound for green
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks: Prioritize color mask if it finds something, otherwise use edges or adaptive
    combined_mask = np.zeros_like(gray)
    if np.sum(mask_green) > 100: # If a significant green line is found
        combined_mask = mask_green
    else:
        # Combine Canny edges and adaptive threshold if color detection is not strong
        combined_mask = cv2.addWeighted(edges, 0.5, adaptive_thresh, 0.5, 0)
    
    # Dilate to connect broken line segments
    kernel = np.ones((3,3),np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Find contours on the processed mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return original_img, "No significant graph line detected after processing."

    # Filter contours to find the most likely graph line
    # Criteria: Area, width, position (often in the middle part of the image)
    main_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50: # Minimum area to consider a contour as part of a line
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            # Filter based on common graph characteristics:
            # - Should be reasonably wide (not just a dot)
            # - Should not be too tall (not a vertical axis label)
            # - Should not span the entire image width/height unless it's the actual line
            if w_cont > 0.1 * w and h_cont < 0.8 * h and area > max_area:
                main_contour = contour
                max_area = area

    if main_contour is None:
        return original_img, "No suitable graph line contour found."

    # Get points from the main contour
    # Simplify the contour to reduce noise points
    epsilon = 0.001 * cv2.arcLength(main_contour, True)
    approx_points = cv2.approxPolyDP(main_contour, epsilon, True) # Use True for closed contour if needed, but False usually works better for lines
    
    points = [p[0] for p in approx_points] # Extract (x,y) tuples
    
    # Sort points by X-coordinate to ensure left-to-right progression
    points.sort(key=lambda p: p[0])

    if len(points) < 2:
        return original_img, "Not enough distinct points on the graph line to determine trend."

    # Focus on the last N% of the graph for trend analysis
    percentage_for_trend = 0.25 # Analyze the last 25% of the graph for better stability
    start_index = int(len(points) * (1 - percentage_for_trend))
    
    # Ensure start_index is valid and leaves at least 2 points for calculation
    if start_index >= len(points) - 1:
        start_index = max(0, len(points) - 2) # Ensure at least two points for a segment

    trend_points = points[start_index:]

    if len(trend_points) < 2:
        return original_img, "Not enough data points in the last segment for trend analysis."

    # Find the leftmost and rightmost points in the trend_points segment
    x_start_trend, y_start_trend = trend_points[0][0], trend_points[0][1]
    x_end_trend, y_end_trend = trend_points[-1][0], trend_points[-1][1]

    # Calculate slope
    # Y-axis is inverted in images: lower y-value is higher on screen.
    # A negative change in y_coordinate (y_end_trend - y_start_trend) means the line went up.
    # A positive change in y_coordinate means the line went down.
    
    y_diff = y_end_trend - y_start_trend
    x_diff = x_end_trend - x_start_trend

    trend = "Undetermined"
    # A threshold for "flatness" relative to image height
    # Adjust this value (e.g., 0.02 for 2% of image height) for sensitivity to vertical change
    vertical_change_threshold = 0.02 * h 

    if x_diff > 10: # Ensure significant horizontal movement in the segment (at least 10 pixels wide)
        if y_diff < -vertical_change_threshold: # Graph went up (y-coordinate decreased significantly)
            trend = "Upward Trend"
        elif y_diff > vertical_change_threshold: # Graph went down (y-coordinate increased significantly)
            trend = "Downward Trend"
        else:
            trend = "Relatively Flat"
    elif abs(y_diff) > vertical_change_threshold: # If little horizontal movement but significant vertical
        if y_diff < 0:
            trend = "Upward Trend (steep vertical)"
        else:
            trend = "Downward Trend (steep vertical)"
    else:
        trend = "Relatively Flat (insufficient movement)"


    # Draw the trend line and arrow on the original image
    trend_start_point_viz = (x_start_trend, y_start_trend)
    trend_end_point_viz = (x_end_trend, y_end_trend)

    color = (0, 255, 0) # Green for up
    if "Downward" in trend:
        color = (0, 0, 255) # Red for down
    elif "Flat" in trend or "Undetermined" in trend:
        color = (255, 255, 0) # Yellow for flat/undetermined

    cv2.arrowedLine(original_img, trend_start_point_viz, trend_end_point_viz, color, 3, tipLength=0.5)

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

    analyzed_img_bgr, trend_direction = analyze_graph_trend_enhanced(uploaded_file)
    
    if analyzed_img_bgr is not None:
        st.subheader("Analysis Result:")
        # Display trend direction with color coding
        display_color = 'green'
        if 'Downward' in trend_direction:
            display_color = 'red'
        elif 'Flat' in trend_direction or 'Undetermined' in trend_direction:
            display_color = 'orange'
        
        st.markdown(f"The last part of the graph shows an: **<span style='color: {display_color}; font-size: 20px;'>{trend_direction}</span>**", unsafe_allow_html=True)
        
        # Convert OpenCV image (BGR) to RGB for Streamlit display
        analyzed_img_rgb = cv2.cvtColor(analyzed_img_bgr, cv2.COLOR_BGR2RGB)
        st.image(analyzed_img_rgb, caption="Analyzed Graph with Trend Indicator", use_column_width=True)
    else:
        st.error(trend_direction)


