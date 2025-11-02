import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page
st.title("Object Movement Direction Detector")
st.write("Upload one image (brightness heuristic) or two images (optical flow) to detect up/down movement.")

# Sidebar controls
st.sidebar.header("Detection settings")
brightness_threshold = st.sidebar.slider("Brightness difference threshold", 0.0, 50.0, 5.0, step=0.5)
flow_threshold = st.sidebar.slider("Optical flow mean-y threshold", 0.0, 20.0, 1.0, step=0.1)
show_debug = st.sidebar.checkbox("Show debug images (halves / flow values)", value=True)

# File uploader: allow multiple files (1 or 2 images)
uploaded_files = st.file_uploader("Choose image(s)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

def pil_to_rgb_array(pil_img):
    img = pil_img.convert('RGB')
    return np.array(img)

def resize_to(img, shape):
    return cv2.resize(img, (shape[1], shape[0]))

if uploaded_files and len(uploaded_files) >= 1:
    try:
        # If one image uploaded -> brightness heuristic
        if len(uploaded_files) == 1:
            image = Image.open(uploaded_files[0])
            image_np = pil_to_rgb_array(image)

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            h, w = blurred.shape
            top_half = blurred[:h//2, :]
            bottom_half = blurred[h//2:, :]

            top_avg = float(np.mean(top_half))
            bottom_avg = float(np.mean(bottom_half))
            diff = abs(top_avg - bottom_avg)

            st.image(image_np, caption='Uploaded Image', use_container_width=True)

            if diff > brightness_threshold:
                if top_avg > bottom_avg:
                    st.success("The object appears to be moving UP (brightness heuristic).")
                    st.write("Top half is brighter than bottom half.")
                else:
                    st.success("The object appears to be moving DOWN (brightness heuristic).")
                    st.write("Bottom half is brighter than top half.")
            else:
                st.info("Cannot determine clear movement direction from brightness.")
                st.write("Brightness difference not significant for the chosen threshold.")

            with st.expander("Show Technical Details"):
                st.write(f"Top half avg: {top_avg:.2f}")
                st.write(f"Bottom half avg: {bottom_avg:.2f}")
                st.write(f"Brightness diff: {diff:.2f}")
                if show_debug:
                    st.image(np.stack([top_half, bottom_half], axis=2), caption='Top (L) and Bottom (R) halves (as grayscale stacked)', use_column_width=True)

        else:
            # Two images uploaded -> use optical flow
            img1 = Image.open(uploaded_files[0])
            img2 = Image.open(uploaded_files[1])
            img1_np = pil_to_rgb_array(img1)
            img2_np = pil_to_rgb_array(img2)

            # Resize second to first if needed
            if img1_np.shape[:2] != img2_np.shape[:2]:
                img2_np = resize_to(img2_np, img1_np.shape[:2])

            prev = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)

            # Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(prev, next, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            mean_flow_y = float(np.mean(flow[..., 1]))
            mean_flow_x = float(np.mean(flow[..., 0]))

            st.image([img1_np, img2_np], caption=['Frame 1', 'Frame 2'], width=300)

            # In OpenCV coordinates positive y means movement downward
            if abs(mean_flow_y) > flow_threshold:
                if mean_flow_y < 0:
                    st.success("Detected movement UP (optical flow).")
                    st.write("Average vertical flow is negative (upwards).")
                else:
                    st.success("Detected movement DOWN (optical flow).")
                    st.write("Average vertical flow is positive (downwards).")
            else:
                st.info("Cannot determine clear movement direction from optical flow.")
                st.write("Average vertical flow magnitude below threshold.")

            with st.expander("Show Optical Flow Details"):
                st.write(f"Mean flow Y: {mean_flow_y:.3f}")
                st.write(f"Mean flow X: {mean_flow_x:.3f}")

                if show_debug:
                    # visualise flow magnitude (grayscale)
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    st.image(mag_norm, caption='Optical flow magnitude (normalized)', use_column_width=True)

