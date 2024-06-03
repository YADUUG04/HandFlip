import cv2
import mediapipe as mp
import time
import math
import csv
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Define a function to calculate the angle between two points
def calculate_angle(point1, point2):
    return math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

# Define Streamlit app
def main():
    st.title("Hand Flip Detection")

    # Get user input for video file upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture("uploaded_video.mp4")  # Use the file path as input to VideoCapture

        prev_hand_direction = None
        prev_time = time.time()
        flip_detected = False
        flip_count = 0
        flip_times = []  # List to store the timestamps of each flip

        # Create a placeholder for the image
        img_placeholder = st.empty()

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                st.warning("Finished processing the video or encountered an error.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            lmList = []
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmark.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                    if len(lmList) >= 2:
                        thumb_tip = lmList[4][1], lmList[4][2]
                        index_tip = lmList[8][1], lmList[8][2]

                        # Calculate the direction vector
                        direction_vector = (index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

                        # Calculate the angle of the direction vector
                        angle = calculate_angle(thumb_tip, index_tip)

                        # Detect hand flip based on angle change
                        if prev_hand_direction is not None:
                            angle_change = abs(angle - prev_hand_direction)
                            if angle_change > 90:  # You may need to adjust this threshold
                                flip_detected = not flip_detected
                                if flip_detected:
                                    flip_count += 1
                                    curr_time = time.time()
                                    hand_speed = abs(angle_change / (curr_time - prev_time))
                                    flip_times.append(curr_time)  # Record the timestamp of the flip
                                    st.write(f"Hand flip detected! Total flips: {flip_count}, Hand speed: {hand_speed:.2f} degrees/sec")
                                    prev_time = curr_time

                        prev_hand_direction = angle

                        # Draw landmarks on the image
                        mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

            img_placeholder.image(img, channels="BGR")

        cap.release()

        # Calculate the time between consecutive flips
        time_between_flips = [flip_times[i] - flip_times[i-1] for i in range(1, len(flip_times))]
        st.write("Time between flips:", time_between_flips)

        # Calculate and display the average time between flips
        if time_between_flips:
            average_time_between_flips = sum(time_between_flips) / len(time_between_flips)
            st.write(f"Average Time Between Flips: {average_time_between_flips:.2f} seconds")
        else:
            average_time_between_flips = None
            st.write("No flips detected, so average time cannot be calculated.")

        # Generate PDF report
        generate_pdf_report(flip_count, time_between_flips, average_time_between_flips)

def generate_pdf_report(flip_count, time_between_flips, average_time_between_flips):
    pdf_file_path = 'hand_flip_report.pdf'
    c = canvas.Canvas(pdf_file_path, pagesize=letter)

    # Title and summary
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "Hand Flip Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Total Hand Flips: {flip_count}")
    c.drawString(50, 680, f"Average Time Between Flips: {average_time_between_flips:.2f} seconds")

    # Detailed data
    c.drawString(50, 640, "Detailed Data:")
    y = 620
    for i, time_between_flip in enumerate(time_between_flips, start=1):
        c.drawString(50, y, f"Flip {i}: Time Between Flips = {time_between_flip:.2f} seconds")
        y -= 20

    c.save()

    # Add a download button for the PDF report
    st.download_button(
        label="Download PDF Report",
        data=open(pdf_file_path, 'rb').read(),
        file_name="hand_flip_report.pdf",
        mime="application/pdf"
    )

# Run the Streamlit app
if __name__ == "__main__":
    main()
