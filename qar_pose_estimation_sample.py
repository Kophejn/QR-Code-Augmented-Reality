"""Sample usage of QR code augmented reality library v1.0 with drawAxes() visualization.

Press 'ESC' or 'q' to close window/end script or close window with mouse cursor.

QR Code is registered trademark of DENSO WAVE INCORPORATED.

Created by DB 2020. Based on OpenCV 4.2.
"""

import cv2
import numpy
from threading import Thread
import sys

import qr_code_augmented_reality as qar


# WebCamVideoStream class taken from https://github.com/rktayal/multithreaded_frame_reading.
class WebCamVideoStream:
    # Added resolution to inputs.
    def __init__(self, src=0, res_width=1280, res_height=720):
        # Initialize the video camera stream and read the first frame from the stream.
        self.stream = cv2.VideoCapture(src)
        # Set output resolution.
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
        # Set camera focus to infinity.
        self.stream.set(cv2.CAP_PROP_FOCUS, 0)
        # Turn off autofocus.
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Force mjpg output from webcam.
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.stream.set(cv2.CAP_PROP_FOURCC, fourcc)

        (self.grabbed, self.frame) = self.stream.read()

        # Initialize the variable used to indicate if the thread should be stopped.
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream.
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped.
        while True:
            # If the thread indicator variable is set, stop the thread.
            if self.stopped:
                return
            # Otherwise read the next frame from the stream.
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read.
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped.
        self.stopped = True


# Load camera matrix from its calibration.
cam_mtx = numpy.load('camera_matrix.npy')
# Load distortion coefficients from camera calibration.
distortion = numpy.load('distortion_coefficients.npy')

# Camera index: 0 - internal; 1 - external; etc.
CAM_INDEX = 1
# Camera resolution.
RES_HEIGHT = 720
RES_WIDTH = 1280

# Camera feed initialization.
stream = WebCamVideoStream(src=CAM_INDEX, res_width=RES_WIDTH, res_height=RES_HEIGHT).start()
# "frame" window initialization.
cv2.namedWindow("Frame")

# Tracker and memory init.
ct, mem = qar.init()
# Creating of DetectorParameters is optional, if not provided, default parameters will be used.
params = qar.DetectorParameters()

while True:
    # Load frame from webcam feed.
    frame = stream.read()
    # Copy frame for further use with drawAxes()
    frame_copy = frame.copy()
    # Detect and try to decode QR Codes.
    qr_codes = qar.detectCodes(frame, ct, mem, params)
    # Note: not_legit flag is set True for estimating pose for codes with no size information encoded in their content.
    qar.estimatePose(qr_codes, cam_mtx, distortion, not_legit=True)
    # Draw functions for demonstration.
    frame_copy = qar.drawAxes(frame_copy, qr_codes, cam_mtx, distortion)

    # Check for key input.
    if (cv2.waitKey(1) == ord("q")) or (cv2.waitKey(1) == 27):
        break
    # Check if window is closed.
    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Show frame
    cv2.imshow("Frame", frame_copy)

stream.stop()
cv2.destroyAllWindows()
sys.exit()
