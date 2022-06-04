"""Demo application with use of QR code augmented reality library v1.0.

SERVER MUST BE STARTED FIRST.

AR highlight of predefined PCB parts and links to its datasheets.
Link to .txt file on local server is encoded in QR Code on the PCB.
Instructions for visualization and other information about each part are in the .txt file.

Press 'ESC' or 'q' to close window/end script or close window with mouse cursor.

QR Code is registered trademark of DENSO WAVE INCORPORATED.

Created by DB 2020. Based on OpenCV 4.2.
"""

import cv2
import numpy
import sys
import webbrowser
from threading import Thread
from urllib.request import urlopen
from typing import List, Tuple

import qr_code_augmented_reality as qar


# WebCamVideoStream class originates from https://github.com/rktayal/multithreaded_frame_reading.
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


class Part:
    """Part object contains all information about part.

        Instance attributes:
            name: str
                Name of the Part.
            specs: str
                Specs of the Part. Multiple lines separated by '\n' newline character.
            contour: numpy.ndarray
                Contour associated to the Part
            link: str
                Link associated to the Part
            projected_contour: None or numpy.ndarray
                Projected contour after use of draw method.
    """
    
    def __init__(self, name: str, specs: str, contour: numpy.ndarray, link: str):
        """
        :param name: name of the part
        :param specs: spec of the part
        :param contour: contour associated to part
        :param link: web link associated to part starting with "http://" or "https://"
        """

        self.name = name
        self.specs = specs
        self.contour = contour
        self.link = link
        self.projected_contour = None

    def project(self, rotation_vector: numpy.ndarray, translation_vector: numpy.ndarray, camera_matrix: numpy.ndarray,
                distortion_coefficients: numpy.ndarray) -> None:
        """Project contour points in QR Code coordinate space to camera (frame) coordinate space.

        :param rotation_vector: QR Code rotation vector from QR Code pose estimation
        :param translation_vector: QR Code translation vector from QR Code pose estimation
        :param camera_matrix: camera matrix of used camera obtained from camera calibration process
        :param distortion_coefficients: distortion coefficients obtained from camera calibration process
        :return: None
        """

        # Reshape to ensure proper form.
        contour = self.contour.reshape(-1, 3)
        # Project contour to camera coordinate space.
        projected_contour, _ = cv2.projectPoints(contour, rotation_vector, translation_vector,
                                                 camera_matrix, distortion_coefficients)
        # Reshape to proper form cv2 contour form and save as Part object attribute.
        self.projected_contour = numpy.int32(projected_contour).reshape(-1, 2)

    def draw(self, img: numpy.ndarray, color: Tuple[int]) -> None:
        """Draws projected contour to given image with given colour.

        :param img: img as numpy ndarray (BGR colorspace)
        :param color: color in BGR order as tuple
        :return: None
        """

        if self.projected_contour is not None:
            cv2.drawContours(img, [self.projected_contour], -1, color, 2)


class PartsOrigin:
    """PartsOrigin object contains qar.Code object which is the origin for contained list of Part objects.

            Instance attributes:
                qr_code: qar.Code
                    qar.Code object obtained from qar.detectCodes() function.
                parts: List[Part]
                    List of Part objects associated with the qar. Code object (QR Code).
        """

    def __init__(self, qr_code: qar.Code):
        """
        :param qr_code: qar.Code object from qar.detectCodes() function
        """
        self.qr_code = qr_code
        self.parts = []


class PartMemory:
    """PartMemory object contains dictionary which serves as memory for this application.

    Instance attributes:
        data: dict (key: qar.Code.id)
            List of Part objects assigned to id attribute of qar.Code object.
        max_hold_count: int
            Number of frames for which PartMemory object will hold data if qar.Code with specific id is not in frame.
    """

    def __init__(self, max_hold_count=50):
        """
        :param max_hold_count: number of frames for which hold data if qar.Code not in frame
        """

        self.data = dict()
        self.hold_counter = dict()
        self.max_hold_count = max_hold_count

    def resetCounter(self, code_id: int) -> None:
        """Reset (set value to 0) hold counter of given id.

        :param code_id: id of qar.Code object
        :return: None
        """

        self.hold_counter[code_id] = 0

    def incrementCounter(self, code_id: int) -> None:
        """Increment by 1 hold counter of given qar.Code id.

        If value exceeds maximal hold counter value,
        all saved information connected to given qar.Code id will be deleted.

        :param code_id: id of qar.Code object
        :return: None
        """

        self.hold_counter[code_id] += 1
        if self.hold_counter[code_id] > self.max_hold_count:
            self.delete(code_id)

    def delete(self, code_id: int) -> None:
        """Delete saved data connected to given qar.Code id.

        :param code_id: id of qar.Code object
        :return: None
        """

        self.data.pop(code_id)
        self.hold_counter.pop(code_id)


def codesToOrigins(qr_codes: List[qar.Code], part_memory: PartMemory, memory: qar.Memory) -> List[PartsOrigin]:
    """Tries to assign size and content to each qar.Code object in a List.

    If qar.Code.id is in Memory, values are assigned from it. Otherwise function will try to decode the code.
    If decoding isn't successful value of size and content attribute of qar.Code will remain as None.
    If qar.Code decoding is successful and content are valid it's saved in part_memory.
    Information about size of code could be included in code in format: " Sx" at the end of content data.
    Where "x" stands for whole number the space before "S".
    Note that in this application it is possible to assign size to qar.Code in .txt instructions.

    :param qr_codes: List of Code class objects
    :param part_memory: PartMemory class object
    :param memory: qar.Memory object
    :return: List of PartsOrigin objects
    """

    origins = []
    if not qr_codes:
        # Keys are qar.Code.ids.
        for key in tuple(part_memory.data):
            part_memory.incrementCounter(key)
        return []

    # Preallocate memory.
    temp = numpy.zeros(len(qr_codes), dtype=int)
    i = 0
    for qr_code in qr_codes:
        if qr_code.id is not None and qr_code.size is not None:
            # Existing values in PartMemory object -> assigment.
            if qr_code.id in part_memory.data:
                # Reset hold counter for qar.Code id because it is currently in frame.
                part_memory.resetCounter(qr_code.id)
                # Create and append PartOrigin object to the list.
                part_origin = PartsOrigin(qr_code)
                part_origin.parts = part_memory.data[qr_code.id]
                origins.append(part_origin)
            # Else if qar.Code is not in PartMemory object.
            elif qr_code.content is not None:
                data_content = qr_code.content.split(" ")
                # Check if there is valid url in content of qar.Code object.
                if data_content[0][0:7] == "http://" or data_content[0][0:8] == "https://":
                    url = data_content[0]
                    # Read instruction from .txt file on url. (Check for ".txt" is in readInstructions function.)
                    parts, qr_size = readInstructions(url)
                    # Url or instructions are not valid parts = [] and qr_size = 0.
                    if qr_size:
                        # Rewrite qar.Code.size and also its information in qar.Memory object.
                        qr_code.size = qr_size
                        memory.size[qr_code.id] = qr_size
                    # Save data to PartMemory object.
                    part_memory.data[qr_code.id] = parts
                    part_memory.resetCounter(qr_code.id)
                    # Create and append PartOrigin object to the list.
                    part_origin = PartsOrigin(qr_code)
                    part_origin.parts = parts
                    origins.append(part_origin)
            temp[i] = qr_code.id
            i += 1
        else:
            temp = numpy.delete(temp, i)

    for key in tuple(part_memory.data):
        # If Code.id exists in part_memory but is not currently in frame.
        if key not in temp:
            part_memory.incrementCounter(key)

    return origins


def readInstructions(url_to_txt: str) -> Tuple[List[Part], float]:
    """Read and interpret instructions from '.txt' file on given url.
    For supported instruction written in '.txt' file see 'demo_syntax.txt'

    :param url_to_txt: url starting ending with '.txt'
    :return: (List of Part objects, qr_size int)
    """

    part_list = []
    specs_multiline_flag = False
    name_flag = False
    contour_flag = False
    specs_flag = False
    link_flag = False
    qr_size = 0
    name = ''
    contour = []
    specs = ''
    link = ''

    # If url string doesn't end with ".txt"
    if not url_to_txt[-4::] == '.txt':
        return [], 0
    # Open txt file on specified url.
    txt = urlopen(url_to_txt)
    for line in txt:
        # Decode bytes to string.
        line_str = line.decode()
        # Remove spaces.
        line_str = line_str.lstrip()
        # Remove newline characters.
        line_str = line_str.rstrip('\r\n')
        try:
            # Skip comments (lines started with '#').
            if line_str[0] == '#':
                continue
            # Handle instructions.
            if line_str[0] == '/':
                specs_multiline_flag = False
                # /end
                if line_str[1:3] == 'end':
                    break
                # /qr_size:
                elif line_str[1:9] == 'qr_size:':
                    qr_size = line_str[9::].lstrip()
                    qr_size = float(qr_size)
                # /rectangle:
                elif line_str[1:11] == 'rectangle:':
                    rectangle = line_str[11::].lstrip()
                    rectangle = rectangle.replace(' ', ',').replace("(", "").replace(")", "").split(',')
                    # Preallocate contour ndarray.
                    contour = numpy.zeros((4, 3), dtype=numpy.float32)
                    # Match syntax.
                    x = float(rectangle[0])
                    y = float(rectangle[1])
                    z = float(rectangle[2])
                    w = float(rectangle[3])
                    h = float(rectangle[4])
                    # Create rectangle contour from given data.
                    contour[0] = (x-(w/2), y+(h/2), z)
                    contour[1] = (x+(w/2), y+(h/2), z)
                    contour[2] = (x+(w/2), y-(h/2), z)
                    contour[3] = (x-(w/2), y-(h/2), z)
                    contour_flag = True
                # /name:
                elif line_str[1:6] == 'name:':
                    name = line_str[6::].lstrip()
                    name_flag = True
                # /contour:
                elif line_str[1:9] == 'contour:':
                    coordinates = line_str[9::].lstrip()
                    coordinates = coordinates.replace(' ', ',').replace("(", "").replace(")", "")
                    # Reshape ndarray to coordinates format.
                    contour = numpy.array(coordinates.split(','), dtype=numpy.float32).reshape(-1, 3)
                    contour_flag = True
                # /specs:
                elif line_str[1:7] == 'specs:':
                    specs = line_str[7::].lstrip()
                    specs_multiline_flag = True
                    specs_flag = True
                # /link:
                elif line_str[1:6] == 'link:':
                    link = line_str[6::].lstrip()
                    link_flag = True
            elif specs_multiline_flag:
                specs = specs + '\n' + line_str
        except IndexError:
            continue
        if link_flag and specs_flag and contour_flag and name_flag:
            # Create and append Part object to list.
            part_list.append(Part(name, specs, contour, link))
            # Empty all variables.
            name_flag = False
            specs_flag = False
            contour_flag = False
            link_flag = False
            name = ''
            contour = []
            specs = ''
            link = ''
    return part_list, qr_size


def click_event(event: int, x: int, y: int, flags, param) -> None:
    """Handle click event.

    :param event: int specifying mouse action
    :param x: x coordinates of cursor
    :param y: y coordinates of cursor
    :param flags: setMouseCallback pushed flags
    :param param: setMouseCallback pushed param
    :return: None
    """

    global cursor_loc, lmb_flag
    # Handle LMB click.
    if event == cv2.EVENT_LBUTTONDOWN:
        lmb_flag = True
        # Write cursor coordinates to global variable.
        cursor_loc = [(x, y)]
    # Handle cursor position.
    elif event == cv2.EVENT_MOUSEMOVE:
        # Write cursor coordinates to global variable.
        cursor_loc = [(x, y)]


def mainAppFunctions(frame: numpy.ndarray, origin_list: List[PartsOrigin]) -> numpy.ndarray:
    """Handle all draw capabilities and actions.
    Note that given frame will remain unedited as function will create copy of it and return this copy.

    :param frame: frame as numpy ndarray (BGR colorspace)
    :param origin_list: list of PartOrigin objects
    :return: frame_copy with drew parts
    """

    global cursor_loc, lmb_flag
    # Copy frame.
    frame_copy = frame.copy()

    if not origin_list:
        return frame_copy

    hovered_part = 0
    # For PartOrigin object in origin list
    for part_origin in origin_list:
        # If for origin QR Code was successfully estimated pose.
        if part_origin.qr_code.rvec is not None:
            # For Part object in list of parts.
            for part in part_origin.parts:
                # Project part contour points via Part object method.
                part.project(part_origin.qr_code.rvec, part_origin.qr_code.tvec, cam_mtx, distortion)
                # Check if cursor is in projected Part contour.
                if cursor_loc and cv2.pointPolygonTest(part.projected_contour, cursor_loc[0], False) > 0:
                    # Draw hovered contour yellow.
                    part.draw(frame_copy, (0, 255, 255))
                    # Save hovered Part to variable to later draw its information on top of all contours.
                    hovered_part = part
                    # If mouse was clicked when Part hovered.
                    if lmb_flag:
                        # And web url is legit.
                        if part.link[0:7] == "http://" or part.link[0:8] == "https://":
                            # Print it to console.
                            print(part.link)
                            # Open web browser with given url.
                            webbrowser.open(part.link, new=2)
                        lmb_flag = False
                else:
                    # Draw not hovered part contours orange.
                    part.draw(frame_copy, (51, 204, 51))
            # Draw all information (text) about part next to the Cursor.
            if hovered_part:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                if hovered_part.link != "":
                    text = hovered_part.name + '\n' + hovered_part.specs + '\n' + "Link available (LMB to open)"
                else:
                    text = hovered_part.name + '\n' + hovered_part.specs
                putMultilineText(text, frame_copy, font, font_scale, font_thickness)
                hovered_part = 0
    lmb_flag = False

    return frame_copy


def putMultilineText(string: str, frame_copy: numpy.ndarray, font: int, font_scale: float, font_thickness: int) -> None:
    """Draws multiple line text next to cursor with background filled rectangle.

    :param string: text to be draw, multiline text must have newline character as such: '\n'
    :param frame_copy: frame as numpy ndarray (BGR colorspace)
    :param font: int specifying font to be rendered
    :param font_scale: scale of font to be rendered
    :param font_thickness: thickness of font to be rendered
    :return: None
    """

    global cursor_loc
    # Split string line bby line.
    string = string.split("\n")
    # Get dimensions of longest line when rendered.
    (width, height), baseline = cv2.getTextSize(max(string, key=len), font, font_scale, font_thickness)
    # Add gap between elements.
    gap = int(height * 0.8)
    # Set points for background rectangle on which text will be rendered.
    p_1 = (cursor_loc[0][0], cursor_loc[0][1])
    p_2 = (cursor_loc[0][0] + width + (2 * gap),
           cursor_loc[0][1] + int((len(list(filter(None, string))) + 0.5) * (height + gap)))
    # Draw background rectangle.
    cv2.rectangle(frame_copy, p_1, p_2, (255, 255, 255), -1)

    i = 1
    # Render whole given string line by line.
    for line in string:
        if line:
            y = i * (height + gap)
            cv2.putText(frame_copy, line, (cursor_loc[0][0] + gap, cursor_loc[0][1] + y), font, font_scale,
                        (0, 0, 0), font_thickness)
            i += 1


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
cv2.namedWindow("Demo app")

# Initialize variables further used as global.
cursor_loc = []
lmb_flag = False

# Hold memory count should be same for qar.init and PartMemory.
hold_mem_count = 120
# Tracker and memory init.
ct, mem = qar.init(hold_mem_count)
# Part memory init.
part_mem = PartMemory(hold_mem_count)

# Create DetectorParameters object.
params = qar.DetectorParameters()
# Modify few parameter for possibly best function in given application.
params.binaryInverse = False
params.dilateErodeIterations = 3
params.thresholdConstant = -11
params.invertFrameBeforeDecoding = True  # Needed for white coloured QR Codes, as ZBar decodes only dark coloured ones.

while True:
    # Load frame frame webcam feed.
    cam_frame = stream.read()
    # Detect and try to decode QR Codes.
    codes = qar.detectCodes(cam_frame, ct, mem, params)
    origs = codesToOrigins(codes, part_mem, mem)
    # Estimate qar.Codes poses.
    qar.estimatePose(codes, cam_mtx, distortion)
    # Handling mouse input/events.
    cv2.setMouseCallback("Demo app", click_event)
    # Draw application elements to the frame (returns frame copy, so original frame is still raw).
    cam_frame_copy = mainAppFunctions(cam_frame, origs)
    # Check for key input.
    if (cv2.waitKey(1) == ord("q")) or (cv2.waitKey(1) == 27):
        break
    # Check if window is closed.
    if cv2.getWindowProperty("Demo app", cv2.WND_PROP_VISIBLE) < 1:
        break
    # Show cam_frame
    cv2.imshow("Demo app", cam_frame_copy)

stream.stop()
cv2.destroyAllWindows()
sys.exit()
