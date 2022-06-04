"""QR code augmented reality library v1.0

This library is intended for estimation of pose of QR Codes without repeatedly decoding them.
Making it usable for AR applications, although QR Codes aren't ideal for this use (as fiducial markers).
Based on OpenCV 4.2, Numpy 1.18, Pyzbar 0.1.8 (for decoding) and Centroid-tracker 0.0.9.

QR Code is registered trademark of DENSO WAVE INCORPORATED.

Created DB 2020 with Python 3.8
"""

import math
from typing import List, Tuple, Union
import cv2
import numpy
from pyzbar.pyzbar import decode
from tracker import CentroidTracker

# Type aliases:
Coordinates = Tuple[int, int]
Vector = Tuple[int, int]


class Memory:
    """Memory object contains dictionaries for QR codes which acts like data memory.

    Instance attributes:
        size: dict (key: Code.id)
            Size data as float.
        data: dict (key: Code.id)
            Content of QR code as str.
        max_hold_count: int
            Number of frames for which Memory object will hold data if Code object with specific id is not in frame.
    """

    def __init__(self, max_hold_count=50):
        """
        :param max_hold_count: number of frames for which object will hold information connected to code id
        """

        self.size = dict()
        self.data = dict()
        self.hold_counter = dict()

        self.max_hold_count = max_hold_count

    def resetCounter(self, code_id: int) -> None:
        """Reset (set value to 0) hold counter of given id.

        :param code_id: id of Code object
        :return: None
        """

        self.hold_counter[code_id] = 0

    def incrementCounter(self, code_id: int) -> None:
        """Increment hold counter by 1 for given code id.

        If value exceeds maximal hold counter value, all saved information connected to given code id will be deleted.

        :param code_id: id of Code object
        :return: None
        """

        self.hold_counter[code_id] += 1
        if self.hold_counter[code_id] > self.max_hold_count:
            self.delete(code_id)

    def delete(self, code_id: int) -> None:
        """Delete all saved information connected to given code id.
        
        :param code_id: id of Code object
        :return: None
        """

        self.size.pop(code_id)
        self.data.pop(code_id)
        self.hold_counter.pop(code_id)


class CodeCandidate:
    """CodeCandidate object holds data for candidate evaluation process."""

    def __init__(self, outline_contour: numpy.ndarray, fp0: numpy.ndarray, fp1: numpy.ndarray, fp2: numpy.ndarray,
                 fp0c: Coordinates, fp1c: Coordinates, fp2c: Coordinates):
        """
        :param outline_contour: contour outlining finder patterns, most likely QR code outline
        :param fp0: first finder pattern contour
        :param fp1: second finder pattern contour
        :param fp2: third finder pattern contour
        :param fp0c: coordinates of first finder pattern contour centroid
        :param fp1c: coordinates of second finder pattern contour centroid
        :param fp2c: coordinates of third finder pattern contour centroid
        """

        self.outline = outline_contour
        self.fp0 = fp0
        self.fp1 = fp1
        self.fp2 = fp2
        self.fp0c = fp0c
        self.fp1c = fp1c
        self.fp2c = fp2c
        # Next section attributes are previous attributes after identifying finder pattern D.
        self.fp_d = None
        self.fp_i = None
        self.fp_j = None
        self.fp_dc = None
        self.fp_ic = None
        self.fp_jc = None
        # Next section attributes are previous attributes after identifying finder pattern A, C.
        self.fp_a = None
        self.fp_b = None
        self.fp_c = None
        self.fp_ac = None
        self.fp_cc = None
        # Next indexes are crucial for identification of corner A, C, D coordinates .
        self.index_i = None
        self.index_j = None
        self.index_k = None
        # In process of evaluation value relative to the QR code area in frame is saved to be passed to Code object.
        self.margin = None
        # Values used in drawCandidates function.
        self.testingContourValid = None
        self.testingContours = None


class Code:
    """Code object holds all relevant information about detected QR Code.

    Notation A, B, C, D is standard geometric notation for polygon.
    Take base position of QR code as finder patterns in lower left, upper left and upper right corner.
    Then A is lower left corner, B is corner without finder pattern, C is upper right corner and D is upper left corner.

    Instance attributes::
        id: int or None
            Holds id of Code object for Memory object utilization.
        content: str or None
            Decoded content of QR Code.
        size: int or None
            Size information from QR Code content. Legit values are bigger than 0. If 0 size information wasn't found.
        tvec: numpy.ndarray or None
            Translation vector from pose estimation.
        rvec: numpy.ndarray or None
            Rotation vector from pose estimation.


        A: Tuple(int, int)
            Coordinates of corner A.
        B: Tuple(int, int)
            Coordinates of corner B.
        C: Tuple(int, int)
            Coordinates of corner C.
        D: Tuple(int, int)
            Coordinates of corner D.
        margin: int
            Value relative to the QR code area in frame.
            Used before decoding to ensure whole QR code will be in cropped frame.
        boundRect:
            Bounding rectangle of the QR code.
    """

    def __init__(self, corner_a: Coordinates, corner_b: Coordinates, corner_c: Coordinates,
                 corner_d: Coordinates, margin: int):
        """
        :param corner_a: coordinates of corner A
        :param corner_b: coordinates of corner B
        :param corner_c: coordinates of corner C
        :param corner_d: coordinates of corner D
        :param margin: int relative to the QR code area in frame
        """

        self.A = corner_a
        self.B = corner_b
        self.C = corner_c
        self.D = corner_d
        self.margin = margin

        self.boundRect = None

        self.id = None
        self.content = None
        self.size = None
        self.tvec = None
        self.rvec = None


class DetectorParameters:
    """DetectorParameters object contains and specifies parameters used in detection and evaluation functions.

    Function detectCodes() uses all contained parameters.
    Default values are set, feel free to modify it as testing was done on limited hardware.

    Attributes:
         ADAPTIVE THRESHOLD PARAMS:
             binaryInverse: bool
                Keep True for dark coloured QR codes, otherwise False.
             thresholdBlockSize: int
                Must be odd and larger than 1.
             thresholdConstant: int

         BLUR PARAMS:
            blurKernelSize: int
                Must be odd and larger than 1.
            
         FINDER PATTERN DETECTION PARAMS:
             epsilonCoefficientFP: float
                Multiplies contour length before polygon approximation to specify accuracy.
             minAreaRateFP: float
                Further multiplied by 0.001.
                Multiplies frame area to create a minimal area condition for detected contours of inner finder pattern.
             maxAreaRateFP: float
                Multiplies frame area to create a maximal area condition for detected contours of inner finder pattern.
             areaRatioTolerance: int
                Tolerance is added to or subtracted from mean value.
                Mean value of the three areas (in finder pattern) is calculated and acting as condition.

         CANDIDATE DETECTION PARAMS:
             dilateErodeKernelSize: int
             dilateErodeIterations: int
             epsilonCoefficientCand: float
                Multiplies contour length before polygon approximation to specify accuracy.
             minAreaRateCand: float
                Further multiplied by 0.001.
                Multiplies frame area to create a minimal area condition for detected contours.
             maxAreaRateCand: float
                Multiplies frame area to create a maximal area condition for detected contours.
             
         CANDIDATE EVALUATION PARAMS:
             testContourDeviationRate: float
                Multiplies distance of points from center line in QR code hypotenuse testing contour.
             magnetB: bool
                If True corner B "magnets" to CodeCandidate outline contour point if its in close distance.
             magnetDistanceRate: float
                Multiplies magnet distance which value is same as in margin attribute of Code and CodeCandidate object.

         ADDITIONAL:
             returnCandidates: bool
                 If True, detectCodes function returns both List[Code] and List[CodeCandidate]
             visualizeDetectorSteps: bool
                 If True, detectCodes function returns frame with drew contours.
                 For debug and parameters setting tests.
             invertFrameBeforeDecoding: bool
                 If True cropped frame will be grayscale inverted before decoding (recommended for white QR Codes).
    """

    def __init__(self):
        # Threshold settings.
        self.binaryInverse = True
        self.thresholdBlockSize = 55
        self.thresholdConstant = 7
        # Blur setting.
        self.blurKernelSize = 5
        # Finder pattern detection settings.
        self.epsilonCoefficientFP = 0.05
        self.minAreaRateFP = 0.09
        self.maxAreaRateFP = 0.7
        self.areaRatioTolerance = 2
        # Candidates find settings.
        self.dilateErodeKernelSize = 3
        self.dilateErodeIterations = 5
        self.epsilonCoefficientCand = 0.05
        self.minAreaRateCand = 2.0
        self.maxAreaRateCand = 0.9
        # Candidate evaluation.
        self.testContourDeviationRate = 1.1
        self.magnetB = True
        self.magnetDistanceRate = 1.5

        # Additional
        self.returnCandidates = False
        self.visualizeDetectorSteps = False
        self.invertFrameBeforeDecoding = False


def init(data_hold_frames_count: int = 50) -> Tuple[CentroidTracker, Memory]:
    """Initialize centroid tracker and memory for QR codes.

    :param data_hold_frames_count: number of frames for which object will hold information about code id
    :return: (CentroidTracker class object, Memory class object)
    """

    centroid_tracker = CentroidTracker(data_hold_frames_count)
    memory = Memory(data_hold_frames_count)
    return centroid_tracker, memory


def detectCodes(frame: numpy.array, centroid_tracker: CentroidTracker, memory: Memory,
                parameters: DetectorParameters = DetectorParameters()) \
        -> Union[List[Code], Tuple[List[Code], List[CodeCandidate]], Tuple[List[Code], numpy.ndarray],
                 Tuple[List[Code], List[CodeCandidate], numpy.ndarray]]:
    """Detect and return QR Codes as List of Code class objects.

    :param frame: frame as numpy ndarray (BGR colorspace)
    :param centroid_tracker: CentroidTracker class object
    :param memory: Memory class object
    :param parameters: DetectorParameters class object
    :return: List of Code class objects (and List of CodeCandidate class objects if specified in DetectorParameters)
    """

    if parameters.visualizeDetectorSteps:
        frame_copy = frame.copy()
    else:
        frame_copy = None

    if parameters.binaryInverse:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (parameters.blurKernelSize, parameters.blurKernelSize), 0)
    threshold = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type,
                                      parameters.thresholdBlockSize, parameters.thresholdConstant)
    fp_candidates, mc = finderPatternDetect(threshold, parameters.epsilonCoefficientFP, parameters.minAreaRateFP,
                                            parameters.maxAreaRateFP, parameters.areaRatioTolerance, frame_copy)
    code_candidates = candidatesFind(threshold, fp_candidates, mc, parameters.epsilonCoefficientCand,
                                     parameters.dilateErodeKernelSize, parameters.dilateErodeIterations,
                                     parameters.minAreaRateCand, parameters.maxAreaRateCand, frame_copy)
    codes = candidatesEvaluate(code_candidates, parameters.magnetB, parameters.magnetDistanceRate,
                               parameters.testContourDeviationRate, frame_copy)
    trackIDs(centroid_tracker, codes)
    assignSizeAndData(codes, memory, frame, parameters.invertFrameBeforeDecoding)

    if parameters.returnCandidates:
        if parameters.visualizeDetectorSteps:
            return codes, code_candidates, frame_copy
        else:
            return codes, code_candidates
    else:
        if parameters.visualizeDetectorSteps:
            return codes, frame_copy
        else:
            return codes


def finderPatternDetect(threshold: numpy.ndarray,
                        epsilon_coefficient: float = DetectorParameters().epsilonCoefficientFP,
                        min_area_rate: float = DetectorParameters().minAreaRateFP,
                        max_area_rate: float = DetectorParameters().maxAreaRateFP,
                        ratio_tolerance: int = DetectorParameters().areaRatioTolerance,
                        frame_copy: numpy.ndarray = None) -> Tuple[List[numpy.ndarray], List[Coordinates]]:
    """Detect finder patterns in given thresholded frame.

    :param threshold: thresholded frame (binary inverse for dark coloured QR codes).
    :param epsilon_coefficient: utilized in polygon approximation accuracy
    :param min_area_rate: multiplies frame area to create a minimal area condition (further multiplied by 0.001)
    :param max_area_rate: multiplies frame area to create a maximal area condition
    :param ratio_tolerance: tolerance around mean value when utilizing 3:5:7 area ratio condition
    :param frame_copy: by default None, copy of frame to which contours will be drawn
    :return: (List of outer finder patten contours, centroids of those contours as List of coordinates)
    """

    # Contour hierarchy aliases.
    child_contour = 2
    parent_contour = 3
    # Resolution area for relative contour detection conditions.
    res_area = numpy.shape(threshold)[0] * numpy.shape(threshold)[1]
    min_area = res_area * (min_area_rate * 0.001)
    max_area = res_area * max_area_rate
    # Find contours and their hierarchy in thresholded image.
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Further used variables init.
    cnt_index = 0
    finder_pattern_candidates = []
    # SEARCHING FOR FINDER PATTERN:
    for contour in contours:
        epsilon = epsilon_coefficient * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area1 = cv2.contourArea(approx)
        # If contour has 4 corners, area larger than minimal, smaller than maximal and is convex.
        if (len(approx) == 4) and (cv2.isContourConvex(approx)) and (area1 > min_area) and (area1 < max_area):
            # If contour is the most inner contour (youngest).
            if (hierarchy[0][cnt_index][child_contour] == -1) and (
                    hierarchy[0][cnt_index][parent_contour] != -1):
                if frame_copy is not None:
                    cv2.drawContours(frame_copy, [approx], -1, (0, 255, 255), 1)
                # Find its parent.
                finder_index = hierarchy[0][cnt_index][parent_contour]
                approx = cv2.approxPolyDP(contours[finder_index], epsilon, True)
                area2 = cv2.contourArea(approx)
                # Find its grandparent (this may be outer finder pattern contour).
                finder_index = hierarchy[0][finder_index][parent_contour]
                epsilon = epsilon_coefficient * cv2.arcLength(contours[finder_index], True)
                approx = cv2.approxPolyDP(contours[finder_index], epsilon, True)
                area3 = cv2.contourArea(approx)
                # Check if contour has 4 corners, area larger than given value and is convex.
                if (len(approx) == 4) and (cv2.isContourConvex(approx)) and (area3 > (min_area * 1.1)) \
                        and (area3 < (max_area * 1.1)):
                    a = math.sqrt(area1) / 3
                    b = math.sqrt(area2) / 5
                    c = math.sqrt(area3) / 7
                    mean = (a + b + c) / 3
                    t = ratio_tolerance
                    if frame_copy is not None:
                        cv2.drawContours(frame_copy, [approx], -1, (0, 125, 255), 1)
                    # Contour areas in finder pattern are in ratio 3:5:7.
                    if (mean - t) < a < (mean + t) and (mean - t) < b < (mean + t) and (mean - t) < c < (mean + t):
                        finder_pattern_candidates.append(approx)
                        if frame_copy is not None:
                            cv2.drawContours(frame_copy, [approx], -1, (0, 255, 255), 2)
        cnt_index += 1

    if not finder_pattern_candidates:
        return [], []

    finder_pattern_centroids = getContourCentroids(finder_pattern_candidates)

    if None in finder_pattern_centroids:
        return [], []

    # Finder pattern cannot contain another finder pattern.
    # Further used variables init.
    i = 0
    delete_flag = 0
    # Check every finder pattern to each other.
    while i < len(finder_pattern_candidates):
        if cv2.contourArea(finder_pattern_candidates[i]) > (min_area * 1.2):
            j = 0
            if j == i:
                j += 1
            while j < len(finder_pattern_candidates):
                # If any candidate contour contains centroid of another candidate contour, delete it.
                if cv2.pointPolygonTest(finder_pattern_candidates[i], finder_pattern_centroids[j], False) > 0:
                    del finder_pattern_candidates[i]
                    del finder_pattern_centroids[i]
                    delete_flag = 1
                    if i != 0:
                        i -= 1
                        break
                j += 1
                if j == i:
                    j += 1
        if delete_flag == 1:
            delete_flag = 0
        else:
            i += 1

    return finder_pattern_candidates, finder_pattern_centroids


def candidatesFind(threshold: numpy.ndarray, finder_pattern_candidates: List[numpy.ndarray],
                   centroids: List[Tuple[int, int]],
                   epsilon_coef: float = DetectorParameters().epsilonCoefficientCand,
                   dilate_erode_kernel_size: int = DetectorParameters().dilateErodeKernelSize,
                   dilate_erode_iterations: int = DetectorParameters().dilateErodeIterations,
                   min_area_rate: float = DetectorParameters().minAreaRateCand,
                   max_area_rate: float = DetectorParameters().maxAreaRateCand,
                   frame_copy: numpy.ndarray = None) -> List[CodeCandidate]:
    """Find QR code candidates in given thresholded frame.

    Designed to be used after finderPatternDetect() function as it need its output.
    Also thresholded image can be the same for both functions.

    :param threshold: thresholded frame (binary inverse for dark coloured QR codes)
    :param finder_pattern_candidates: List of outer finder patten contours
    :param centroids: List of centroid coordinates of finder pattern candidates contours
    :param epsilon_coef: utilized in polygon approximation accuracy
    :param dilate_erode_kernel_size: size of kernel (kernel is square) for dilate and erode functions
    :param dilate_erode_iterations: iterations of dilate and erode functions
    :param min_area_rate: multiplies frame area to create a minimal area condition (further multiplied by 0.001)
    :param max_area_rate: multiplies frame area to create a maximal area condition
    :param frame_copy: by default None, copy of frame to which contours will be drawn
    :return: List of CodeCandidate class objects.
    """

    if not centroids:
        return []

    # Dilate and erode thresholded frame to make QR code look as one white filled 4 point polygon.
    ker = dilate_erode_kernel_size
    i = dilate_erode_iterations
    threshold = cv2.dilate(threshold, numpy.ones((ker, ker), numpy.uint8), iterations=i)
    threshold = cv2.erode(threshold, numpy.ones((ker, ker), numpy.uint8), iterations=i)
    # Find contours in previously edited thresholded frame.
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Resolution area for relative contour detection conditions.
    res_area = numpy.shape(threshold)[0] * numpy.shape(threshold)[1]
    min_area = res_area * (min_area_rate * 0.001)
    max_area = res_area * max_area_rate
    # Further used variables init.
    contour_index = 0
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        epsilon = epsilon_coef * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # If contour has 4 corners, area larger than minimal, smaller than maximal and is convex.
        if (len(approx) == 4) and (area > min_area) and (area < max_area) and (cv2.isContourConvex(approx)):
            if frame_copy is not None:
                cv2.drawContours(frame_copy, [approx], -1, (20, 20, 255), 1)
            # Further used variables init.
            test_counter = 0
            index = 0
            fp_0, fp_1, fp_2 = 0, 0, 0
            for centroid in centroids:
                # Test if finder pattern candidate's centroid is inside contour.
                if cv2.pointPolygonTest(cnt, centroid, False) > 0:
                    test_counter += 1
                    if test_counter == 1:
                        fp_0 = index
                    elif test_counter == 2:
                        fp_1 = index
                    elif test_counter == 3:
                        fp_2 = index
                    else:
                        fp_0, fp_1, fp_2 = 0, 0, 0
                index += 1
            # If contour contains exactly 3 finder pattern candidate's centroids, create CodeCandidate class object.
            if test_counter == 3:
                candidates.append(CodeCandidate(approx, finder_pattern_candidates[fp_0],
                                                finder_pattern_candidates[fp_1], finder_pattern_candidates[fp_2],
                                                centroids[fp_0], centroids[fp_1], centroids[fp_2]))
                if frame_copy is not None:
                    cv2.drawContours(frame_copy, [approx], -1, (0, 0, 255), 1)
        contour_index += 1

    # Each trio of finder patterns must be included only in one QR code candidate.
    # Further used variables init.
    i = 0
    delete_flag = 0
    while i < len(candidates):
        j = i + 1
        while j < len(candidates):
            if candidates[i].fp0c == candidates[j].fp0c:
                # If the same finder pattern is included in two candidates, candidate with smaller area remains.
                if cv2.contourArea(candidates[i].outline) > cv2.contourArea(candidates[j].outline):
                    del candidates[i]
                    delete_flag = 1
                    if i != 0:
                        i -= 1
                    break
                else:
                    del candidates[j]
                    delete_flag = 1
                    break
            j += 1
        if delete_flag == 1:
            delete_flag = 0
        else:
            i += 1

    if frame_copy is not None:
        for candidate in candidates:
            cv2.drawContours(frame_copy, [candidate.outline], -1, (0, 0, 255), 2)

    return candidates


def candidatesEvaluate(candidates: List[CodeCandidate], magnet_b: bool = DetectorParameters().magnetB,
                       magnet_distance_rate: float = DetectorParameters().magnetDistanceRate,
                       test_contour_deviation_rate: float = DetectorParameters().testContourDeviationRate,
                       frame_copy: numpy.ndarray = None) -> List[Code]:
    """Evaluate each QR code candidate and return recognized QR codes as List of Code class objects.

    :param candidates: List of CodeCandidate class objects
    :param magnet_b: if True corner B "magnets" to CodeCandidate outline contour point if its in close distance
    :param magnet_distance_rate: multiplies magnet distance (CodeCandidate.margin)
    :param test_contour_deviation_rate: multiplies distance of points from center line in "hypotenuse" testing contour
    :param frame_copy: by default None, copy of frame to which contours will be drawn
    :return: List of Code class objects
    """

    if not candidates:
        return []

    qr_codes = []
    for candidate in candidates:
        identifyFinderPatternD(candidate, test_contour_deviation_rate, frame_copy)
        # Identify finder patterns A, C, outer QR code corners and points for corner B estimation.
        a1, a2, c1, c2, d = identifyFinderPatterns(candidate)

        if None in (a1, a2, c1, c2, d):
            continue

        # Assign coordinates of all 4 code corners to its identifier.
        corner_a = a1
        corner_b = intersectionOfLines(a1, a2, c1, c2)
        corner_c = c1
        corner_d = d

        if corner_b is None:
            continue

        if frame_copy is not None:
            cv2.line(frame_copy, corner_a, corner_b, (180, 120, 120), 2)
            cv2.line(frame_copy, corner_c, corner_b, (180, 120, 120), 2)
            cv2.circle(frame_copy, a1, 3, (0, 255, 0), -1)
            cv2.circle(frame_copy, a2, 3, (100, 255, 100), -1)
            cv2.circle(frame_copy, c1, 3, (0, 255, 0), -1)
            cv2.circle(frame_copy, c2, 3, (100, 255, 100), -1)

        if magnet_b:
            # CodeCandidate's attribute margin holds value relative to code size in frame.
            magnet_distance = int(candidate.margin * magnet_distance_rate)

            if frame_copy is not None:
                cv2.circle(frame_copy, corner_b, magnet_distance, (100, 255, 100), 1)

            for point in candidate.outline:
                if int(pointsDistance(corner_b, point[0])) < magnet_distance:
                    corner_b = (point[0][0], point[0][1])
                    break

        if frame_copy is not None:
            cv2.circle(frame_copy, corner_b, 3, (0, 255, 0), -1)

        # If evaluation is successful Code class object is appended to List.
        qr_codes.append(Code(corner_a, corner_b, corner_c, corner_d, candidate.margin))

    return qr_codes


def trackIDs(centroid_tracker: CentroidTracker, qr_codes: List[Code]) -> None:
    """Track ids of codes using "simple centroid tracker".

    :param centroid_tracker: CentroidTracker class object
    :param qr_codes: List of Code class objects
    :return: None
    """

    if not qr_codes:
        return

    bounding_rects = []
    for qr_code in qr_codes:
        # Make pseudo contour (same structure as cv2 contours) for use in boundingRect() function.
        code_contour = numpy.array([[qr_code.A], [qr_code.B], [qr_code.C], [qr_code.D]], dtype=numpy.int32)
        x, y, w, h = cv2.boundingRect(code_contour)
        # Modify structure for proper centroid tracker function.
        rect = (x, y, x + w, y + h)
        qr_code.boundRect = rect
        bounding_rects.append(rect)
    # Use centroid tracker.
    objects = centroid_tracker.update(bounding_rects)
    # Assign ids from CentroidTracker to individual QR Codes.
    for (objectID, rect) in objects[1].items():
        i = 0
        for r in bounding_rects:
            if r[0] == rect[0] and r[1] == rect[1]:
                qr_codes[i].id = objectID
                continue
            i += 1


def assignSizeAndData(qr_codes: List[Code], memory: Memory, frame: numpy.ndarray,
                      invert_crop: bool = DetectorParameters().invertFrameBeforeDecoding) -> None:
    """Tries to assign size and content to each Code object in a List.

    If Code.id is in Memory, values are assigned from it. Otherwise function will try to decode the code.
    If decoding isn't successful value of size and content attribute of Code will remain as None.
    If Code decoding is successful and content are valid it's saved in memory.
    Information about size of code must be included in code in format: " Sx" at the end of content data.
    Where "x" stands for whole number the space before "S".

    :param qr_codes: List of Code class objects
    :param memory: Memory class object
    :param frame: unprocessed frame as numpy ndarray (BGR colorspace)
    :param invert_crop: if True cropped image will be inverted before decoding
    :return: None
    """

    if not qr_codes:
        # Keys are Code.ids.
        for key in tuple(memory.size):
            memory.incrementCounter(key)
        return

    # Preallocate memory.
    temp = numpy.zeros(len(qr_codes), dtype=int)
    i = 0
    for qr_code in qr_codes:
        if qr_code.id is not None:
            # Existing values in Memory object -> assigment.
            if qr_code.id in memory.size:
                # Reset hold counter for code id because it is currently in frame.
                memory.resetCounter(qr_code.id)
                qr_code.size = memory.size[qr_code.id]
                qr_code.content = memory.data[qr_code.id]
            temp[i] = qr_code.id
            i += 1
        else:
            temp = numpy.delete(temp, i)

    for key in tuple(memory.size):
        # If Code.id exists in memory but currently is not in frame.
        if key not in temp:
            memory.incrementCounter(key)

    # Obtain resolution of frame.
    x_max = numpy.shape(frame)[1]
    y_max = numpy.shape(frame)[0]

    for qr_code in qr_codes:
        # If size for code wasn't assigned yet.
        if qr_code.size is None:
            # Add a margin around bounding box to ensure whole QR code will be in cropped frame.
            margin = qr_code.margin
            (x1, y1, x2, y2) = qr_code.boundRect
            x1 = x1 - margin
            x2 = x2 + margin
            y1 = y1 - margin
            y2 = y2 + margin
            # Cant crop out of frame dimensions (resolution of frame).
            if x1 < 1:
                x1 = 1
            if y1 < 1:
                y1 = 1
            if x2 > (x_max - 1):
                x2 = x_max - 1
            if y2 > (y_max - 1):
                y2 = y_max - 1
            # Crop the frame.
            crop_frame = frame[y1:y2, x1:x2]
            if invert_crop:
                crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
                crop_frame = cv2.bitwise_not(crop_frame)
            # Decode QR code from cropped frame using zbar function decode(). May be replaced with different decoder.
            decoded = decode(crop_frame)
            # Check if decoded data are valid. (Most likely they would be.)
            if decoded and (decoded[0].type == 'QRCODE'):
                data_content = decoded[0].data
                # Decode bytes to str and assign to content attribute of Code object.
                qr_code.content = data_content.decode()
                data_content = qr_code.content
                # Save content to Memory object.
                memory.data[qr_code.id] = data_content
                # Split string by spaces " ".
                data_content = data_content.split(" ")
                # Search for size information in end section of split content string. If not found size = 0.
                index = data_content[-1].find("S")
                # Extract number from string. Also handling wrong formats properly.
                if index != -1:
                    number_digits = len(data_content[-1]) - 1 - index
                    verified_digits = 0
                    for i in range(number_digits):
                        if not data_content[-1][index + 1 + i].isdigit():
                            break
                        verified_digits += 1
                    size = ''
                    for j in range(verified_digits):
                        size += data_content[-1][index + 1 + j]
                    if size == '':
                        size = 0
                    else:
                        size = float(size)
                else:
                    size = 0
                # Assign size to size attribute of Code object.
                qr_code.size = size
                # Update Memory class object.
                memory.size[qr_code.id] = size
                # Reset hold counter for code id because it is currently in frame.
                memory.resetCounter(qr_code.id)


def estimatePose(qr_codes: List[Code], camera_matrix: numpy.ndarray,
                 distortion_coefficients: numpy.ndarray, not_legit: bool = False) -> None:
    """Solve PnP for each Code object given in a List and assign rvec and tvec attributes to them.

    rvec - Rotation vector.
    tvec - Translation vector.

    If parameter not_legit is set to True, function "estimates pose" even for Code objects with size attribute set to 0.
    This may be useful in some use cases, keep in mind that rvecs and tvecs values will be relative to "one code size".

    :param qr_codes: List of Code class objects
    :param camera_matrix: camera matrix of used camera obtained from camera calibration process
    :param distortion_coefficients: distortion coefficients obtained from camera calibration process
    :param not_legit: by default False
    :return: None
    """

    if not qr_codes:
        return

    for qr_code in qr_codes:
        if qr_code.size is not None:
            if qr_code.size != 0:
                # Set real object point coordinates.
                objp = numpy.float32([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]]) * qr_code.size
            else:
                if not_legit:
                    # Set object point coordinates without real world size.
                    objp = numpy.float32([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]])
                else:
                    continue
            # Use coordinates of QR Code corners.
            imgp = numpy.float32([qr_code.D, qr_code.C, qr_code.B, qr_code.A])
            # Solve PnP with method suitable for marker pose estimation.
            _, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, distortion_coefficients, cv2.SOLVEPNP_IPPE_SQUARE)
            # Assign resulting vectors to relevant attributes of Code class object.
            qr_code.tvec = tvec
            qr_code.rvec = rvec


def getContourCentroids(contours: List[numpy.ndarray]) -> List[Tuple[int, int]]:
    """Return list of centroids of input list of contours.

    :param contours: List of contours
    :return: List of centroid coordinates
    """

    # Preallocate memory.
    moments = [None] * len(contours)
    centroids = [None] * len(contours)
    # Get moments for finder pattern candidates.
    for i in range(len(contours)):
        moments[i] = cv2.moments(contours[i])
    # Get mass centers coordinates of finder pattern candidates.
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        centroids[i] = (int(moments[i]['m10'] / (moments[i]['m00'] + 1e-5)),
                        int(moments[i]['m01'] / (moments[i]['m00'] + 1e-5)))
    if None in centroids:
        return []
    else:
        return centroids


def identifyFinderPatternD(candidate: CodeCandidate,
                           test_contour_deviation_rate: float = DetectorParameters().testContourDeviationRate,
                           frame_copy: numpy.ndarray = None) -> None:
    """Resolves which of finder patterns is finder pattern D.

    :param candidate: CodeCandidate class object
    :param test_contour_deviation_rate: multiplies distance of points from center line in "hypotenuse" testing contour
    :param frame_copy: by default None, copy of frame to which contours will be drawn
    :return: None
    """

    distance = [pointsDistance(candidate.fp1c, candidate.fp2c),
                pointsDistance(candidate.fp0c, candidate.fp2c),
                pointsDistance(candidate.fp0c, candidate.fp1c)]
    # Make triangle contour from finder pattern centroids.
    triangle = numpy.array([candidate.fp0c, candidate.fp1c, candidate.fp2c], dtype=numpy.int32)
    area = cv2.contourArea(triangle)
    # Distance from center point to one of side points of QR Code hypotenuse testing contour.
    test_contour_distance = area ** (1/3)
    # Save this code size relative value to margin attribute of CodeCandidate object.
    candidate.margin = int(test_contour_distance)
    test_contour_distance = int(test_contour_distance * test_contour_deviation_rate)
    # Further used variables init.
    temp_max = 0
    index = 0
    # Find largest distance, which is may be hypotenuse of QR Code (line connecting finder patterns A and C).
    for i in range(3):
        if temp_max < distance[i]:
            temp_max = distance[i]
            index = i

    contours = []
    for x in range(4):
        # Not involved finder pattern must finder pattern D. Reassign all finder patterns for further evaluation.
        if index == 0:
            # Finder pattern contour reassign.
            candidate.fp_d = candidate.fp0
            candidate.fp_i = candidate.fp1
            candidate.fp_j = candidate.fp2
            # Finder pattern centroid coordinates reassign.
            candidate.fp_dc = candidate.fp0c
            candidate.fp_ic = candidate.fp1c
            candidate.fp_jc = candidate.fp2c
        elif index == 1:
            candidate.fp_d = candidate.fp1
            candidate.fp_i = candidate.fp0
            candidate.fp_j = candidate.fp2
            candidate.fp_dc = candidate.fp1c
            candidate.fp_ic = candidate.fp0c
            candidate.fp_jc = candidate.fp2c
        elif index == 2:
            candidate.fp_d = candidate.fp2
            candidate.fp_i = candidate.fp1
            candidate.fp_j = candidate.fp0
            candidate.fp_dc = candidate.fp2c
            candidate.fp_ic = candidate.fp1c
            candidate.fp_jc = candidate.fp0c

        if distance[index] == 0:
            return

        # Make vector from fp_ic to fp_jc further referred to as VECTOR.
        vec = pointsToVec(candidate.fp_ic, candidate.fp_jc)
        # Calculate sine and cosine of VECTOR.
        sin = vec[1] / distance[index]
        cos = vec[0] / distance[index]
        # Calculate distance for middle point of VECTOR.
        dp = distance[index] / 2
        # Calculate coordinates of middle point of VECTOR.
        px = candidate.fp_ic[0] + int(cos * dp)
        py = candidate.fp_ic[1] + int(sin * dp)
        # Calculate coordinates of two points used for QR Code hypotenuse testing contour.
        one_x = px + int(sin * test_contour_distance)
        one_y = py - int(cos * test_contour_distance)
        two_x = px - int(sin * test_contour_distance)
        two_y = py + int(cos * test_contour_distance)
        p1 = (one_x, one_y)
        p2 = (two_x, two_y)
        # Make testing contour (same structure as cv2 contours) for use in pointPolygonTest() function.
        contour = [numpy.array([candidate.fp_ic, p1, candidate.fp_jc, p2], dtype=numpy.int64)]
        if frame_copy is not None:
            cv2.drawContours(frame_copy, [contour[0]], -1, (255, 0, 0), 1)
        # Further used variables init.
        counter_i = 0
        counter_j = 0
        index_i = 0
        index_j = 0
        index_k = 0
        temp_index_i = 0
        temp_index_j = 0
        point_i = None
        point_j = None
        for pnt in candidate.fp_i:
            point = (pnt[0][0], pnt[0][1])
            if cv2.pointPolygonTest(contour[0], point, False) > 0:
                point_i = point
                index_i = temp_index_i
                counter_i += 1
            temp_index_i += 1
        for pnt in candidate.fp_j:
            point = (pnt[0][0], pnt[0][1])
            if cv2.pointPolygonTest(contour[0], point, False) > 0:
                point_j = point
                index_j = temp_index_j
                counter_j += 1
            temp_index_j += 1
        # Test if exactly two points of involved finder patterns (fp_i and fp_j) contours are inside testing contour.
        if counter_i == 1 and counter_j == 1:
            if frame_copy is not None:
                cv2.drawContours(frame_copy, [contour[0]], -1, (255, 0, 0), 2)
            # Assigning indexes of inner corners of fp_i and fp_j to relevant attributes of CodeCandidate object.
            candidate.index_i = index_i
            candidate.index_j = index_j
            # Make another testing contour, now to find inner corner of finder pattern D.
            contour = [numpy.array([point_i, point_j, candidate.fp_dc], dtype=numpy.int64)]
            for pnt in candidate.fp_d:
                point = (pnt[0][0], pnt[0][1])
                if cv2.pointPolygonTest(contour[0], point, False) > 0:
                    if frame_copy is not None:
                        cv2.drawContours(frame_copy, [contour[0]], -1, (255, 100, 100), 2)
                        cv2.circle(frame_copy, point_i, 3, (0, 0, 255), -1)
                        cv2.circle(frame_copy, point_j, 3, (0, 0, 255), -1)
                        cv2.circle(frame_copy, point, 3, (0, 0, 255), -1)
                    # Assigning index of inner corner of fp_d to relevant attributes of CodeCandidate object.
                    candidate.index_k = index_k
                    break
                index_k += 1
            return
        # If QR Code hypotenuse test isn't successful try a different side of triangle.
        else:
            # Saving values used in drawCandidates function.
            contours.append(contour)
            candidate.testingContours = contours
            if index == 2:
                index = 0
            else:
                index += 1


def identifyFinderPatterns(candidate: CodeCandidate) \
        -> Union[Tuple[Coordinates, Coordinates, Coordinates, Coordinates, Coordinates],
                 Tuple[None, None, None, None, None]]:
    """Identify finder patterns A, C, outer QR code corners (A, C, D) and points for corner B estimation.

    Notation A, B, C, D is standard geometric notation for polygon.
    Take base position of QR code as finder patterns in lower left, upper left and upper right corner.
    Then A is lower left corner, B is corner without finder pattern, C is upper right corner and D is upper left corner.

    :param candidate: CodeCandidate class object
    :return: a1, a2, c1, c2, d (all corner coordinates, a2 and c2 are points for B estimation)
    """

    if candidate.fp_d is None or candidate.index_i is None or candidate.index_j is None or candidate.index_k is None:
        return None, None, None, None, None

    i2 = None
    j2 = None
    # Since finder pattern contours are formed by 4 points opposite point is 2 indexes away.
    # Index attributes relates to inner corners of finder patterns. Index_k belongs to finder pattern D contour.
    index_i_opposite = candidate.index_i + 2
    index_j_opposite = candidate.index_j + 2
    index_k_opposite = candidate.index_k + 2
    if index_i_opposite > 3:
        index_i_opposite -= 4
    if index_j_opposite > 3:
        index_j_opposite -= 4
    if index_k_opposite > 3:
        index_k_opposite -= 4
    # Outer corner D coordinates.
    d = candidate.fp_d[index_k_opposite][0]
    # In case of other two finder patterns we also need points for estimation of corner B.
    # If i1 and j1 are outer corners, lines (i1, i2) and (j1, j2) must intersect at point B.
    index_i_next = candidate.index_i + 1
    index_j_next = candidate.index_j + 1
    if index_i_next > 3:
        index_i_next -= 4
    if index_j_next > 3:
        index_j_next -= 4
    adept_i1 = candidate.fp_i[index_i_next][0]
    adept_j1 = candidate.fp_j[index_j_next][0]
    # There are two adepts for points i2 and j2.
    index_i_next_opposite = index_i_next + 2
    index_j_next_opposite = index_j_next + 2
    if index_i_next_opposite > 3:
        index_i_next_opposite -= 4
    if index_j_next_opposite > 3:
        index_j_next_opposite -= 4
    adept_i2 = candidate.fp_i[index_i_next_opposite][0]
    adept_j2 = candidate.fp_j[index_j_next_opposite][0]
    # Correct point is obtained from vector angle test.
    testing_vec = pointsToVec(d, candidate.fp_dc)
    test_vec_i1 = pointsToVec(candidate.fp_ic, adept_i1)
    test_vec_i2 = pointsToVec(candidate.fp_ic, adept_i2)
    test_vec_j1 = pointsToVec(candidate.fp_jc, adept_j1)
    test_vec_j2 = pointsToVec(candidate.fp_jc, adept_j2)

    test_angle = math.degrees(math.atan2(testing_vec[1], testing_vec[0]))
    angle_i1 = math.degrees(math.atan2(test_vec_i1[1], test_vec_i1[0]))
    angle_i2 = math.degrees(math.atan2(test_vec_i2[1], test_vec_i2[0]))
    angle_j1 = math.degrees(math.atan2(test_vec_j1[1], test_vec_j1[0]))
    angle_j2 = math.degrees(math.atan2(test_vec_j2[1], test_vec_j2[0]))
    # Critical is difference between testing vector angle and tested vector angle.
    tests = [test_angle - angle_i1, test_angle - angle_i2, test_angle - angle_j1, test_angle - angle_j2]
    # Correction for angle difference values close to "angle border".
    for ind in range(4):
        if tests[ind] > 180:
            tests[ind] = - (360 - tests[ind])
        if tests[ind] < -180:
            tests[ind] = - (- 360 - tests[ind])
    # If difference is smaller than 90 degrees vectors have same direction.
    if abs(tests[0]) < 90:
        i2 = adept_i1
    elif abs(tests[1]) < 90:
        i2 = adept_i2

    if abs(tests[2]) < 90:
        j2 = adept_j1
    elif abs(tests[3]) < 90:
        j2 = adept_j2

    if i2 is None or j2 is None:
        return None, None, None, None, None

    # Make vectors from centroid of finder pattern D to other two finder pattern centroids.
    vec_d_i = pointsToVec(candidate.fp_dc, candidate.fp_ic)
    vec_d_j = pointsToVec(candidate.fp_dc, candidate.fp_jc)

    angle_i = math.degrees(math.atan2(vec_d_i[1], vec_d_i[0]))
    angle_j = math.degrees(math.atan2(vec_d_j[1], vec_d_j[0]))
    # Identifying finder pattern A and C is done via another vector angle test.
    difference = angle_i - angle_j
    # Correction for angle difference values close to "angle border".
    if difference > 180:
        difference = - (360 - difference)
    if difference < -180:
        difference = - (- 360 - difference)
    # We know that in frame coordination system A vector angle always precedes the C vector angle.
    if difference > 0:
        candidate.fp_a = candidate.fp_i
        candidate.fp_ac = candidate.fp_ic
        candidate.fp_c = candidate.fp_j
        candidate.fp_cc = candidate.fp_jc

        a1 = candidate.fp_i[index_i_opposite][0]
        a2 = i2
        c1 = candidate.fp_j[index_j_opposite][0]
        c2 = j2
    else:
        candidate.fp_a = candidate.fp_j
        candidate.fp_ac = candidate.fp_jc
        candidate.fp_c = candidate.fp_i
        candidate.fp_cc = candidate.fp_ic

        a1 = candidate.fp_j[index_j_opposite][0]
        a2 = j2
        c1 = candidate.fp_i[index_i_opposite][0]
        c2 = i2

    return tuple(a1), tuple(a2), tuple(c1), tuple(c2), tuple(d)


def pointsDistance(point_one: Coordinates, point_two: Coordinates) -> float:
    """Calculates distance between two points.

    :param point_one: coordinates of first point
    :param point_two: coordinates of second point
    :return: distance as float
    """

    points_distance = math.sqrt(abs((point_one[0]) - (point_two[0])) ** 2 +
                                abs((point_one[1]) - (point_two[1])) ** 2)

    return points_distance


def pointsToVec(point_one: Coordinates, point_two: Coordinates) -> Coordinates:
    """Return vector from two given points.

    :param point_one: coordinates of first point
    :param point_two: [x, y] coordinates of second point
    :return: [x, y] Resulting vector.
    """

    x_dif = point_two[0] - point_one[0]
    y_dif = point_two[1] - point_one[1]

    return x_dif, y_dif


def intersectionOfLines(a1: Coordinates, a2: Coordinates, c1: Coordinates, c2: Coordinates) -> Coordinates:
    """Find and return coordinates of line-line intersection.

    Finds intersection coordinates of two lines, each defined by two points.
    Written out equation obtained from Wikipedia: Line–line intersection.

    :param a1: coordinates of first point defining first line.
    :param a2: coordinates of second point defining first line.
    :param c1: coordinates of first point defining second line.
    :param c2: coordinates second point defining second line.
    :return: coordinates of intersection point.
    """

    # Written out determinants.
    numerator_1 = (a1[0] * a2[1] - a1[1] * a2[0]) * (c1[0] - c2[0]) - (a1[0] - a2[0]) * (c1[0] * c2[1] - c1[1] * c2[0])
    denumerator_1 = (a1[0] - a2[0]) * (c1[1] - c2[1]) - (a1[1] - a2[1]) * (c1[0] - c2[0])
    numerator_2 = (a1[0] * a2[1] - a1[1] * a2[0]) * (c1[1] - c2[1]) - (a1[1] - a2[1]) * (c1[0] * c2[1] - c1[1] * c2[0])
    denumerator_2 = (a1[0] - a2[0]) * (c1[1] - c2[1]) - (a1[1] - a2[1]) * (c1[0] - c2[0])
    # Avoiding zero division by adding 1e-5.
    intersection_x = numerator_1 / (denumerator_1 + 1e-5)
    intersection_y = numerator_2 / (denumerator_2 + 1e-5)
    intersection_x = int(intersection_x)
    intersection_y = int(intersection_y)

    return intersection_x, intersection_y


def drawAxes(frame: numpy.ndarray, qr_codes: List[Code], camera_matrix: numpy.ndarray,
             distortion_coefficients: numpy.ndarray) -> numpy.ndarray:
    """For each Code object in List draw id, size, corners and axes.

    Axes are drawn only if neither of tvec and rvec attributes is None.

    :param frame: frame as numpy ndarray (BGR colorspace)
    :param qr_codes: List of Code class objects
    :param camera_matrix: camera matrix of used camera obtained from camera calibration process
    :param distortion_coefficients: distortion coefficients obtained from camera calibration process
    :return: frame as numpy ndarray (BGR colorspace)
    """

    if not qr_codes:
        return frame

    for qr_code in qr_codes:
        text = "ID {}".format(qr_code.id)
        cv2.putText(frame, text, (qr_code.D[0] - 10, qr_code.D[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 75, 255), 2)
        text = "SIZE {}".format(qr_code.size)
        cv2.putText(frame, text, (qr_code.D[0] - 10, qr_code.D[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 75, 255), 2)

        cv2.circle(frame, qr_code.A, 3, (255, 0, 0), -1)
        cv2.circle(frame, qr_code.B, 3, (0, 255, 0), -1)
        cv2.circle(frame, qr_code.C, 3, (0, 0, 255), -1)
        cv2.circle(frame, qr_code.D, 3, (0, 255, 255), -1)

        if qr_code.tvec is not None:
            if qr_code.size is not None:
                if qr_code.size != 0:
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients,
                                      qr_code.rvec, qr_code.tvec, qr_code.size, 2)
                else:
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients,
                                      qr_code.rvec, qr_code.tvec, 1, 2)

    return frame


def drawCodes(frame: numpy.ndarray, qr_codes: List[Code]) -> numpy.ndarray:
    """For each Code object in List draw id, size, corners and its labels.

    :param frame: frame as numpy ndarray (BGR colorspace)
    :param qr_codes: List of Code class objects
    :return: frame as numpy ndarray (BGR colorspace)
    """

    if not qr_codes:
        return frame

    for qr_code in qr_codes:
        text = "ID {}".format(qr_code.id)
        cv2.putText(frame, text, (qr_code.D[0] - 10, qr_code.D[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 75, 255), 2)
        text = "SIZE {}".format(qr_code.size)
        cv2.putText(frame, text, (qr_code.D[0] - 10, qr_code.D[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 75, 255), 2)

        cv2.circle(frame, qr_code.A, 3, (255, 0, 0), -1)
        cv2.circle(frame, qr_code.B, 3, (0, 255, 0), -1)
        cv2.circle(frame, qr_code.C, 3, (0, 0, 255), -1)
        cv2.circle(frame, qr_code.D, 3, (0, 255, 255), -1)

        cv2.putText(frame, "A", (qr_code.A[0] + 25, qr_code.A[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
        cv2.putText(frame, "B", (qr_code.B[0] + 25, qr_code.B[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        cv2.putText(frame, "C", (qr_code.C[0] - 25, qr_code.C[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
        cv2.putText(frame, "D", (qr_code.D[0] - 25, qr_code.D[1] + 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)

    return frame


def drawCandidateEvaluation(frame: numpy.ndarray, qr_candidates: List[CodeCandidate]) -> numpy.ndarray:
    """For each CodeCandidate object in List draw finder pattern contours, testing contours and evaluation points.

    :param frame: frame as numpy ndarray (BGR colorspace)
    :param qr_candidates: List of CodeCandidate class objects
    :return: frame as numpy ndarray (BGR colorspace)
    """

    if not qr_candidates:
        return frame

    for candidate in qr_candidates:
        if (candidate.fp_a is not None and len(candidate.fp_a) != 0) and \
                (candidate.fp_c is not None and len(candidate.fp_c) != 0) and \
                (candidate.fp_d is not None and len(candidate.fp_d) != 0):
            cv2.drawContours(frame, [candidate.outline], -1, (120, 120, 255), 1)
            cv2.drawContours(frame, [candidate.fp_a], -1, (255, 100, 100), 1)
            cv2.drawContours(frame, [candidate.fp_c], -1, (100, 255, 100), 1)
            cv2.drawContours(frame, [candidate.fp_d], -1, (100, 100, 255), 1)

            cv2.line(frame, candidate.fp1c, candidate.fp2c, (180, 120, 120), 2)
            cv2.line(frame, candidate.fp0c, candidate.fp2c, (120, 180, 120), 2)
            cv2.line(frame, candidate.fp0c, candidate.fp1c, (120, 120, 180), 2)

            if candidate.index_i is not None or candidate.index_j is not None:
                point_i = tuple(candidate.fp_i[candidate.index_i][0])
                point_j = tuple(candidate.fp_j[candidate.index_j][0])

                cv2.circle(frame, point_i, 3, (0, 230, 255), -1)
                cv2.circle(frame, point_j, 3, (0, 255, 255), -1)
                if candidate.index_k is not None:
                    point_k = tuple(candidate.fp_d[candidate.index_k][0])
                    cv2.circle(frame, point_k, 3, (0, 110, 255), -1)

            cv2.putText(frame, "A", (candidate.fp_ac[0] + 25, candidate.fp_ac[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 100, 100), 2)
            cv2.putText(frame, "C", (candidate.fp_cc[0] + 25, candidate.fp_cc[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 230, 100), 2)
            cv2.putText(frame, "D", (candidate.fp_dc[0] + 25, candidate.fp_dc[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 230), 2)

    return frame
