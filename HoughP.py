import cv2
import numpy as np
import time
import sys

def average_slope_intercept(lines):
    left_lines = []  # to store lines on the left side of the image
    left_weights = []  # to store the lengths of the corresponding lines
    right_lines = []  # to store lines on the right side of the image
    right_weights = []  # to store the lengths of the corresponding lines
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:  # avoid dividing by zero
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        # Filter out lines with extreme slopes or lengths
        if abs(slope) < 0.5 or length < 50:
            continue
        
        if slope <= 0:  # left lane
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:  # right lane
            right_lines.append((slope, intercept))
            right_weights.append(length)
    
    # Calculate weighted average of slopes and intercepts for left and right lanes
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane

def get_lane_points(y1, y2, line):
    if line is None:
        return None
    
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return (x1, y1, x2, y2)

def hough(img):
    lower_white = np.array([200, 230, 230])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    white_image = cv2.bitwise_and(img, img, mask = mask)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90,100,100])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(img, img, mask=yellow_mask)
    
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    
    gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    height, width = edges.shape
    roi_mask = np.zeros_like(edges)
    vertices = np.array([[\
        (0, height),\
        (0, height/2 + height * 0.1),\
        (width/2 - width * 0.1, height/2 - height * 0.2),\
        (width/2 + width * 0.1, height/2 - height * 0.2),\
        (width, height/2 + height * 0.1),\
        ((width, height))]]\
        , dtype=np.int32)
    polygon = np.array([[(0, height), (width/2, height/2), (width, height)]], np.int32)
    cv2.fillPoly(roi_mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask) 
    cv2.imshow("Mask", masked_edges)
    # distance resolution in pixels of the Hough grid
    # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    #minimum number of pixels making up a line
    # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=10)
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = height
    y2 = height/2 + 100
    left_lane_points = get_lane_points(y1, y2, left_lane)
    right_lane_points = get_lane_points(y1, y2, right_lane)

    # Draw the lane lines on the original image
    line_image = np.zeros_like(frame)
    if left_lane_points is not None:
        cv2.line(line_image, (left_lane_points[0], left_lane_points[1]), (left_lane_points[2], left_lane_points[3]), (0, 255, 0), thickness=5)
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result
video = cv2.VideoCapture('output.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
fps /= 200
framerate = time.time()
elapsed = int()

while True:
    # img = cv2.resize(cv2.imread("D:\Lane Detection\lane_detection-master\lane_detection-master\img\pic" + str(i)+".jpg"), (1080,720))
    # img = cv2.resize(cv2.imread(r"D:\Lane Detection\lane_detection-master\lane_detection-master\img\test4.jpg"), (1080,720))
    start = time.time()
    _, frame = video.read()
    

    result = hough(frame)

    cv2.imshow("Lane Detection", result)
 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    diff = time.time() - start
    while  diff < fps:
        diff = time.time() - start

    elapsed += 1
    if elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (time.time() - framerate)))
        sys.stdout.flush()

    # Wait for a key press and close the windows
    # cv2.waitKey(0)
time.time()
cv2.destroyAllWindows()
