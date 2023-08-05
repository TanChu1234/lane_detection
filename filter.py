import cv2
import numpy as np
import time
import math
import sys
def processImage(inpImage):

    # Apply HLS color filtering to filter out white lane lines
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    
    lower_white = np.array([0, 200,200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(inpImage, lower_white, upper_white)
    
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)
    # cv2.imshow("m", mask)
    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh,(5, 5), 0)
    # cv2.imshow("m", mask)
    
    canny = cv2.Canny(blur, 50, 150)

    return canny
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, vertices1):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # cv2.fillPoly(mask, vertices1, 0)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def distance_point_to_line(point, line_point1, line_point2):
    x0, y0 = point
    x1, y1 = line_point1
    x2, y2 = line_point2

    # Calculate the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = (x2 * y1) - (x1 * y2)

    # Calculate the distance
    distance = abs((A * x0) + (B * y0) + C) / math.sqrt((A * A) + (B * B))
    return distance

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # In case of error, don't draw the line
    draw_right = False
    draw_left = True
    
    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slope_threshold = 0.3
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
        
        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
            slope = 999.  # practically infinite slope
        else:
            slope = (y2 - y1) / (x2 - x1)
            
        # Filter lines based on slope
        if abs(slope) > slope_threshold:
            slopes.append(slope)
            new_lines.append(line)
        
    lines = new_lines
    
    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        img_x_center = img.shape[1] / 2  # x coordinate of center of image
        if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
            right_lines.append(line)
        elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
            left_lines.append(line)
            
    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []
    
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    if len(right_lines_x) > 0:
        right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
        right_m, right_b = 1, 1
        draw_right = False
        
    # Left lane lines
    left_lines_x = []
    left_lines_y = []
    
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        
        left_lines_y.append(y1)
        left_lines_y.append(y2)
        
    if len(left_lines_x) > 0:
        left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
        left_m, left_b = 1, 1
        draw_left = False
    
    # Find 2 end points for right and left lines, used for drawing the line
    # y = m*x + b --> x = (y - b)/m
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - 0.8)
    
    right_x1 = (y1 - right_b) / right_m
    right_x2 = (y2 - right_b) / right_m
    
    left_x1 = (y1 - left_b) / left_m
    left_x2 = (y2 - left_b) / left_m
    
    # Convert calculated end points from float to int
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)
    
    # Draw the right and left lines on image
    if draw_right:
        cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
        
    if draw_left:
        cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)
        # distance = distance_point_to_line((540, 720), (left_x1, y1), (left_x2, y2))
    # return distance    


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # if lines is None: 
    #     return
    #line_img = np.zeros(img.shape, dtype=np.uint8)  # this produces single-channel (grayscale) image
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
    
    # distance = draw_lines(line_img, lines)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 100    # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 50  # maximum gap in pixels between connectable line segments
video = cv2.VideoCapture('output.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
fps /= 100
framerate = time.time()
elapsed = int()
while True:
    # img = cv2.resize(cv2.imread("D:\Lane Detection\lane_detection-master\lane_detection-master\img\pic" + str(i)+".jpg"), (1080,720))
    # img = cv2.resize(cv2.imread(r"D:\Lane Detection\lane_detection-master\lane_detection-master\img\test4.jpg"), (1080,720))
    start = time.time()
    _, frame = video.read()
    img = cv2.resize(frame, (640,480))
    ########################################################################################
    lower_white = np.array([0, 100, 100])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)
    denoised = cv2.medianBlur(mask, 5)
    edges = cv2.Canny(denoised, 50, 150)
    # edges = processImage(denoised)
    imshape = img.shape
    vertices = np.array([[\
        (0, imshape[0]),\
        (0, imshape[0]/2 + imshape[0] * 0.1),\
        (imshape[1]/2 - imshape[1] * 0.1, imshape[0]/2 - imshape[0] * 0.2),\
        (imshape[1]/2 + imshape[1] * 0.1, imshape[0]/2 - imshape[0] * 0.2),\
        (imshape[1], imshape[0]/2 + imshape[0] * 0.1),\
        ((imshape[1], imshape[0]))]]\
        , dtype=np.int32)


    vertices1 = np.array([[\
            (imshape[1] * 0.2, imshape[0]),\
            (imshape[1] * 0.4, imshape[0]/2),\
            (imshape[1] * 0.6, imshape[0]/2),\
            (imshape[1] * 0.8, imshape[0])]]\
            , dtype=np.int32)
    # print(vertices)
    masked_edges = region_of_interest(edges, vertices, vertices1)
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # print(deviation/8.25)
    # Draw lane lines on the original image
    # initial_image = masked_edges.astype('uint8')
    # print(line_image.shape)
    annotated_image = weighted_img(line_image, img)
    cv2.imshow('window', masked_edges)

    cv2.imshow('Denoised', annotated_image)

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
