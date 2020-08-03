import cv2 
import numpy as np
import math


def rotate_image(image, angle):
    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def getAngle(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
	ret = min(255, int(1.5*ret))
	x = bw.copy()
	y = bw.copy()

	# Appplying dilation on vertical lines
	rows = x.shape[0]
	kernel = np.ones((3, 3), np.uint8)
	vertical_size = rows / 100
	vertical_size = int(vertical_size)

	# Create structure element for extracting vertical lines through morphology operations
	print("defining vertical lines...", flush=True)
	verticalStructure = cv2.getStructuringElement(
		cv2.MORPH_RECT, (1, vertical_size))

	# Apply morphology operations
	x = cv2.erode(x, verticalStructure)
	x = cv2.dilate(x, verticalStructure)
	print("writing ...")
	cv2.imwrite('X.png', x)
	minLineLength = 200
	maxLineGap = 20
	print("extracting vertical lines...", flush=True)
	#ines=[]
	lines = cv2.HoughLinesP(x, 1, np.pi/180, 60, minLineLength, maxLineGap)
	print(lines)
	angle = 0.0
	val = 0
    #if lines!=[]:
	for line in lines:
	    for x1, y1, x2, y2 in line:
	        if(y1<y2):
	            x1,x2 = x2,x1
	            y1,y2 = y2,y1
	        lineAngle = ((math.atan2(y1 - y2, x1 - x2))*180/ 3.14159265)
	        # print(lineAngle)
	        val += abs(y1-y2)
	        angle += (abs(y1-y2)*(lineAngle))

	#applying dilation to horizontal lines
    
	print("defining horizontal lines...", flush=True)
	print("Angle :")
	print(angle/val)
	rotateAngle = angle/val - 90
	cols = y.shape[1]
	horizontal_size = cols / 40
	horizontal_size = int(horizontal_size)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(
		cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	y = cv2.erode(y, horizontalStructure)
	y = cv2.dilate(y, horizontalStructure)
	print("writing ...", flush=True)
	cv2.imwrite('Y.png', y)

	print("extracting horizontal lines...", flush=True)
	minLineLength = 200
	maxLineGap = 20
	lines = cv2.HoughLinesP(y, 1, np.pi/180, 60, minLineLength, maxLineGap)
	print(lines)
    #lines=[]
	#if lines != []:
	angle = 0.0
	val = 0

	for line in lines:
	    for x1, y1, x2, y2 in line:
	        if(x1<x2):
	            x1,x2 = x2,x1
	            y1,y2 = y2,y1
	        lineAngle = ((math.atan2(y1 - y2, x1 - x2))*180/ 3.14159265)
	        print(lineAngle)
	        if(lineAngle):
	            print(line)
	        val += abs(y1-y2)
	        angle += (abs(y1-y2)*(lineAngle))

	print("Angle :")
	rotateAngle += angle/val
	print(angle/val)

	return rotateAngle/2

   
filename = 'F:/Flipkart/invoice-extractor/invoice_images/Sample9/1.jpg'
img = cv2.imread(filename)
angle = getAngle(img)
rotatedImg = rotate_image(img, -angle)
cv2.imwrite('rotated.png', rotatedImg)