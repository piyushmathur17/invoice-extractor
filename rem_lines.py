import cv2 
import pytesseract 
import numpy as np

def remove_lines(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	laplacian = cv2.Laplacian(gray,cv2.CV_8UC1) # Laplacian Edge Detection
	minLineLength = 100
	maxLineGap = 40
	lines = cv2.HoughLinesP(laplacian,2,np.pi/180,60,minLineLength,maxLineGap)
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
	#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#laplacian = cv2.Laplacian(gray,cv2.CV_8UC1) # Laplacian Edge Detection	
	#lines = cv2.HoughLinesP(laplacian,2,np.pi/180,60,minLineLength,maxLineGap)
	#for line in lines:
	#    for x1,y1,x2,y2 in line:
	#        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
	cv2.imwrite('lines_removed0.png',img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)) 

	# Appplying dilation on the threshold image 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	cv2.imwrite('dilated0.png', dilation)
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	#return contours, hierarchy

def ignore_lines(img):
	#img= cv2.imread('lines_removed0.png')
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	hlinek = np.zeros((11,11),dtype=np.uint8)
	hlinek[...,5]=1
	vlinek = np.zeros((11,11),dtype=np.uint8)
	vlinek[5,...]=1
	#canny = cv2.Canny(gray, 50, 200, apertureSize=3)
	#canny = cv2.Laplacian(gray,cv2.CV_8UC1) # Laplacian Edge Detection
	#x=cv2.morphologyEx(canny, cv2.MORPH_OPEN, hlinek ,iterations=1)
	#y=cv2.morphologyEx(canny, cv2.MORPH_OPEN, vlinek ,iterations=1)
	#rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)) 
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	ret= min(255,int(1.5*ret))
	print(ret)
	cv2.imshow('dnkl',bw)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	x=bw.copy()
	y=bw.copy()
	# Appplying dilation on verticle lines
	rows = x.shape[0]
	kernel = np.ones((3,3),np.uint8)
	verticle_size = rows / 60
	verticle_size=int(verticle_size)

	# Create structure element for extracting verticle lines through morphology operations
	verticleStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticle_size))

	# Apply morphology operations
	x = cv2.erode(x, verticleStructure)
	x = cv2.dilate(x, verticleStructure) 

	cv2.imwrite('X.png',x)
	minLineLength = 200
	maxLineGap = 20
	lines = cv2.HoughLinesP(x,1,np.pi/180,60,minLineLength,maxLineGap)
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)

	#applying dilation to horizontal lines

	cols = y.shape[1]
	horizontal_size = cols / 40
	horizontal_size=int(horizontal_size)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	y = cv2.erode(y, horizontalStructure)
	y = cv2.dilate(y, horizontalStructure) 
	cv2.imwrite('Y.png',y)
	minLineLength = 200
	maxLineGap = 20
	lines = cv2.HoughLinesP(y,1,np.pi/180,60,minLineLength,maxLineGap)
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)


	cv2.imwrite('lines_removed.png',img)
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
	# Appplying dilation on the threshold image 
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	cv2.imwrite('dilated.png', dilation)
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	return contours,hierarchy

#img = cv2.imread("/home/piyush/Pictures/invoice1.png") 
#img0 = img.copy()
#img1 = img.copy()
#remove_lines(img1)
#contours, hierarchy = ignore_lines(img0)