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
	#rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)) 
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	ret= min(255,int(1.5*ret))
	#print(ret)
	x=bw.copy()
	y=bw.copy()
	# Appplying dilation on vertical lines
	rows = x.shape[0]
	kernel = np.ones((3,3),np.uint8)
	vertical_size = rows / 100
	vertical_size=int(vertical_size)

	# Create structure element for extracting vertical lines through morphology operations
	print("defining vertical lines...",flush=True)
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))

	# Apply morphology operations
	x = cv2.erode(x, verticalStructure)
	x = cv2.dilate(x, verticalStructure) 
	print("writing ...")
	cv2.imwrite('X.png',x)
	minLineLength = 200
	maxLineGap = 20
	print("extracting vertical lines...",flush=True)
	#ines=[]
	lines = cv2.HoughLinesP(x,1,np.pi/180,60,minLineLength,maxLineGap)
	#if lines!=[]:
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)

	#applying dilation to horizontal lines
	print("defining horizontal lines...",flush=True)
	cols = y.shape[1]
	horizontal_size = cols / 40
	horizontal_size=int(horizontal_size)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	y = cv2.erode(y, horizontalStructure)
	y = cv2.dilate(y, horizontalStructure) 
	print("writing ...",flush=True)
	cv2.imwrite('Y.png',y)

	print("extracting horizontal lines...",flush=True)
	minLineLength = 200
	maxLineGap = 20
	lines = cv2.HoughLinesP(y,1,np.pi/180,60,minLineLength,maxLineGap)
	#lines=[]
	#if lines != []:
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)

	print("writing ...",flush=True)
	cv2.imwrite('lines_removed.png',img)
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
	# Appplying dilation on the threshold image 
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	print("writing ...",flush=True)
	cv2.imwrite('dilated.png', dilation)
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	print("preprocessing done.",flush=True)
	return contours,hierarchy

#img = cv2.imread("/home/piyush/Pictures/invoice1.png") 
#img0 = img.copy()
#img1 = img.copy()
#remove_lines(img1)
#contours, hierarchy = ignore_lines(img0)