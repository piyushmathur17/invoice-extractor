import cv2 
import numpy as np
from graph import make_graph
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


def stretch_columns(img):
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
	img = cv2.erode(img, structure,iterations=1) 
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
	x = cv2.dilate(img, structure,iterations=2) 
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
	x = cv2.dilate(x, structure,iterations=1) 
	contours, hierarchy = cv2.findContours(x, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)

	cv2.imwrite("columnstretched.jpg",x)
	return x

def segment_columns(img,shape,contours):
	mask = np.zeros((shape[0],shape[1]),np.uint8)
	for key in contours:
		for i in contours[key]:
			[x,y,w,h]= i
			#ones = np.ones((h,w),np.uint8) * 255
			mask = cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1) 
			print(y," ",y+h," ",x," ",x+w," is 255")

	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (4,1))
	mask = cv2.erode(mask, structure,iterations=1)
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
	mask = cv2.dilate(mask, structure,iterations=2)
	cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	for c in cont:
		[x,y,w,h]= cv2.boundingRect(c)
		#ones = np.ones((h,w),np.uint8) * 255
		img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,255), 3) 

	cv2.imwrite("blocks.jpg",img)
	return cont
def ignore_lines(img,save_dir,file_name):

	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	ret= min(255,int(1.5*ret))
	
	#after thresholding, to connect pixels in weak images
	structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
	x = cv2.dilate(bw, structure,iterations=1) 
	#structure = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
	#y = cv2.dilate(bw, structure,iterations=1) 
	y=bw

	# Applying dilation on vertical lines
	rows = x.shape[0]
	kernel = np.ones((3,3),np.uint8)
	vertical_size = rows / 80
	vertical_size=int(vertical_size)

	# Create structure element for extracting vertical lines through morphology operations
	print("defining vertical lines...",flush=True)
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))

	# Apply morphology operations
	x = cv2.erode(x, verticalStructure)
	x = cv2.dilate(x, verticalStructure) 
	cv2.imwrite(save_dir+ 'X_' + file_name,x)

	#applying mask of vertical lines to image
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
	x = cv2.dilate(x, rect_kernel, iterations = 1)

	#minLineLength = 200
	#maxLineGap = 20
	#print("extracting vertical lines...",flush=True)
	#ines=[]palta ki cgpa kitni h 8.07
	#lines = cv2.HoughLinesP(x,1,np.pi/180,60,minLineLength,maxLineGap)
	#garb=0
	#if lines is not None: 
	##else:
	#	for line in lines: 
	#	    for x1,y1,x2,y2 in line:
	#	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)


	#applying dilation to horizontal lines
	print("defining horizontal lines...",flush=True)
	cols = y.shape[1]
	horizontal_size = cols / 90
	horizontal_size=int(horizontal_size)

	# Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

	# Apply morphology operations
	y = cv2.erode(y, horizontalStructure)
	y = cv2.dilate(y, horizontalStructure) 

	# Applying dilation on horizontal lines
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  
	y = cv2.dilate(y, rect_kernel, iterations = 1)

	cv2.imwrite(save_dir+'Y_' + file_name,y)

	#applying vertical and horizontal line masks
	ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU ) 
	bw = bw + x + y


	#print("extracting horizontal lines...",flush=True)
	#minLineLength = 200
	#maxLineGap = 20
	#lines = cv2.HoughLinesP(y,1,np.pi/180,60,minLineLength,maxLineGap)
	#lines=None
	#if lines is not None:
	#	for line in lines:
	#	    for x1,y1,x2,y2 in line:
	#	        cv2.line(img,(x1,y1),(x2,y2),(ret,ret,ret),3)

	cv2.imwrite(save_dir+'lines_removed_'+file_name,bw)

	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 
	
	# Appplying dilation on the threshold image 
	ret, thresh1 = cv2.threshold(bw, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
	cv2.imwrite(save_dir+'dilated_'+file_name, dilation)
	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_SIMPLE)
	column_dilation = stretch_columns(dilation)
	print("preprocessing done.",flush=True)
	return contours,hierarchy, bw

#img = cv2.imread("/home/piyush/Pictures/invoice1.png") 
#img0 = img.copy()
#img1 = img.copy()
#remove_lines(img1)
#contours, hierarchy = ignore_lines(img0)