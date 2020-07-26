# Import required packages 
import cv2 
import pytesseract 
import numpy as np
from rem_lines import ignore_lines, remove_lines

#initializing parameters
cnt_thresh_x=0.8
cnt_thresh_y=0.8
# Mention the installed location of Tesseract-OCR in your system 
#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
print('hellow');
# Read image from which text needs to be extracted 
img = cv2.imread("/home/piyush/Pictures/invoice2.png") 
img0 = img.copy()
contours, hierarchy = ignore_lines(img0)

# Creating a copy of image 
im2 = img.copy() 
# A text file is created and flushed 
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 
rect = img

#combining contours on the bases of contour thresh x
contoursBBS = {}
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    cnt= [x,y,w,h]
    search_key= y
    #check if current contour is part of any existing row
    if contoursBBS: 
    	text_row = min(contoursBBS.keys(), key = lambda key: abs(key-search_key))
    	#if diff btw nearest row and y is greater than the threshold 
    	if(abs(text_row-y) > h*cnt_thresh_y):
    		contoursBBS[y]=[]
    		contoursBBS[y].append(cnt)
    	else :  contoursBBS[text_row].append(cnt)
    #else make new row
    else: contoursBBS[y]=[cnt]
#print(str(contoursBBS))
#sort contours
for row in contoursBBS:
	contoursBBS[row].sort(key = lambda x: x[0])
print(str(contoursBBS))
merge_cnt={}
i=0
for key in contoursBBS:
	j=1
	i=0
	de=[]
	merge_cnt[key]=[]
	[x1,y1,w1,h1]=contoursBBS[key][i]
	new_width = w1
	new_height = h1
	miny=y1
	#iterating through row to see if current contour can be merged with previous
	while j< len(contoursBBS[key]):

		[x2,y2,w2,h2]=contoursBBS[key][j]
		if( abs(y1-y2)<h1*cnt_thresh_y and abs(x1+new_width-x2) < h1*cnt_thresh_x and abs(new_height-h2)<h2*cnt_thresh_y):
			miny=min(miny,y2)
			new_width= x2-x1+w2
			new_height= max(new_height, y2+h2-miny)
			j+=1
			if j==len(contoursBBS[key]):
				merge_cnt[key].append([x1,miny,new_width,new_height])
		else:
			merge_cnt[key].append([x1,miny,new_width,new_height])
			i=j
			j+=1
			[x1,y1,w1,h1]=contoursBBS[key][i]
			new_width = w1
			new_height = h1
			miny=y1
			if j==len(contoursBBS[key]):
				merge_cnt[key].append(contoursBBS[key][j-1])
	if(len(contoursBBS[key])==1):merge_cnt[key].append(contoursBBS[key][0])
print("merged")
for i in sorted (merge_cnt) : 
    print ((i, merge_cnt[i]), end =" ") 

# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
# Open the file in append mode 
file = open("recognized.txt", "a")
for cnt in merge_cnt: 
	#print(cnt)
	
	for contour in merge_cnt[cnt] :
		[x, y, w, h] = contour
		# Drawing a rectangle on copied image 
		rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		
		# Cropping the text block for giving input to OCR 
		#offset= int (h*cnt_thresh_y)
		#y=y-offset
		#x=x-offset

		cropped = im2[y :y + h , x :x + w ]  
		
		# Apply OCR on the cropped image 
		text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 6')
		
		# Appending the text into file 
		file.write(text) 
		file.write("\n") 
		
# Close the file 
file.close 
cv2.imwrite('boxed.png', rect)
cv2.imshow('ho hey',rect)
cv2.waitKey(0)
cv2.destroyAllWindows()
