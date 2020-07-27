# Import required packages 
import cv2 
import pytesseract 
import numpy as np
from rem_lines import ignore_lines, remove_lines
from merge_boxes import make_rows, merge_boxes
import time

def main():
	# Mention the installed location of Tesseract-OCR in your system 
	#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
	print('hellow');
	# Read image from which text needs to be extracted 
	img = cv2.imread("/home/piyush/Pictures/invoice2.png") 
	contours, hierarchy = ignore_lines(img)
	cv2.imwrite('savedee.png',img)
	# Creating a copy of image 
	rect = img.copy() 
	# A text file is created and flushed 
	file = open("recognized.txt", "w+") 
	file.write("") 
	file.close() 

	#assigning rows to contours
	contoursBBS = make_rows(contours)

	#combining contours on the bases of contour thresh x
	merge_cnt = merge_boxes(contoursBBS, thresh_x = 0.7, thresh_y = 0.6)

	print("recognizing text",flush=True)
	# Looping through the identified contours 
	# Then rectangular part is cropped and passed on 
	# to pytesseract for extracting text from it 
	# Extracted text is then written into the text file 
	# Open the file in append mode 
	croptime=0
	tesstime=0
	tt=time.time()
	file = open("recognized.txt", "a")

	for cnt in merge_cnt: 
		#print(cnt)
		cv2.line(rect,(0,cnt),(len(rect[0])-1,cnt),(255,255,0),2)
		for contour in merge_cnt[cnt] :
			[x, y, w, h] = contour
			if h<10 : continue
			# Drawing a rectangle on copied image 
			rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2) 
			
			# Cropping the text block for giving input to OCR 
			#offset= int (h*cnt_thresh_y)
			#y=y-offset
			#x=x-offset
			start = time.time()
			cropped = img[y :y + h , x :x + w ]  
			end = time.time()
			croptime+= end-start
			# Apply OCR on the cropped image 

			text = ""
			text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 7')
			end2 = time.time()
			tesstime += end2-end
			# Appending the text into file 
			if text!="":
				file.write(text) 
				file.write("\n") 
	tt = time.time()- tt
	print("croptime: ",croptime, "  tesstime: ",tesstime,"   tt: ",tt)
	# Close the file 
	file.close 
	cv2.imwrite('boxed.png', rect)
	#cv2.imshow('ho hey',rect)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
if __name__ == '__main__':
	main()
