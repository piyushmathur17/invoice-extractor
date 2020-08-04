# Import required packages 
from os import listdir
import cv2 
import pytesseract 
import numpy as np
from rem_lines import ignore_lines, remove_lines
from merge_boxes import make_rows, merge_boxes
from graph import make_graph
#from correctPerspective import getAngle, rotate_image
import time

def get_text(save_dir,file_name, write_ = False):
	read_dir = save_dir + file_name
	# Mention the installed location of Tesseract-OCR in your system 
	#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
	print(read_dir)
	# Read image from which text needs to be extracted 
	img = cv2.imread(read_dir) 
	#remove lines and form contours
	contours, hierarchy, img = ignore_lines(img,save_dir,file_name)
	# Creating a copy of image 
	rect = img.copy() 
	# A text file is created and flushed 
	file = open("recognized.txt", "a") 
	file_ = open(save_dir+"recognized.txt","a")
	file.write("")
	file_.write("")

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
			text = pytesseract.image_to_string(cropped, lang='eng', config='--psm 6')
			end2 = time.time()
			tesstime += end2-end
			# Appending the text into file 
			if text!="":
				if write_: file.write(text)
				file_.write(text)
				if write_: file_.write("\n") 
				file.write("\n") 

	tt = time.time()- tt
	print("croptime: ",croptime, "  tesstime: ",tesstime,"   tt: ",tt)
	# Close the file 
	file.close 
	cv2.imwrite(save_dir+'boxed_'+file_name, rect)


	make_graph(rect,merge_cnt,[5,14,19,23,17,24])
	#cv2.imshow('ho hey',rect)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def main():
	
	path = "/home/piyush/mygit/invoice-extractor/invoice_images/"

	folders = listdir(path)
	#print(pdfs)
	for folder in folders:
		dir_path = path + folder + "/"
		images = listdir(dir_path)
		if(folder!="Sample15"): continue
		for image in images:
			if len(image.split('.')[0])>1:continue
			file_name = image
			#file_path = dir_path
			#mkdir(target)
			get_text(dir_path,file_name,write_ = False)
			
if __name__ == '__main__':
	main()
