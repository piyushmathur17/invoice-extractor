# Import required packages 
from os import listdir
import cv2 
import pytesseract 
import numpy as np
from rem_lines import ignore_lines, remove_lines, segment_columns
from merge_boxes import make_rows, merge_boxes
from graph import make_graph
#from correctPerspective import getAngle, rotate_image
import time
keys = ['supplier','taxable','item','freight','shipping','address','Discount','info','amt','amount','vehicle','bill','details','state','payment','insurance','charges','tax', 'value', 'dispatch', 'dispatched','seller', 'buyer', 'name', 'id', 'no.', 'number', 'gst', 'date', 'percent', 'invoice', 'total', 'cost', 'price', 'rate', 'description','article', 'quantity','amount', 'hsn','sl']

def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) 
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return distance[row][col]



def get_text(save_dir,file_name, write_ = False):
	read_dir = save_dir + file_name
	# Mention the installed location of Tesseract-OCR in your system 
	#pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
	print(read_dir)
	# Read image from which text needs to be extracted 
	img = cv2.imread(read_dir) 
	rect= img
	img2 = img
	#remove lines and form contours
	contours, hierarchy, img = ignore_lines(img,save_dir,file_name)
	# Creating a copy of image 
	#rect = img2.copy() 
	# A text file is created and flushed 
	file = open("recognized.txt", "a") 
	file_ = open(save_dir+"recognized.txt","a")
	file.write("")
	file_.write("")

	#assigning rows to contours
	contoursBBS = make_rows(contours)

	#combining contours on the bases of contour thresh x
	merge_cnt = merge_boxes(rect, contoursBBS, thresh_x = 1.0, thresh_y = 0.6)
	column_contours = segment_columns(img2,img.shape,merge_cnt)

	print("recognizing text",flush=True)

	#make_graph(rect,merge_cnt,[1,3,9,21,14,37,24,29,35,74,65,43,45,12,56],column_contours)
	#return
	# Looping through the identified contours 
	# Then rectangular part is cropped and passed on 
	# to pytesseract for extracting text from it 
	# Extracted text is then written into the text file 
	# Open the file in append mode 
	croptime=0
	tesstime=0
	tt=time.time()
	key_nodes=[]
	node_number=0
	node_columns={}

	for cnt in sorted(merge_cnt): 
		#print(cnt)
		#cv2.line(rect,(0,cnt),(len(rect[0])-1,cnt),(255,0,0),2)
		for contour in merge_cnt[cnt] :
			node_number+=1
			[x, y, w, h] = contour
			if h<10 : continue
			# Drawing a rectangle on copied image 
			rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 1) 
			
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
			p=text.split(' ')
			for tex in p:
				tex=tex.lower()
				if(tex in keys):
					rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
					key_nodes.append(node_number-1)
					break
				else:
					for k in keys:
						#print(k + " " + tex)
						if(k in tex):
							rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
							key_nodes.append(node_number-1)
							break
						if(len(text)>=1 and levenshtein_ratio_and_distance(k,tex)>0.8 ): 			
							rect = cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 0, 255), 1)
							key_nodes.append(node_number-1)
							break

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


	make_graph(rect,merge_cnt,key_nodes,column_contours)
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
		if(folder!="Sample3"): continue
		for image in images:
			if len(image.split('.')[0])>1:continue
			file_name = image
			#file_path = dir_path
			#mkdir(target)
			get_text(dir_path,file_name,write_ = False)
			
if __name__ == '__main__':
	main()
