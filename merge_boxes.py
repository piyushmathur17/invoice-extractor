import numpy as np
import cv2


#a utility function to assign each word a row 
def make_rows(contours, thresh_y = 0.6):
	contoursBBS = {}
	height_list=[]
	for contour in contours:
	    [x, y, w, h] = cv2.boundingRect(contour)
	    height_list.append(h)
	height_list.sort()
	#contours with height less than min_height will be discarded
	min_height = height_list[int(len(height_list)/2)]*0.6
	print("min_height: ",min_height)
	#finding suitable line height
	alpha = int(len(height_list)*0.3)
	line_height = 1.2*sum(height_list[alpha:len(height_list)-alpha])/(len(height_list)-2*alpha)

	for contour in contours:
	    [x, y, w, h] = cv2.boundingRect(contour)
	    if h< min_height : continue
	    cnt= [x,y,w,h]
	    search_key= y
	    #check if current contour is part of any existing row
	    if contoursBBS: 
	    	text_row = min(contoursBBS.keys(), key = lambda key: abs(key-search_key))
	    	#if diff btw nearest row and y is greater than the threshhold 
	    	#if(abs(text_row-y) > h*thresh_y):
	    	if(abs(text_row-y) > line_height):
	    		contoursBBS[y]=[]
	    		contoursBBS[y].append(cnt)
	    	else :  contoursBBS[text_row].append(cnt)
	    #else make new row
	    else: contoursBBS[y]=[cnt]
	
	#sort contours
	for row in contoursBBS:
		contoursBBS[row].sort(key = lambda x: x[0])
	
	return contoursBBS

def detect_line(rect,x1,x2,y1,y2,w1,w2,h1,h2):
	x1=x1+w1+1
	y=int((y1+h1)/2 + (y2+h2)/2)
	pos_edge=0
	neg_edge=0
	for i in range(x1,x2):
		if (int(rect[y][i][0])+int(rect[y][i][1])+int(rect[y][i][2]) - int(rect[y][i-2][0])-int(rect[y][i-2][1])-int(rect[y][i-2][2]))/2 >= 80 : pos_edge=1
		if (int(rect[y][i][0])+int(rect[y][i][1])+int(rect[y][i][2]) - int(rect[y][i-2][0])-int(rect[y][i-2][1])-int(rect[y][i-2][2]) ) /2  <= -80 : neg_edge=1
		if(pos_edge and neg_edge): 
			print("line detected between ",x1+w1," ",x2)
			return True
	return False
#a utility function to merge two words based on their nearness 
def merge_boxes(rect, contoursBBS, thresh_x = 0.3, thresh_y = 0.3):
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
			if( abs(y1-y2)<h1*thresh_y and abs(x1+new_width-x2) < h1*thresh_x and abs(new_height-h2)<h2*thresh_y and not(detect_line(rect,x1,x2,miny,y2, new_width,-1,new_height,h2)and detect_line(rect,x1,x2,miny,y2, new_width,-1,int(new_height/2),int(h2/2)) ) ):
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
	#print("merged")
	#for i in sorted (merge_cnt) : 
	#    print ((i, merge_cnt[i]), end =" ") 

	return merge_cnt