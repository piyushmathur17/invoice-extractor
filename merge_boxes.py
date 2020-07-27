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
	min_height = height_list[int(len(height_list)/2)]*0.6
	print("min_height: ",min_height)
	for contour in contours:
	    [x, y, w, h] = cv2.boundingRect(contour)
	    if h< min_height : continue
	    cnt= [x,y,w,h]
	    search_key= y
	    #check if current contour is part of any existing row
	    if contoursBBS: 
	    	text_row = min(contoursBBS.keys(), key = lambda key: abs(key-search_key))
	    	#if diff btw nearest row and y is greater than the threshhold 
	    	if(abs(text_row-y) > h*thresh_y):
	    		contoursBBS[y]=[]
	    		contoursBBS[y].append(cnt)
	    	else :  contoursBBS[text_row].append(cnt)
	    #else make new row
	    else: contoursBBS[y]=[cnt]
	
	#sort contours
	for row in contoursBBS:
		contoursBBS[row].sort(key = lambda x: x[0])
	
	return contoursBBS

#a utility function to merge two words based on their nearness 
def merge_boxes(contoursBBS, thresh_x = 0.3, thresh_y = 0.3):
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
			if( abs(y1-y2)<h1*thresh_y and abs(x1+new_width-x2) < h1*thresh_x and abs(new_height-h2)<h2*thresh_y):
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