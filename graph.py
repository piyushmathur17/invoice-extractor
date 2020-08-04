import cv2 
import numpy as np

def make_edges(img,key,pos,contours,graph,node_map,key_fields):
	print("making edges")
	parent = contours[key][pos]
	i=pos+1
	#creating edges between parent and nodes to the right
	while not(i>=len(contours[key]) or ( contours[key][i] in key_fields) ):
		graph[parent][contours[key][i]] = True
		graph[contours[key][i]][parent] = True 
		x1= node_map[parent][0]
		x2= node_map[contours[key][i]][0]
		y1= node_map[parent][1]
		y2= node_map[contours[key][i]][1]
		cv2.line(img,(x1,y1),(x2,y2),(100,100,100),2)
		i+=1
	#make edges between parent and nodes down below
	return img

def make_graph(img,contours,key_fields):

	#indexing countours with node numbers
	i=0
	node_map={}
	contours_as_nodes={}
	for key in contours:
		contours_as_nodes[key]=[]
		for node in contours[key]:
			node_map[i] = node
			contours_as_nodes[key].append(i)
			i+=1

	graph = np.zeros((i,i),dtype= bool )
	for key in contours:
		i=0
		for node in contours_as_nodes[key]:
			if node in key_fields:
				img = make_edges(img,key,i,contours_as_nodes,graph,node_map,key_fields)
	cv2.imwrite("graph.jpg",img )

