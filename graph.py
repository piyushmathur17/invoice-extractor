import cv2 
import numpy as np
class node:  
    def __init__(self, i,x1,x2,y1,y2):  
        self.i = i
        self.x1 = x1  
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.edges = []
        self.parents = []
        self.is_key = False
        self.column = -1

def make_edges(img,key,pos,contours,graph,node_map,key_fields,Nodes):
	print("making edges")
	parent = contours[key][pos]
	i=pos+1
	#creating edges between parent and nodes to the right
	while not(i>=len(contours[key]) or ( contours[key][i] in key_fields) ):
		graph[parent][contours[key][i]] = True
		graph[contours[key][i]][parent] = True 
		Nodes[parent].edges.append(contours[key][i])
		Nodes[contours[key][i]].parents.append(parent)
		x1= node_map[parent][0]
		x2= node_map[contours[key][i]][0]
		y1= node_map[parent][1]
		y2= node_map[contours[key][i]][1]
		cv2.line(img,(x1,y1),(x2,y2),(204,100,0),2)
		i+=1
	#make edges between parent and nodes down below
	return img
def make_columns(img,contours,node_map,graph,key_fields,Nodes,node_to_key):
	for field in key_fields:
		x1 = Nodes[field].x1
		x2 = Nodes[field].x2
		Nodes[field].is_key = True
		for row in contours:
			if  row <= node_to_key[field]: continue
			for row_node in contours[row]: 
				a = node_map[row_node][0]
				b = node_map[row_node][0] + a
				#if ( (a<=x1 and b>=x1) or (b>=x2 and a<=x2) ):
				if Nodes[field].column == Nodes[row_node].column:
					#find node in Nodes corresponding to current element row
					graph[field][row_node] = True
					graph[row_node][field] = True 
					Nodes[field].edges.append(row_node)
					Nodes[row_node].parents.append(field)
					a1= node_map[field][0]
					a2= node_map[row_node][0]
					b1= node_map[field][1]
					b2= node_map[row_node][1]
					cv2.line(img,(a1,b1),(a2,b2),(167,88,162),2)

def assign_columns(img,columns, Nodes):
	node_columns = {}
	i=0

	def overlapping(x1,x2,y1,y2,a1,a2,b1,b2):
		if(x2<=a1 or x1>=a2):
			print(a1,a2,b1,b2,x1,x2,y1,y2)
			print(False)
			return False
		elif(y2<=b1 or y1>=b2):return False
		else : return True

	for i in range(0,len(Nodes) ):
		x1=Nodes[i].x1
		x2=Nodes[i].x2
		y1=Nodes[i].y1
		y2=Nodes[i].y2
		for j in range(0,len(columns) ):
			[a1,b1,w,h] = cv2.boundingRect(columns[j])
			a2 = a1+w
			b2 = b1+h
			if ( overlapping(x1,x2,y1,y2,a1,a2,b1,b2) ):
				Nodes[i].column = j
				#cv2.line(img,(x1,y1),(a1,b1),(204,100,0),2)
				#print(a1,a2,b1,b2,x1,x2,y1,y2)
				break
			

def make_graph(img,contours,key_fields,column_contours):

	#indexing countours with node numbers
	i=0
	node_map={}
	contours_as_nodes={}
	Nodes = []
	node_to_key={}
	for key in contours:
		contours_as_nodes[key]=[]
		for Node in contours[key]:
			Nodes.append(node(i,Node[0],Node[0]+Node[2],Node[1],Node[1]+Node[3]))
			node_map[i] = Node
			contours_as_nodes[key].append(i)
			node_to_key[i] = key
			i+=1

	graph = np.zeros((i,i),dtype= bool )
	for key in contours:
		i=0
		for Node in contours_as_nodes[key]:
			if Node in key_fields:
				img = make_edges(img,key,i,contours_as_nodes,graph,node_map,key_fields,Nodes)
	assign_columns(img,column_contours,Nodes)
	for i in Nodes:
		print(i.column)
	make_columns(img,contours_as_nodes,node_map,graph,key_fields,Nodes,node_to_key)

	cv2.imwrite("graph.jpg",img )

