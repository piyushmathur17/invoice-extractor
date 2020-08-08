import cv2 
import numpy as np
class node:  
    def __init__(self, i,x1,x2,y1,y2):  
        self.i = i
        #x1,x2,y1,y2 
        self.x1 = x1  
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
       	#edges store the nodes this node is connected to
       	#they are integers denoting node number
        self.edges = []
        #parent of current node
        self.parent = -1
        #is node a key
        self.is_key = False
        #stores column number it is a part of
        self.column = -1
        #stores node present below it
        self.down = -1
        #stores node present to it's right
        self.right = -1
        #used in unused function
        self.parents = [] #stores all parent in path to current node

#check if two nodes are overlapping
def overlapping(x1,x2,y1,y2,a1,a2,b1,b2):
		if(x2<=a1 or x1>=a2): return False
		elif(y2<=b1 or y1>=b2):return False
		else : return True
#check if two nodes are verically above and below (if one node's 80% lies within other node's width)
#return value 0 : not overlapping
#return value 1 : overlapping but not 70%
#return value 2 : overlapping and >=70%
def x_overlapping(x1,x2,y1,y2,a1,a2,b1,b2):
		if(x2<=a1 or x1>=a2): return 0
		if(abs(x1-a1)>200):return 0
		else:
			l = max(x1,a1)
			r = min(x2,a2)
			if((r-l)/(x2-x1) >= 0.7):return 2
			if((r-l)/(a2-a1) >= 0.7):return 2
			return 1
			

#unused function		
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

#unused function
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
				if ( not( b<=x1 or a>=x2 ) ):
				#if abs(x1-a)<200 and Nodes[field].column == Nodes[row_node].column:
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

#look at this function carefully
def dfs(img,row,column,contours,Nodes,go_down,go_right,parent,root):
	#contours is a dict with key as row numbers (0,1,2,...,total rows-1)
	#eg. { 0:[0,1,2], 1:[3,4,5,6], 2:[7,8,9]} when image has 3 rows and 10 nodes
	#row is the key/row number
	#column is index of current node in list contours[row]
	#Nodes is a list of all node objects
	#go_down specifies whether to look for node below
	#go_right species whther to look for node to the right
	#parent is immediate parent node of current node (parent would be at the left or at top of current node)
	#root is the first node of the path of current node(should be a keyfield node)
	if(row>= len(contours) or column>= len(contours[row])):return
	current_node = contours[row][column]
	x1=Nodes[current_node].x1
	x2=Nodes[current_node].x2
	y1=Nodes[current_node].y1
	y2=Nodes[current_node].y2
	
	#look right if node is not last in row and next node is not a key
	if(go_right and column<len(contours[row])-1 and (not Nodes[contours[row][column+1]].is_key)):
		next_node = contours[row][column+1]
		a1=Nodes[next_node].x1
		b1=Nodes[next_node].y1
		cv2.line(img,(x1,y1),(a1,b1),(167,88,162),2)
		dfs(img,row,column+1,contours, Nodes, False, True,current_node,root)
	
	#look down if current row is not last row
	if(go_down and row<len(contours)-1 ):
		flag=0 
		#search for a node maximum 4 rows below
		#if any node is found recur dfs with that node and then break
		#if current node is not key but next node is key, break
		for next_row in range(row+1,min(row+5,len(contours))):
			if flag: break
			for i in range(0,len(contours[next_row])):
				if flag: break #means we've already met one node below
				a1=Nodes[contours[next_row][i]].x1
				a2=Nodes[contours[next_row][i]].x2
				b1=Nodes[contours[next_row][i]].y1
				b2=Nodes[contours[next_row][i]].y2

				overlap = x_overlapping(x1,x2,y1,y2,a1,a2,b1,b2)
				
				if overlap == 1: 
					flag=1
					break 
				elif overlap ==2 :
					if not Nodes[contours[next_row][i]].is_key:
						Nodes[current_node].down = contours[next_row][i]
						Nodes[contours[next_row][i]].parent = current_node
						Nodes[contours[next_row][i]].root = root
						print("printing line")
						cv2.line(img,(x1,y1),(a1,b1),(167,88,162),2)
						dfs(img,next_row,i,contours, Nodes, True ,False,current_node,root)
						flag=1
					else: 
						flag=1
						break

def assign_columns(img,columns, Nodes):
	node_columns = {}
	i=0

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
	node_to_row={}
	row_key = {}
	j=0

	for key in sorted(contours):
		contours_as_nodes[j]=[]
		row_key[j]=key 
		for Node in contours[key]:
			Nodes.append(node(i,Node[0],Node[0]+Node[2],Node[1],Node[1]+Node[3]))
			node_map[i] = Node
			contours_as_nodes[j].append(i)
			node_to_row[i] = j
			i+=1
		j+=1

	graph = np.zeros((i,i),dtype= bool )
	
	for Node in key_fields:
		Nodes[Node].is_key = True
	#for row in contours_as_nodes:
	#	for column in range(0,len(contours_as_nodes[row])):
	for keyfield in key_fields:
		row =  node_to_row[keyfield]
		for i in range(0,len(contours_as_nodes[row]) ):
			if contours_as_nodes[row][i] == keyfield: 
				column = i
				break
		root = contours_as_nodes[row][column]
		parent = contours_as_nodes[row][column]
		dfs(img,row,column,contours_as_nodes,Nodes,True,True,parent,root)
			#img = make_edges(img,key,i,contours_as_nodes,graph,node_map,key_fields,Nodes)
	#assign_columns(img,column_contours,Nodes)
	#for i in Nodes:
		#print(i.column)
	#make_columns(img,contours_as_nodes,node_map,graph,key_fields,Nodes,node_to_row)

	cv2.imwrite("graph.jpg",img )

