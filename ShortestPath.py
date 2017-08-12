import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import os
import datetime
from time import gmtime, strftime
from itertools import islice


#The distance function
def haversine(lat1, lon1, OD):
    

    # Calculate the great circle distance between two points
    # on the earth (specified in decimal degrees)

    km_Min=1000000;
    lat_1=float(lat1);lon_1=float(lon1)
    for key in OD:
        
       
         (lat,lon)=OD[key]
         lat2=float(lat);lon2=float(lon);
    # convert decimal degrees to radians
         lon1, lat1, lon2, lat2 = map(np.radians, [lon_1, lat_1, lon2, lat2])
    # haversine formula
         dlon = lon2 - lon1
         dlat = lat2 - lat1
         a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
         c = 2 * math.asin(math.sqrt(a))
         km = 6367 * c
      
         if km<km_Min:
              km_Min=km
              Min_latlon=(lat,lon)
    Nearest_OD=OD.keys()[OD.values().index(Min_latlon)]
    return Nearest_OD
	

	
def locationToZoneNumberList(lat,lng,MAX_ZONE_NUMBER,searchArea,MIN_LAT,MIN_LNG):
    # Given a location, calculate the grid number of the neighboring areas
    # searchArea can adjust number of neighboring grids to search. if searchArea = 1, only search the 9 neighboring grids. If searchArea = 2, only search the 25 neighboring grids
    num_lat = int(round( (lat - MIN_LAT)/GRID_LENGTH_LAT))
    num_lng = int(round( (lng - MIN_LNG)/GRID_LENGTH_LNG))
    x_ID_list = []
    y_ID_list = []
    for i in range(2*searchArea + 1):
        x_ID_list.append(num_lat-searchArea + i)
        y_ID_list.append(num_lng-searchArea + i)
    zone_list = []
    for x in x_ID_list:
        for y in y_ID_list:
            tempID = x * ZONE + y
            if withinStudyArea(tempID, MAX_ZONE_NUMBER):
                zone_list.append(tempID)
    return zone_list

def locationToZoneNumber(lat,lng,MIN_LAT,MIN_LNG):
    num_lat = int(round( (lat - MIN_LAT)/GRID_LENGTH_LAT))
    num_lng = int(round( (lng - MIN_LNG)/GRID_LENGTH_LNG))
    return num_lat * ZONE + num_lng
def withinStudyArea(zoneID, maxZoneID):
    if zoneID >= 0 and zoneID <= maxZoneID:
        return True
    else:
        return False

GRID_LENGTH_LAT = 0.045  # 5 km
# 100m is 0.0009 for latitude
GRID_LENGTH_LNG = 0.001157 * 50   # 5 km
# 100m is 0.001157 for longitude
ZONE = 20000
# A number used for calculating the grid number

def findStudyArea(filename_node):
    # find the minimum rectangular study area that covers all the nodes
    minLat = 1000
    maxLat = -1000
    minLng = 1000
    maxLng = -1000
    tripDF = pd.read_csv(filename_node, index_col='Unnamed: 0')
    for i in tripDF.index.values:
        minLat = min(minLat, tripDF.loc[i,'y'])
        maxLat = max(maxLat, tripDF.loc[i,'y'])
        minLng = min(minLng, tripDF.loc[i,'x'])
        maxLng = max(maxLng, tripDF.loc[i,'x'])
    print minLat,maxLat,minLng,maxLng
    return [minLat,maxLat,minLng,maxLng]
findStudyArea('HERE_less_detailed_node_filtered.csv')

# read the filtered node/link files
# generate the network

node_dict = {}
cr = pd.read_csv('HERE_less_detailed_node_filtered.csv')
#df_tmc.iloc[0]
for i in cr.index.values:
    x1, y1 = float(cr.loc[i,'x']), float(cr.loc[i,'y'])
    node_dict[cr.loc[i,'node_id']] = (x1, y1)
# add a column "TMC" to df1: HERE_less_detailed_filtered.csv
link_list = []   #  From_node, to_node, fftt (seconds), length (meters), TMC
cr2 = pd.read_csv('HERE_less_detailed_filtered_TMC.csv')
for i in cr2.index.values:
    if cr2.loc[i,'TRAVEL_DIR']=="T":
        link_list.append((cr2.loc[i,'NON_REF_NO'], cr2.loc[i,'REF_NODE_I'], float(cr2.loc[i,'LENGTH'])/float(cr2.loc[i,'SPD_LIMIT'])*3.6, float(cr2.loc[i,'LENGTH']), cr2.loc[i,'TMC_T']))
    elif cr2.loc[i,'TRAVEL_DIR']=="F":
        link_list.append((cr2.loc[i,'REF_NODE_I'], cr2.loc[i,'NON_REF_NO'], float(cr2.loc[i,'LENGTH'])/float(cr2.loc[i,'SPD_LIMIT'])*3.6, float(cr2.loc[i,'LENGTH']), cr2.loc[i,'TMC_F']))
    else:
        link_list.append((cr2.loc[i,'REF_NODE_I'], cr2.loc[i,'NON_REF_NO'], float(cr2.loc[i,'LENGTH'])/float(cr2.loc[i,'SPD_LIMIT'])*3.6, float(cr2.loc[i,'LENGTH']), cr2.loc[i,'TMC_F']))
        link_list.append((cr2.loc[i,'NON_REF_NO'], cr2.loc[i,'REF_NODE_I'], float(cr2.loc[i,'LENGTH'])/float(cr2.loc[i,'SPD_LIMIT'])*3.6, float(cr2.loc[i,'LENGTH']), cr2.loc[i,'TMC_T']))


pos = {}
labels = {}

net_id = 0 # figure id

######################-create an undirected graph
shareGraph = nx.DiGraph() # this is an undirected graph

for nodeName, coord in node_dict.items():
    shareGraph.add_node(nodeName, coord=coord)
    pos[nodeName] = coord
    labels[nodeName] = str(nodeName)

random.seed(153)

for source, target, fftt, linkLen, TMC in link_list:
    # print source,target
    if (source in node_dict.keys() and target in node_dict.keys()):
        shareGraph.add_edge(source, target, length=linkLen, fftravelTime=fftt, tmc = TMC)
    else:
        print source, target, "not found in node_list"
        
        

connected_indicator = nx.is_strongly_connected(shareGraph)
print connected_indicator # False

print('check in out degres')
list_of_node = [n for n in shareGraph.nodes_iter()]
list_of_node_no_in =[]
list_of_node_no_out = []

for n_node in shareGraph.nodes_iter():
    if shareGraph.in_degree(n_node) == 0:
        # can only be origin
        list_of_node_no_in.append(n_node)
    elif shareGraph.out_degree(n_node) == 0:
        # can only be destination
        list_of_node_no_out.append(n_node)

		
import math
import numpy as np
import time

time_start = time.clock()
[minLat,maxLat,minLng,maxLng] = [38.52034776, 39.721312879999999, -77.853849999999994, -76.08861379999999]
MAX_ZONE_NUMBER = locationToZoneNumber(maxLat, maxLng,minLat,minLng)
print strftime("%Y-%m-%d %H:%M:%S", gmtime())




num_run = 1
#nx.draw_networkx_edges(shareGraph,pos)
for i in range(num_run):
    print('Shortest path: '+str(i))
    node1 = (39.399691, -76.606969)    # replace with input from API
    lat = node1[0]
    lng = node1[1]
    zoneList = locationToZoneNumberList(lat,lng,MAX_ZONE_NUMBER, 1,minLat,minLng)
    Onodes={}
    for item in zoneList:
        filename = 'grids/'+str(item)+'.txt'
        text_file = open(filename, "r")
        lines = text_file.readlines()
        for row in lines:
            row = row.strip('\n')
            row=row.split(',')
            if row[4]=='1':
                Onodes.update({row[1]:(row[2],row[3])})
        text_file.close()
    
    
    origin = int(haversine(lat, lng, Onodes))
    
    if origin in list_of_node_no_out:
        print origin,' cannot be a trip origin.'
        break

    node2 = (38.879951, -77.428798)
    lat = node2[0]
    lng = node2[1]
    zoneList = locationToZoneNumberList(lat,lng,MAX_ZONE_NUMBER, 1,minLat,minLng)
    Dnodes={}
    for item in zoneList:
        filename = 'grids/'+str(item)+'.txt'
        text_file = open(filename, "r")
        lines = text_file.readlines()
        for row in lines:
            row = row.strip('\n')
            row=row.split(',')
            if row[5]=='1':
                Dnodes.update({row[1]:(row[2],row[3])})
        text_file.close()
    destin = int(haversine(lat, lng, Dnodes))
    if destin in list_of_node_no_in:
        print destin, ' cannot be a trip destination.'
        break
    path=nx.astar_path(shareGraph, source=origin, target=destin,weight='fftravelTimes')
    path_edges = zip(path,path[1:])
    nx.draw_networkx_nodes(shareGraph,pos,nodelist=path,node_color='r')
    nx.draw_networkx_edges(shareGraph,pos,edgelist=path_edges,edge_color='r',width=10)

#print path    
#print path_edges
plt.show()
print strftime("%Y-%m-%d %H:%M:%S", gmtime())


time_elapsed = (time.clock() - time_start)
print time_elapsed