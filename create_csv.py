import csv
import os
import time
import numpy as np
import zipfile
import urllib.request
import shutil
import math
import json 
import sys



def getImages (Images):
    Image_Dict = dict()
    for image in Images:
        Image_Dict[image['id']] = image['file_name']

    return Image_Dict

def getCategories(Categories):
    Category_Dict = dict()
    #OutdoorSet = {"sports" , "accessory" , "animal" , "outdoor" , "vehicle" , "person"}
    OutdoorSet = {"sports"}
    for category in Categories:
        if category["supercategory"] in OutdoorSet:
            Category_Dict[category["id"]] = category["name"]
    return Category_Dict

  
def getData (annotations , Category_Dict , Image_Dict):
    dataDict = dict()
    for ann in annotations:
        if ann["category_id"] not in Category_Dict.keys():
            continue
        if Image_Dict[ann['image_id']] not in dataDict.keys():
            dataDict[Image_Dict[ann['image_id']]] = [[ann['bbox']] , [ann["category_id"]]]
        else:
            dataDict[Image_Dict[ann['image_id']]][0].append(ann['bbox'])
            dataDict[Image_Dict[ann['image_id']]][1].append(ann["category_id"])
    
    return dataDict

def create_csv(dataDict , imagePath):
    csv_file = open("csvFile.csv" , 'wb')
    l = 'filename,rois,classes'
    csv_file.write(l)
    csv_file.write("\n")
    for image in dataDict.keys():
        l = os.path.join(imagePath , image) + ',' + dataDict[image][0] + ',' + dataDict[image][1] 
        #csv_file.write(l)
        #csv_file.write('\n')
    
def process(dataDict , imagePath):
    filenames = []
    rois = []
    classes = []

    for image in dataDict.keys():
        filenames.append(os.path.join(imagePath , image))
        rois.append(dataDict[image][0])
        classes.append(dataDict[image][1])
       
    return filenames , rois , classes
        
        


