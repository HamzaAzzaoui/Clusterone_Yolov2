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

from create_csv import *
from make_tfrecord import *


def create_record(ImgPath , JsonPath):
    f = open(JsonPath)
    data = json.load(f)

    Image_Dict = getImages(data['images'])
    annotations = data['annotations']
    Category_Dict = getCategories(data["categories"])

    dataDict = getData(annotations , Category_Dict , Image_Dict)

    
    results = process(dataDict , ImgPath)
    
    return results

if __name__ == "__main__":
    JsonPath = "/media/bnsfuser/KINGSTON/hamza/instances_val2017.json"
    imgPath = "/home/bnsfuser/Desktop/val2017"
    results = create_record(imgPath , JsonPath)
    make_record(results)
    