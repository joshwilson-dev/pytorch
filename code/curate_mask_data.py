################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for mask annotations and creates a csv with the gsd
# of each mask instance

###############
#### Setup ####
###############

import os
import csv
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
from requests import get
from pandas import json_normalize
from PIL import Image
import piexif

#################
#### Content ####
#################

# create function for user to select dir
root = tkinter.Tk()
root.withdraw()

def search_for_file_path ():
    currdir = os.getcwd()
    tempdir = filedialog.askdirectory(
        parent=root,
        initialdir=currdir,
        title='Please select a directory')
    if len(tempdir) > 0:
        print ("You chose: %s" % tempdir)
    return tempdir

def degrees(tag):
    d = tag[0][0] / tag[0][1]
    m = tag[1][0] / tag[1][1]
    s = tag[2][0] / tag[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def get_elevation(latitude, longitude):
    query = ('https://api.open-elevation.com/api/v1/lookup'f'?locations={latitude},{longitude}')
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)
    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def get_xmp(image_path):
    # get xmp information
    f = open(image_path, 'rb')
    d = f.read()
    xmp_start = d.find(b'<x:xmpmeta')
    xmp_end = d.find(b'</x:xmpmeta')
    xmp_str = (d[xmp_start:xmp_end+12]).lower()
    # Extract dji info
    dji_xmp_keys = ['relativealtitude']
    dji_xmp = {}
    for key in dji_xmp_keys:
        search_str = (key + '="').encode("UTF-8")
        value_start = xmp_str.find(search_str) + len(search_str)
        value_end = xmp_str.find(b'"', value_start)
        value = xmp_str[value_start:value_end]
        dji_xmp[key] = float(value.decode('UTF-8'))
    height = dji_xmp["relativealtitude"]
    return height

def get_gps(exif_dict):
    latitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    longitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    latitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
    longitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
    latitude = degrees(latitude_tag)
    longitude = degrees(longitude_tag)
    latitude = -latitude if latitude_ref == 'S' else latitude
    longitude = -longitude if longitude_ref == 'W' else longitude
    return latitude, longitude

def get_altitude(exif_dict):
    altitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
    altitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef]
    altitude = altitude_tag[0]/altitude_tag[1]
    below_sea_level = altitude_ref != 0
    altitude = -altitude if below_sea_level else altitude
    return altitude

def get_gsd(exif_dict, image_path, image_height, image_width):
    print("Trying to get GSD")
    # try to get gsd from comments
    try:
        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
        gsd = is_float(comments["gsd"])
        print("Got GSD from comments")
    except:
        print("Couldn't get GSD from comments")
        try:
            print("Trying to calculate GSD from height and camera")
            height = get_xmp(image_path)
            print("Got height from xmp")
        except:
            print("Couldn't get height from xmp")
            print("Trying to infer height from altitude and elevation a GPS")
            height = 0
            # try:
            #     latitude, longitude = get_gps(exif_dict)
            #     print("Got GPS from exif")
            # except:
            #     print("Couldn't get GPS")
            #     return
            # try:
            #     altitude = get_altitude(exif_dict)
            #     print("Got altiude from exif")
            # except:
            #     print("Couldn't get altitude")
            #     return
            # try:
            #     elevation = get_elevation(latitude, longitude)
            #     print("Got elevation from exif")
            #     height = altitude - elevation
            # except:
            #     print("Couldn't get elevation")
            #     return
        try:
            camera_model = exif_dict["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
            print("Got camera model from exif")
        except:
            print("Couldn't get camera model from exif")
            return
        try:
            sensor_width, sensor_length = sensor_size[camera_model]
            print("Got sensor dimensions")
        except:
            print("Couldn't get sensor dimensions from sensor size dict")
            return
        try:
            focal_length = exif_dict["Exif"][piexif.ExifIFD.FocalLength][0] / exif_dict["Exif"][piexif.ExifIFD.FocalLength][1]
            print("Got focal length from exif")
        except:
            print("Couldn't get focal length from exif")
            return
        pixel_pitch = max(sensor_width / image_width, sensor_length / image_height)
        gsd = height * pixel_pitch / focal_length
        print("GSD: ", gsd)
        return gsd

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string

sensor_size = {
    "FC220": [6.16, 4.55],
    "FC330": [6.16, 4.62],
    "FC7203": [6.3, 4.7],
    "FC6520": [17.3, 13],
    "FC6310": [13.2, 8.8],
    "L1D-20c": [13.2, 8.8],
    "Canon PowerShot G15": [7.44, 5.58],
    "NX500": [23.50, 15.70],
    "Canon PowerShot S100": [7.44, 5.58],
    "Survey2_RGB": [6.17472, 4.63104]
    }

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to curate the files in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        header = False
        dataset = {"label": [], "gsd": [], "image": []}
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if "mask" in root:
                    if file.endswith(".json"):
                        print(file)
                        # load annotation
                        annotation_path = os.path.join(root, file)
                        annotation = json.load(open(annotation_path))
                        # load image
                        image_name = annotation["imagePath"]
                        image_path = os.path.join(root, image_name)
                        image = Image.open(image_path)
                        image_width, image_height = image.size
                        # read exif data
                        exif_dict = piexif.load(image.info['exif'])
                        # get gsd
                        gsd = get_gsd(exif_dict, image_path, image_height, image_width)
                        for instance in annotation["shapes"]:
                            dataset["label"].append(instance["label"])
                            dataset["gsd"].append(gsd)
                            dataset["image"].append(image_name)
        # save dict to csv
        with open("masks.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dataset.keys())
            writer.writerows(zip(*dataset.values()))