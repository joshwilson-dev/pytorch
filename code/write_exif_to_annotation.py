################
#### Header ####
################

# Title: Add GSD to Annotation
# Author: Josh Wilson
# Date: 26-12-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for images with annotation files and adds the image
# gsd to the flag key of the annotation

###############
#### Setup ####
###############

import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
from PIL import Image
import piexif
import re
from requests import get
from pandas import json_normalize

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

def dms_to_dd(d, m, s):
    dd = d + float(m)/60 + float(s)/3600
    return dd

def get_gps(exif_dict):
    # altitue
    altitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
    # altitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef]
    altitude = altitude_tag[0]/altitude_tag[1]
    # below_sea_level = altitude_ref != 0
    # altitude = -altitude if below_sea_level else altitude
    # latitude
    latitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
    latitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
    latitude = degrees(latitude_tag)
    latitude = -latitude if latitude_ref == 'S' else latitude
    # longitude
    longitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
    longitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
    longitude = degrees(longitude_tag)
    longitude = -longitude if longitude_ref == 'W' else longitude
    return altitude, latitude, longitude

def get_elevation(latitude, longitude):
    # query = 'https://api.open-elevation.com/api/v1/lookup'f'?locations={latitude},{longitude}'
    query = 'https://api.opentopodata.org/v1/aster30m?locations='f'{latitude},{longitude}'
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)
    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def get_gsd(exif, height):
    # get camera specs
    camera_model = exif["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
    image_width = exif["Exif"][piexif.ExifIFD.PixelXDimension]
    image_length = exif["Exif"][piexif.ExifIFD.PixelYDimension]
    sensor_width, sensor_length = sensor_size[camera_model]
    focal_length = exif["Exif"][piexif.ExifIFD.FocalLength][0] / exif["Exif"][piexif.ExifIFD.FocalLength][1]
    pixel_pitch = max(sensor_width / image_width, sensor_length / image_length)
    # calculate gsd
    gsd = height * pixel_pitch / focal_length
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
    "Survey2_RGB": [6.17472, 4.63104],
    "ILCE-7R": [35.9, 24]
    }

file_path_variable = search_for_file_path()

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to add gsd to the annotations in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    print("Adding GSD to: ", file)
                    image_file_path = os.path.join(root, file)
                    annotation_file = os.path.splitext(file)[0] + ".json"
                    annotation_file_path = os.path.join(root, annotation_file)
                    # get the gsd & associated metrics
                    image = Image.open(image_file_path)
                    exif_dict = piexif.load(image.info['exif'])
                    if os.path.exists(annotation_file_path):
                        annotation = json.load(open(annotation_file_path))
                    else:
                        print("\tAnnotation doesn't exist, skipping file")
                        continue
                    try:
                        gsd = annotation["gsd"]
                        print("\tGSD already in annotation, skipping file")
                        continue
                    except:
                        print("\tCouldn't get GSD from annotation")
                        try:
                            # get xmp information
                            f = open(image_file_path, 'rb')
                            d = f.read()
                            xmp_start = d.find(b'<x:xmpmeta')
                            xmp_end = d.find(b'</x:xmpmeta')
                            xmp_str = (d[xmp_start:xmp_end+12]).lower()
                            # Extract dji info
                            dji_xmp_keys = ['relativealtitude']
                            dji_xmp = {}
                            for key in dji_xmp_keys:
                                search_str = (key).encode("UTF-8")
                                value_start = xmp_str.find(search_str)
                                value = re.search(b"(\-|\+)[0-9]+.[0-9]+", xmp_str[value_start:]).group(0)
                                dji_xmp[key] = float(value.decode('UTF-8'))
                            height = dji_xmp["relativealtitude"]
                            print("\tGot height from xmp")
                        except:
                            print("\tCouldn't get height from xmp")
                            try:
                                altitude, latitude, longitude = get_gps(exif_dict)
                                elevation = get_elevation(latitude, longitude)
                                height = altitude - elevation
                                print("\tGot height from EXIF and elevation")
                            except:
                                print("\tCouldn't get height from exif")
                                print("\tCouldn't determine height, skipping file")
                                continue
                        print("\tCalculating gsd...")
                        try:
                            gsd = get_gsd(exif_dict, height)
                        except:
                            print("\tCouldn't calculate GSD, skipping file")
                            continue
                        # add gsd to annotation
                        annotation["gsd"] = gsd
                        annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                        with open(annotation_file_path, 'w') as annotation_file:
                            annotation_file.write(annotation)
                        print("\tWrote GSD to annotation")