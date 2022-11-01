################
#### Header ####
################

# Title: curate data within sub directory
# Author: Josh Wilson
# Date: 02-06-2022
# Description: 
# This script runs through sub-dirs of the selected directory
# looking for images, creates a csv with info about each image

###############
#### Setup ####
###############

import os
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

def get_gsd(exif, height):
    # get camera specs
    try:
        camera_model = exif["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
        image_width = exif["Exif"][piexif.ExifIFD.PixelXDimension]
        image_length = exif["Exif"][piexif.ExifIFD.PixelYDimension]
        sensor_width, sensor_length = sensor_size[camera_model]
        focal_length = exif["Exif"][piexif.ExifIFD.FocalLength][0] / exif["Exif"][piexif.ExifIFD.FocalLength][1]
        pixel_pitch = max(sensor_width / image_width, sensor_length / image_length)
        # calculate gsd
        gsd = height * pixel_pitch / focal_length
    except:
        gsd = "NA"
        image_width = "NA"
        image_length = "NA"
        sensor_width = "NA"
        sensor_length = "NA"
        focal_length = "NA"

    metrics = [gsd, image_width, image_length, sensor_width, sensor_length, focal_length]
    return metrics

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
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image_file_path = os.path.join(root, file)
                    print(image_file_path)
                    # get exif data
                    image = Image.open(image_file_path)
                    exif_dict = piexif.load(image.info['exif'])
                    try:
                        # check if XPcomment tag already contains required metrics
                        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        ecosystem_typology = is_float(comments["ecosystem typology"])
                        latitude = is_float(comments["latitude"])
                        longitude = is_float(comments["longitude"])
                        altitude = is_float(comments["altitude"])
                        height = is_float(comments["height"])
                        elevation = is_float(comments["elevation"])
                        image_width = is_float(comments["image_width"])
                        image_length = is_float(comments["image_length"])
                        sensor_width = is_float(comments["sensor_width"])
                        sensor_length = is_float(comments["sensor_length"])
                        focal_length = is_float(comments["focal_length"])
                        gimbalroll = is_float(comments["gimbalroll"])
                        gimbalpitch = is_float(comments["gimbalpitch"])
                        gimbalyaw = is_float(comments["gimbalyaw"])
                        flightroll = is_float(comments["flightroll"])
                        flightyaw = is_float(comments["flightyaw"])
                        flightpitch = is_float(comments["flightpitch"])
                        gsd = is_float(comments["gsd"])
                        print("got metrics from comments")
                        ecosystem_typology = os.path.basename(root)
                        comments = json.dumps({
                            "ecosystem typology":str(ecosystem_typology),
                            "latitude":str(latitude),
                            "longitude": str(longitude),
                            "altitude":str(altitude),
                            "height":str(height),
                            "elevation":str(elevation),
                            "image_width":str(image_width),
                            "image_length":str(image_length),
                            "sensor_width":str(sensor_width),
                            "sensor_length":str(sensor_length),
                            "focal_length":str(focal_length),
                            "gimbalroll":str(gimbalroll),
                            "gimbalpitch":str(gimbalpitch),
                            "gimbalyaw":str(gimbalyaw),
                            "flightroll":str(flightroll),
                            "flightyaw":str(flightyaw),
                            "flightpitch":str(flightpitch),
                            "gsd":str(gsd)})
                        exif_dict["0th"][piexif.ImageIFD.XPComment] = comments.encode('utf-16le')
                        # Convert into bytes and dump into file
                        exif_bytes = piexif.dump(exif_dict)
                        piexif.insert(exif_bytes, image_file_path)
                    except:
                        print("couldn't get metrics from comments")
                        try:
                            # get xmp information
                            f = open(image_file_path, 'rb')
                            d = f.read()
                            xmp_start = d.find(b'<x:xmpmeta')
                            xmp_end = d.find(b'</x:xmpmeta')
                            xmp_str = (d[xmp_start:xmp_end+12]).lower()
                            # Extract dji info
                            dji_xmp_keys = [
                                'absolutealtitude',
                                'relativealtitude',
                                'gimbalrolldegree',
                                'gimbalyawdegree',
                                'gimbalpitchdegree',
                                'flightrolldegree',
                                'flightyawdegree',
                                'flightpitchdegree']
                            dji_xmp = {}
                            for key in dji_xmp_keys:
                                search_str = (key + '="').encode("UTF-8")
                                value_start = xmp_str.find(search_str) + len(search_str)
                                value_end = xmp_str.find(b'"', value_start)
                                value = xmp_str[value_start:value_end]
                                dji_xmp[key] = float(value.decode('UTF-8'))
                            altitude = dji_xmp["absolutealtitude"]
                            height = dji_xmp["relativealtitude"]
                            gimbalroll = dji_xmp["gimbalrolldegree"]
                            gimbalpitch = dji_xmp["gimbalpitchdegree"]
                            gimbalyaw = dji_xmp["gimbalyawdegree"]
                            flightroll = dji_xmp["flightrolldegree"]
                            flightyaw = dji_xmp["flightyawdegree"]
                            flightpitch = dji_xmp["flightpitchdegree"]
                            print("got drone metrics from xmp")
                        except:
                            try:
                                print("couldn't get drone metrics from xmp")
                                altitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSAltitude]
                                altitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSAltitudeRef]
                                altitude = altitude_tag[0]/altitude_tag[1]
                                below_sea_level = altitude_ref != 0
                                altitude = -altitude if below_sea_level else altitude
                                gimbalroll = "NA"
                                gimbalpitch = "NA"
                                gimbalyaw = "NA"
                                flightroll = "NA"
                                flightyaw = "NA"
                                flightpitch = "NA"
                                print("got altitude from exif")
                            except:
                                print("couldn't get altitude from exif")
                                altitude = 0
                                gimbalroll = "NA"
                                gimbalpitch = "NA"
                                gimbalyaw = "NA"
                                flightroll = "NA"
                                flightyaw = "NA"
                                flightpitch = "NA"
                        try:
                            latitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLatitude]
                            longitude_tag = exif_dict['GPS'][piexif.GPSIFD.GPSLongitude]
                            latitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
                            longitude_ref = exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
                            latitude = degrees(latitude_tag)
                            longitude = degrees(longitude_tag)
                            latitude = -latitude if latitude_ref == 'S' else latitude
                            longitude = -longitude if longitude_ref == 'W' else longitude
                            print("got GPS from exif")
                        except:
                            print("couldn't get GPS location from exif")
                            latitude = 0
                            longitude = 0
                        print("calculating gsd...")
                        # calculate drone height above surface
                        # get ground elevation at drone location, if possible
                        elevation = get_elevation(latitude, longitude)
                        try: height
                        except: height = altitude - elevation
                        # get the gsd & associated metrics
                        gsd, image_width, image_length, sensor_width, sensor_length, focal_length = get_gsd(exif_dict, height)
                        # show image and ask user for ecosystem_typology
                        # image.show()
                        # ecosystem_typology = input("Enter a ecosystem_typology: ")
                        # ecosystem_typology = "rocks"
                        ecosystem_typology = os.path.basename(root)
                        # write gsd and elevation to comments
                        # latitude = "NA"
                        # longitude = "NA"
                        # altitude = "NA"
                        # height = "NA"
                        # elevation = "NA"
                        # image_width = "NA"
                        # image_length = "NA"
                        # sensor_width = "NA"
                        # sensor_length = "NA"
                        # focal_length = "NA"
                        # gimbalroll = "NA"
                        # gimbalpitch = "NA"
                        # gimbalyaw = "NA"
                        # flightroll = "NA"
                        # flightyaw = "NA"
                        # flightpitch = "NA"
                        # gsd = 0.01

                        comments = json.dumps({
                            "ecosystem typology":str(ecosystem_typology),
                            "latitude":str(latitude),
                            "longitude": str(longitude),
                            "altitude":str(altitude),
                            "height":str(height),
                            "elevation":str(elevation),
                            "image_width":str(image_width),
                            "image_length":str(image_length),
                            "sensor_width":str(sensor_width),
                            "sensor_length":str(sensor_length),
                            "focal_length":str(focal_length),
                            "gimbalroll":str(gimbalroll),
                            "gimbalpitch":str(gimbalpitch),
                            "gimbalyaw":str(gimbalyaw),
                            "flightroll":str(flightroll),
                            "flightyaw":str(flightyaw),
                            "flightpitch":str(flightpitch),
                            "gsd":str(gsd)})
                        exif_dict["0th"][piexif.ImageIFD.XPComment] = comments.encode('utf-16le')
                        # Convert into bytes and dump into file
                        exif_bytes = piexif.dump(exif_dict)
                        piexif.insert(exif_bytes, image_file_path)
                        del height