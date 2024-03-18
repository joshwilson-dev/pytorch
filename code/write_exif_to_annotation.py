import os
import json
from PIL import Image
import piexif
import re
from requests import get
from pandas import json_normalize

#################
#### Content ####
#################

def degrees(tag):
    d = tag[0][0] / tag[0][1]
    m = tag[1][0] / tag[1][1]
    s = tag[2][0] / tag[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def dms_to_dd(d, m, s):
    dd = d + float(m)/60 + float(s)/3600
    return dd

def get_gps(exif):
    # altitue
    altitude_tag = exif['GPS'][piexif.GPSIFD.GPSAltitude]
    # altitude_ref = exif['GPS'][piexif.GPSIFD.GPSAltitudeRef]
    altitude = altitude_tag[0]/altitude_tag[1]
    # below_sea_level = altitude_ref != 0
    # altitude = -altitude if below_sea_level else altitude
    # latitude
    latitude_tag = exif['GPS'][piexif.GPSIFD.GPSLatitude]
    latitude_ref = exif['GPS'][piexif.GPSIFD.GPSLatitudeRef].decode("utf-8")
    latitude = degrees(latitude_tag)
    latitude = -latitude if latitude_ref == 'S' else latitude
    # longitude
    longitude_tag = exif['GPS'][piexif.GPSIFD.GPSLongitude]
    longitude_ref = exif['GPS'][piexif.GPSIFD.GPSLongitudeRef].decode("utf-8")
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

def get_gsd(exif, camera, height):
    image_width = exif["Exif"][piexif.ExifIFD.PixelXDimension]
    image_height = exif["Exif"][piexif.ExifIFD.PixelYDimension]
    focal_length = exif["Exif"][piexif.ExifIFD.FocalLength][0] / exif["Exif"][piexif.ExifIFD.FocalLength][1]
    if camera == "M3E" and image_height == 3000: camera = "M3EZ"
    sensor_width, sensor_length = sensor_size[camera]
    pixel_pitch = max(sensor_width / image_width, sensor_length / image_height)
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
    "FC300C": [6.16, 4.62],
    "FC7203": [6.3, 4.7],
    "FC6520": [17.3, 13],
    "FC6310": [13.2, 8.8],
    "FC3411": [13.2, 8.8],
    "L1D-20c": [13.2, 8.8],
    "Canon PowerShot G15": [7.44, 5.58],
    "NX500": [23.50, 15.70],
    "Canon PowerShot S100": [7.44, 5.58],
    "Survey2_RGB": [6.17472, 4.63104],
    "ILCE-7R": [35.9, 24],
    "DSC-RX1": [35.8, 23.8],
    "iXU150": [53.4, 40],
    "M30T": [6.4, 4.8],
    "MAVIC2-ENTERPRISE-ADVANCED": [6.4, 4.8],
    "M3EZ": [6.4, 4.8],
    "M3E": [17.3, 13]
    }

root = "C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBD"
# root = "C:/Users/uqjwil54/Documents/Projects/bird-detector/images"
# root = "C:/Users/uqjwil54/Documents/Projects/object-detection-inference/images"

new = False
os.chdir(root)
# iterate through files in dir
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if "fully annotated" in root or "background" in root:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                # print("Adding metrics to: ", file)
                image_file_path = os.path.join(root, file)
                annotation_file = os.path.splitext(file)[0] + ".json"
                annotation_file_path = os.path.join(root, annotation_file)
                # get the gsd & associated metrics
                image = Image.open(image_file_path)
                try:
                    exif = piexif.load(image.info['exif'])
                except:
                    print("\t Couldn't get exif, skipping: ", file)
                    continue

                # print(exif)
                if os.path.exists(annotation_file_path):
                    annotation = json.load(open(annotation_file_path))
                else:
                    print("\tAnnotation doesn't exist, skipping: ", file)
                    continue

                # get the camera
                try:
                    if new == True: error
                    camera = annotation["camera"]
                    # print("\tGot camera from annotation: ", camera)
                except:
                    # print("\tCouldn't get camera from annotation")
                    try:
                        camera = exif["0th"][piexif.ImageIFD.Model].decode("utf-8").rstrip('\x00')
                        # print("\tGot camera from exif: ", camera)
                    except:
                        print("\t Couldn't get camera, skipping: ", file)
                        continue

                # get the altitude, latitude, and longitude
                try:
                    if new == True: error
                    altitude = annotation["altitude"]
                    latitude = annotation["latitude"]
                    longitude = annotation["longitude"]
                    # print("\tAltitude, latitude, and longitude already in annotation")
                except:
                    # print("\tCouldn't get altitude, latitude, and longitude from annotation")
                    try:
                        altitude, latitude, longitude = get_gps(exif)
                        # print("\tGot altitude, latitude, and longitude from exif")
                    except:
                        print("\t Couldn't get altitude, latitude, and longitude, skipping: ", file)
                        continue
                
                # get the gsd
                try:
                    if new == True: error
                    gsd = annotation["gsd"]
                    # print("\tGSD already in annotation")
                except:
                    # print("\tCouldn't get GSD from annotation")
                    try:
                        # get xmp information
                        f = open(image_file_path, 'rb')
                        d = f.read()
                        xmp_start = d.find(b'<x:xmpmeta')
                        xmp_end = d.find(b'</x:xmpmeta')
                        xmp_str = (d[xmp_start:xmp_end+12]).lower()
                        # print(xmp_str)
                        # Extract dji info
                        dji_xmp_keys = ['relativealtitude']
                        dji_xmp = {}
                        for key in dji_xmp_keys:
                            search_str = (key).encode("UTF-8")
                            value_start = xmp_str.find(search_str)
                            value = re.search(b"(\-|\+)[0-9]+.[0-9]+", xmp_str[value_start:]).group(0)
                            dji_xmp[key] = float(value.decode('UTF-8'))
                        height = dji_xmp["relativealtitude"]
                        # print("\tGot height from xmp")
                    except:
                        # print("\tCouldn't get height from xmp")
                        try:
                            elevation = get_elevation(latitude, longitude)
                            height = altitude - elevation
                            # print("\tGot height from EXIF")
                        except:
                            # print("\tCouldn't get height from exif")
                            print("\tCouldn't determine height, skipping: ", file)
                            continue
                    # print("\tCalculating GSD...")
                    try:
                        gsd = get_gsd(exif, camera, height)
                        # print("\tGot GSD from exif")
                    except:
                        print("\tCouldn't calculate GSD, skipping: ", file)
                        continue
                # check if the annotation is complete
                try:
                    if new == True: error
                    complete = annotation["complete"]
                    # print("\tComplete already in annotation")
                except:
                    # print("\tCouldn't get complete from annotation")
                    if "partially annotated" in root:
                        complete = "false"
                    else: complete = "true"
                    # print("\tGot complete from directory")
                # get the date
                try:
                    if new == True: error
                    datetime = annotation["datetime"]
                    # print("\tdatetime already in annotation")
                except:
                    # print("\tCouldn't get datetime from annotation")
                    datetime = exif["Exif"][piexif.ExifIFD.DateTimeDigitized].decode("utf-8")
                    # print("\tGot datetime from exif")

                # add metrics to annotation
                if "christian_pfeifer" in root: gsd = int(file[:3])/1000
                annotation["gsd"] = gsd
                annotation["camera"] = camera
                annotation["altitude"] = altitude
                annotation["latitude"] = latitude
                annotation["longitude"] = longitude
                annotation["complete"] = complete
                annotation["datetime"] = datetime

                # save annotation
                annotation = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                with open(annotation_file_path, 'w') as annotation_file:
                    annotation_file.write(annotation)
                # print("\tWrote metrics to annotation")