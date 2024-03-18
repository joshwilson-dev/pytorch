import os
import tkinter
from tkinter import filedialog
from tkinter import messagebox
from requests import get
from pandas import json_normalize
import piexif
from fractions import Fraction
from PIL import Image

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

file_path_variable = search_for_file_path()

def to_deg(value, loc):
    """convert decimal coordinates into degrees, munutes and seconds tuple
    Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
    return: tuple like (25, 13, 48.343 ,'N')
    """
    if value < 0:
        loc_value = loc[0]
    elif value > 0:
        loc_value = loc[1]
    else:
        loc_value = ""
    abs_value = abs(value)
    deg =  int(abs_value)
    t1 = (abs_value-deg)*60
    min = int(t1)
    sec = round((t1 - min)* 60, 5)
    return (deg, min, sec, loc_value)


def change_to_rational(number):
    """convert a number to rantional
    Keyword arguments: number
    return: tuple like (1, 2), (numerator, denominator)
    """
    f = Fraction(str(number))
    return (f.numerator, f.denominator)


def set_gps_location(file_name, lat, lng, altitude, exif_dict):
    """Adds GPS position as EXIF metadata
    Keyword arguments:
    file_name -- image file
    lat -- latitude (as float)
    lng -- longitude (as float)
    altitude -- altitude (as float)
    """
    lat_deg = to_deg(lat, ["S", "N"])
    lng_deg = to_deg(lng, ["W", "E"])

    exiv_lat = (change_to_rational(lat_deg[0]), change_to_rational(lat_deg[1]), change_to_rational(lat_deg[2]))
    exiv_lng = (change_to_rational(lng_deg[0]), change_to_rational(lng_deg[1]), change_to_rational(lng_deg[2]))

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
        piexif.GPSIFD.GPSAltitudeRef: 1,
        piexif.GPSIFD.GPSAltitude: change_to_rational(round(altitude)),
        piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
        piexif.GPSIFD.GPSLatitude: exiv_lat,
        piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
        piexif.GPSIFD.GPSLongitude: exiv_lng,
    }
    exif_dict['GPS'] = gps_ifd
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)

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

def set_camera_model(file_name, camera, exif_dict):
    image_ifd = {piexif.ImageIFD.Model: camera}
    exif_dict["0th"] = image_ifd
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)

def set_image_size_fl_dt(file_name, image_width, image_height, focal_length, datetime, exif_dict):
    exif_ifd = {piexif.ExifIFD.PixelXDimension: image_width,
                piexif.ExifIFD.PixelYDimension: image_height,
                piexif.ExifIFD.FocalLength: focal_length,
                piexif.ExifIFD.DateTimeDigitized: datetime}
    exif_dict["Exif"] = exif_ifd
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, file_name)

# latitude, longitude = -33.313854, 146.381643 # john lake
# latitude, longitude  = -30.085742, 151.782221 # john swamp
# latitude, longitude  = 53.690568, 23.304776 # white stork
# latitude, longitude = -51.076046, -61.084785 # GJ SEblob
# latitude, longitude = -51.033843, -61.230824 # SJ Hump
# latitude, longitude = -51.037927, -61.220572 # SJ Blob
# latitude, longitude = -51.031877, -61.231642 # SJ Bubble
# latitude, longitude = -51.066962, -61.107015 # GJ Middle Third
latitude, longitude = 44.766667, 12.250000 # Po Delta

height = 33.881
# height = 22 # white stork

# camera = "FC330" # DJI P4
# camera = "FC6310" # DJI PP4
camera = "FC7203" # DJI Mini

# focal_length = (361, 100) # white stork
# focal_length = (880, 100) # DJI PP4
focal_length = (449, 100) # DJI Mini

# datetime = "2018:06:22 15:12:44" # white stork
datetime = "2020:06:30 14:36:23"

elevation = get_elevation(latitude, longitude)
altitude = height + elevation

# did the user select a dir or cancel?
if len(file_path_variable) > 0:
    # confirm dir with user
    check = messagebox.askquestion(
        "CONFIRM",
        "Are you sure you want to add a metric to the annotations in:\n" + file_path_variable)
    if check =="yes":
        os.chdir(file_path_variable)
        # iterate through files in dir
        for root, dirs, files in os.walk(os.getcwd()):
            for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        print("Adding metrics to: ", file)
                        filepath = os.path.join(root, file)
                        image = Image.open(filepath)
                        image_width, image_height = image.size
                        try:
                            exif_dict = piexif.load(image.info['exif'])
                        except:
                            exif_dict = {}
                        set_gps_location(file, latitude, longitude, altitude, exif_dict)
                        set_camera_model(file, camera, exif_dict)
                        set_image_size_fl_dt(file, image_width, image_height, focal_length, datetime, exif_dict)