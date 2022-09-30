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

from lib2to3.pgen2.pgen import DFAState
import os
import csv
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import json
from requests import get
from pandas import json_normalize
import piexif
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageOps
import numpy 
import hashlib
import torchvision.transforms as T
import shutil

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

def crop_mask(im, points, label, gsd_cat):
    polygon = [tuple(l) for l in points]
    pad = 1
    # find bounding box
    max_x = max([t[0] for t in polygon])
    min_x = min([t[0] for t in polygon])
    max_y = max([t[1] for t in polygon])
    min_y = min([t[1] for t in polygon])
    # subtract xmin andymin from each point
    origin = [(min_x - pad, min_y - pad)] * len(polygon)
    polygon = tuple(tuple(a - b for a, b in zip(tup1, tup2))\
        for tup1, tup2 in zip(polygon, origin))
    # crop to bounding box
    bird = im.crop((min_x - pad, min_y - pad, max_x + pad, max_y + pad))
    # convert to numpy (for convenience)
    imArray = numpy.asarray(bird)
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)
    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')
    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]
    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")
    # calculate md5#
    md5hash = hashlib.md5(newIm.tobytes()).hexdigest()
    # save image
    # if path doesn't exist, create it
    path = os.path.join("instances", label, gsd_cat)
    if not os.path.exists(path):
        os.makedirs(path)
    name = md5hash + ".png"
    newIm.save(os.path.join(path, name), format = "png")
    return [name, polygon]

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
        instances = {"species": [], "instance_gsd": [], "instance_gsd_cat": [], "instance_id": [], "instance_points": [], "instance_downsample": []}
        backgrounds = {"substrate": [], "background_gsd": [], "background_gsd_cat": [],"background_id": [], "background_downsample": []}
        gsd_cats = ["very fine", "fine", "coarse", "very coarse"]
        gsd_bins = [0.004, 0.007, 0.01, 0.013, 0.016]
        paths = ["instances", "substrates", "dataset"]
        for path in paths:
            if os.path.exists(path):
                shutil.rmtree(path)
        for root, dirs, files in os.walk(os.getcwd()):
            instance_id = 0
            for file in files:
                if "mask" in root:
                    if file.endswith(".json"):
                        # load annotation
                        annotation_path = os.path.join(root, file)
                        annotation = json.load(open(annotation_path))
                        # load image
                        image_name = annotation["imagePath"]
                        image_path = os.path.join(root, image_name)
                        image = Image.open(image_path).convert("RGBA")
                        image_width, image_height = image.size
                        # read exif data
                        exif_dict = piexif.load(image.info['exif'])
                        # get gsd
                        comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                        gsd = is_float(comments["gsd"])
                        try: gsd_cat = gsd_cats[numpy.digitize(gsd,gsd_bins) - 1]
                        except: gsd_cat = numpy.nan
                        # add to dictionary
                        for instance in annotation["shapes"]:
                            instance_id, points = crop_mask(image.convert("RGBA"), instance["points"], instance["label"], gsd_cat)
                            instances["species"].append(instance["label"])
                            instances["instance_gsd"].append(gsd)
                            instances["instance_gsd_cat"].append(gsd_cat)
                            instances["instance_id"].append(instance_id)
                            instances["instance_points"].append(points)
                            instances["instance_downsample"].append(1)

                if "backgrounds" in root:
                    image_name = file
                    image_path = os.path.join(root, image_name)
                    image = Image.open(image_path).convert("RGBA")
                    image_width, image_height = image.size
                    # read exif data
                    exif_dict = piexif.load(image.info['exif'])
                    # get gsd
                    comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
                    gsd = is_float(comments["gsd"])
                    try: gsd_cat = gsd_cats[numpy.digitize(gsd,gsd_bins) - 1]
                    except: gsd_cat = numpy.nan
                    # calculate md5#
                    md5hash = hashlib.md5(image.tobytes()).hexdigest()
                    # save image
                    path = os.path.join("substrates", os.path.basename(root), gsd_cat)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    name =  md5hash + ".png"
                    shutil.copy(image_path, os.path.join(path, name))
                    # add to dictionary
                    backgrounds["substrate"].append(os.path.basename(root))
                    backgrounds["background_gsd"].append(gsd)
                    backgrounds["background_gsd_cat"].append(gsd_cat)
                    backgrounds["background_id"].append(name)
                    backgrounds["background_downsample"].append(1)
        # convert dictionary to df
        instances = pd.DataFrame(data=instances)
        backgrounds = pd.DataFrame(data=backgrounds)
        # remove masks with unsuitable gsd values
        instances = instances.dropna()
        backgrounds = backgrounds.dropna()
        # check that each species has enough instances in each gsd category
        instances_per_species = 2
        for species in instances.species.unique():
            for gsd_cat in reversed(gsd_cats):
                count = instances[instances['species'] == species]
                count = count[instances['instance_gsd_cat'] == gsd_cat]
                count = len(count)
                # if gsd does not have at least instances_per_species instances per gsd_cat:
                # re-sample if very fine
                # otherwise downscale images from high res
                if count < instances_per_species:
                    min_gsd = gsd_bins[gsd_cats.index(gsd_cat)]
                    max_gsd = gsd_bins[gsd_cats.index(gsd_cat) + 1]
                    mean_gsd = (max_gsd - min_gsd)/2
                    high_res = instances[instances['species'] == species]
                    if gsd_cat == "very fine":
                        high_res = high_res[instances['instance_gsd'] <= max_gsd]
                    else:
                        high_res = high_res[instances['instance_gsd'] <= min_gsd]
                    downscale = high_res.sample(n=instances_per_species - count, replace=True)
                    downscale = downscale.assign(instance_downsample=lambda df: random.uniform(min_gsd, max_gsd)/df.instance_gsd)
                    downscale = downscale.reset_index(drop=True)
                    for index, row in downscale.iterrows():
                        path = os.path.join("instances", row["species"], row["instance_gsd_cat"])
                        instance = Image.open(os.path.join(path, row["instance_id"])).convert("RGBA")
                        size = max(instance.size) / row["instance_downsample"]
                        instance = T.Resize(size=int(size))(instance)
                        # calculate md5#
                        md5hash = hashlib.md5(instance.tobytes()).hexdigest()
                        # save image
                        path = os.path.join("instances", row["species"], gsd_cat)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        name = md5hash + ".png"
                        instance.save(os.path.join(path, name), format = "png")
                        # resize points
                        points = row["instance_points"]
                        points = tuple(tuple(item / row["instance_downsample"] for item in one) for one in points)
                        downscale.at[index,'instance_points'] = points
                        downscale.at[index,'instance_id'] = name
                    downscale = downscale.assign(instance_gsd_cat=gsd_cat)
                    instances = pd.concat([instances, downscale])
                # if gsd has more than instances_per_species instances drop it down to instances_per_species
                elif count > instances_per_species:
                    drop = instances[instances['species'] == species]
                    drop = drop[drop['instance_gsd_cat'] == gsd_cat]
                    drop = drop.sample(n=count - instances_per_species)
                    for index, row in drop:
                        os.remove(os.path.join("instances", row["species"], gsd_cat, row["instance_id"]))
                    instances.drop(drop.index)

        # check that each background has enough instances in each gsd category
        backgrounds_per_gsd_cat = 2
        for substrate in backgrounds.substrate.unique():
            for gsd_cat in reversed(gsd_cats):
                count = backgrounds[backgrounds['substrate'] == substrate]
                count = count[backgrounds['background_gsd_cat'] == gsd_cat]
                count = len(count)
                # if gsd does not have at least backgrounds_per_gsd_cat instances per gsd_cat:
                # re-sample if very fine
                # otherwise downscale images from high res
                if count < backgrounds_per_gsd_cat:
                    min_gsd = gsd_bins[gsd_cats.index(gsd_cat)]
                    max_gsd = gsd_bins[gsd_cats.index(gsd_cat) + 1]
                    mean_gsd = (max_gsd - min_gsd)/2
                    high_res = backgrounds[backgrounds['substrate'] == substrate]
                    if gsd_cat == "very fine":
                        high_res = high_res[backgrounds['background_gsd'] <= max_gsd]
                    else:
                        high_res = high_res[backgrounds['background_gsd'] <= min_gsd]
                    downscale = high_res.sample(n=backgrounds_per_gsd_cat - count, replace=True)
                    downscale = downscale.assign(background_downsample=lambda df: random.uniform(min_gsd, max_gsd)/df.background_gsd)
                    downscale = downscale.reset_index(drop=True)
                    for index, row in downscale.iterrows():
                        path = os.path.join("substrates", row["substrate"], row["background_gsd_cat"])
                        background = Image.open(os.path.join(path, row["background_id"])).convert("RGBA")
                        size = max(background.size) / row["background_downsample"]
                        background = T.Resize(size=int(size))(background)
                        # calculate md5#
                        md5hash = hashlib.md5(background.tobytes()).hexdigest()
                        # save image
                        path = os.path.join("substrates", row["substrate"], gsd_cat)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        name = md5hash + ".png"
                        background.save(os.path.join(path, name), format = "png")
                        downscale.at[index,'background_id'] = name
                    downscale = downscale.assign(background_gsd_cat=gsd_cat)
                    backgrounds = pd.concat([backgrounds, downscale])
                # if gsd has more than backgrounds_per_gsd_cat instances drop it down to backgrounds_per_gsd_cat
                elif count > backgrounds_per_gsd_cat:
                    drop = backgrounds[backgrounds['substrate'] == substrate]
                    drop = drop[drop['background_gsd_cat'] == gsd_cat]
                    drop = drop.sample(n=count - backgrounds_per_gsd_cat)
                    for index, row in drop:
                        os.remove(os.path.join("substrates", row["substrate"], gsd_cat, row["background_id"]))
                    backgrounds.drop(drop.index)
        # place 1 instance of each species on each of the 2 background
        # images for each of 1 background types the until all 2
        # instances of each species have been placed 1 times
        # 2 * 1 * 2 = 4 images per gsd 16 total
        # add common column to merge by
        instances["temp"] = 1
        backgrounds["temp"] = 1
        dataset = pd.DataFrame()
        # want 5 copies of each bird
        for repeats in range(5):
            instances = instances.sample(n=len(instances))
            backgrounds = backgrounds.sample(n=len(backgrounds))
            rep = instances.merge(backgrounds, on='temp').drop('temp', axis=1)
            rep = rep[rep["instance_gsd_cat"] == rep["background_gsd_cat"]]
            rep = rep.sort_values(by=['substrate', 'instance_gsd_cat', 'species'])
            dataset = pd.concat([dataset, rep])
        dataset.to_csv("dataset.csv")
        # actually create data
        previous_background_path = "none"
        path = "dataset"
        if not os.path.exists(path):
            os.makedirs(path)
        dataset = dataset.reset_index(drop=True)
        for index, row in dataset.iterrows():
            background_path = os.path.join("substrates", row["substrate"], row["background_gsd_cat"], row["background_id"])
            instance_path = os.path.join("instances", row["species"], row["instance_gsd_cat"], row["instance_id"])
            if previous_background_path != background_path:
                print(background_path)
                if previous_background_path != "none":
                    md5hash = hashlib.md5(background.tobytes()).hexdigest()
                    image_name = md5hash + ".jpg"
                    label_name = md5hash + ".json"
                    background = background.convert("RGB")
                    background.save(os.path.join(path, image_name))
                    annotation = {
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": image_name,
                        "imageData": 'null',
                        "imageHeight": height,
                        "imageWidth": width}
                    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
                    with open(os.path.join(path, label_name), 'w') as annotation_file:
                        annotation_file.write(annotation_str)
                background = Image.open(background_path).convert("RGBA")
                width, height = background.size
                annotation = {""}
                previous_background_path = background_path
                # transforms
                background_width, background_height = background.size
                shapes = []
            instance = Image.open(instance_path).convert("RGBA")
            instance_width, instance_height = instance.size
            left = random.randint(0, background_width - instance_width)
            top = random.randint(0, background_height - instance_height)
            points = row["instance_points"]
            position = ((left, top))
            points = tuple(tuple(sum(x) for x in zip(a, position)) for a in points)
            # rotate and scale points too
            # transforms
            background.paste(instance, (left, top), instance)
            shapes.append({
                "label": row["species"],
                "points": points,
                "group_id": 'null',
                "shape_type": 'polygon',
                "flags": {}})
            # make annotation file

        #TODO make test set