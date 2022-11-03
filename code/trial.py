import math
from PIL import Image, ImageDraw, ImageEnhance
import json
import os
import torchvision.transforms as T
import numpy
import piexif

def crop_mask(im, points):
    polygon = [tuple(l) for l in points]
    pad = 0
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
    return [newIm, polygon]

def is_float(string):
    try:
        string = float(string)
        return string
    except: return string
# image = T.RandomRotation(degrees=(0, 360))(orig_img)
image = Image.open("datasets/trial/1c378c3af7c14a913a0aed1e5583a660.JPG")
annotation = json.load(open("datasets/trial/1c378c3af7c14a913a0aed1e5583a660.json"))
points = annotation["shapes"][0]["points"]
instance, points = crop_mask(image.convert("RGBA"), points)
instance_width, instance_height = instance.size
centre = (instance_width/2, instance_height/2)
instance = T.RandomHorizontalFlip(1)(instance)
print(points)
points = tuple(tuple([2 * centre[0] - point[0], point[1]]) for point in points)
print(" ")
print(centre[0])
print(" ")
print(points)
background = Image.open("datasets/trial/0a3d2b2ca5423b2e9ebd67c207a59a84.JPG")
width, height = background.size
# read exif data
# instance_exif_dict = piexif.load(image.info['exif'])
# instance_comments = json.loads("".join(map(chr, [i for i in instance_exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
# instance_gsd = is_float(instance_comments["gsd"])
# print("instance_gsd: ", instance_gsd)
#background
# background_exif_dict = piexif.load(image.info['exif'])
# background_comments = json.loads("".join(map(chr, [i for i in background_exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
# background_gsd = is_float(background_comments["gsd"])
# print("background_gsd: ", background_gsd)
# set scale equal
name = "blong"
# desired_gsd = 0.005
# instance_downscale = desired_gsd/instance_gsd
# instance_size = min(instance.size) / instance_downscale
# instance_005 = T.Resize(size=int(instance_size))(instance)
# instance_005.save("trial/instance_005.png")
# desired_gsd = 0.0075
# instance_downscale = desired_gsd/instance_gsd
# instance_size = min(instance.size) / instance_downscale
# instance_0075 = T.Resize(size=int(instance_size))(instance)
# instance_0075.save("trial/instance_0075.png")
# desired_gsd = 0.01
# instance_downscale = desired_gsd/instance_gsd
# instance_size = min(instance.size) / instance_downscale
# instance_01 = T.Resize(size=int(instance_size))(instance)
# instance_01.save("trial/instance_01.png")
# print("new instance size: ", instance.size)
# instance.save("trial/instance.png")
# points = tuple(tuple(item / instance_downscale for item in point) for point in points)
# background_downscale = desired_gsd/background_gsd
# background_size = min(background.size) / background_downscale
# print("original background size: ", background.size)
# background = T.Resize(size=int(background_size))(background)
# print("new background size: ", background.size)

# def rotate_point(point, centre, deg):
#     rotated_point = (
#         centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
#         centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg)))
#     return rotated_point

# instance_width, instance_height = instance.size
# centre = (instance_width / 2, instance_height / 2)
# rotation = 0
# rotation = 90
# rotation = 180
# rotation = 270
# instance = T.RandomRotation((rotation, rotation))(instance)
# instance = instance.rotate(rotation)
# points = tuple(rotate_point(point, centre, -rotation) for point in points)

# colour = 1.1
# instance = ImageEnhance.Color(instance)
# instance = instance.enhance(colour)
# contrast = 1.1
# instance = ImageEnhance.Contrast(instance)
# instance = instance.enhance(contrast)
# brightness = 1.1
# instance = ImageEnhance.Brightness(instance)
# instance = instance.enhance(brightness)

background.paste(instance, (200, 200), instance)
position = ((200, 200))
points = tuple(tuple(sum(x) for x in zip(a, position)) for a in points)
background.save("trial/" + name + ".jpg")

shapes = []
shapes.append({
    "label": "Bird",
    "points": points,
    "group_id": 'null',
    "shape_type": 'polygon',
    "flags": {}})

annotation = {
    "version": "5.0.1",
    "flags": {},
    "shapes": shapes,
    "imagePath": name + ".jpg",
    "imageData": 'null',
    "imageHeight": height,
    "imageWidth": width}
annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
with open("trial/" + name + ".json", 'w') as annotation_file:
    annotation_file.write(annotation_str)