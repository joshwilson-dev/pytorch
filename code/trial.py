import math
from PIL import Image, ImageDraw
import json
import os
import torchvision.transforms as T
import numpy

def crop_mask(im, points):
    polygon = [tuple(l) for l in points]
    pad = 1000
    # find bounding box
    max_x = max([t[0] for t in polygon])
    min_x = min([t[0] for t in polygon])
    max_y = max([t[1] for t in polygon])
    min_y = min([t[1] for t in polygon])
    # subtract xmin andymin from each point
    origin = [(min_x - pad, min_y - pad)] * len(polygon)
    polygon = tuple(tuple(a - b for a, b in zip(tup1, tup2)) for tup1, tup2 in zip(polygon, origin))
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

image = Image.open("datasets/trial/mask/DJI_0064.JPG")
annotation = json.load(open("datasets/trial/mask/DJI_0064.json"))
points = annotation["shapes"][0]["points"]
instance, points = crop_mask(image.convert("RGBA"), points)
instance_width, instance_height = instance.size
# instance.show()
background = Image.open("datasets/trial/substrates/grass/very fine/bf7539fe5ff9cbef2fdc30e0eb1dc557.png")
width, height = background.size
# background.show()
downscale = 2
print(instance.size)
size = min(instance.size) / downscale
instance = T.Resize(size=int(size))(instance)
print(instance.size)
print(points)
points = tuple(tuple(item / downscale for item in point) for point in points)

def rotate_point(point, centre, deg):
    rotated_point = (
        centre[0] + (point[0]-centre[0])*math.cos(math.radians(deg)) - (point[1]-centre[1])*math.sin(math.radians(deg)),
        centre[1] + (point[0]-centre[0])*math.sin(math.radians(deg)) + (point[1]-centre[1])*math.cos(math.radians(deg)))
    return rotated_point
instance_width, instance_height = instance.size
centre = (instance_width / 2, instance_height / 2)
rotation = 20
instance = T.RandomRotation((rotation, rotation), expand = False)(instance)
# instance = instance.rotate(rotation)
points = tuple(rotate_point(point, centre, -rotation) for point in points)
background.paste(instance, (200, 200), instance)
position = ((200, 200))
points = tuple(tuple(sum(x) for x in zip(a, position)) for a in points)
background.save("trial/trial20.png")

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
    "imagePath": "trial20.png",
    "imageData": 'null',
    "imageHeight": height,
    "imageWidth": width}
annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
with open("trial/trial20.json", 'w') as annotation_file:
    annotation_file.write(annotation_str)