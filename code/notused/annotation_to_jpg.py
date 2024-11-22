from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torch
from PIL import Image

import os
import json
import colorsys

root = "C:/Users/uqjwil54/Documents/Projects/trial"
# Load the annotation
annotation_path = os.path.join(root, "268e843e8bdaabb6f265b1ae3be75ce2.json")
annotation = json.load(open(annotation_path))

# Load the image
imagepath = os.path.join(root, annotation["imagePath"])
image = Image.open(imagepath)

# Generate distinct colours
def generate_distinct_colors(n):
    hues = [i / n for i in range(n)]
    saturation = 0.8
    value = 0.8
    colors = []
    for hue in hues:
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_color = (int(r * 255), int(g * 255), int(b * 255))
        colors.append(rgb_color)
    return colors

boxes = []
labels = []
colours = []
for shape in annotation["shapes"]:
    # Get the instance shape_type
    shapetype = shape["shape_type"]

    # Get the instance points
    points = tuple((point[0], point[1]) for point in shape["points"])

    # Convert rectangle to points format
    if shapetype == "rectangle":
        points = tuple((
            (points[0][0], points[0][1]),
            (points[1][0], points[0][1]),
            (points[1][0], points[1][1]),
            (points[0][0], points[1][1])))
    
    # Create rectangle
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    boxes.append([min_x, min_y, max_x, max_y])
    labels.append(shape["label"])

# Make colour list
unique_labels = list(set(labels))
distinct_colours = generate_distinct_colors(len(unique_labels))

colours = []
for label in labels:
    colours.append(distinct_colours[unique_labels.index(label)])

# Convert lists to tensors
image = F.to_tensor(image)
image = (image * 255).to(torch.uint8)
boxes = torch.Tensor(boxes)

# Create labelled image
labelled_image = draw_bounding_boxes(image, boxes, colors = colours, width = 2)
labelled_image = F.to_pil_image(labelled_image)
labelled_image.save(os.path.join(root, "trial.jpg"))