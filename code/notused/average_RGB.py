import os
import json
from statistics import mean

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

def crop_mask(image, points):

    # Find the bounding box
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # Calculate crop size and center coordinates
    width = max_x - min_x
    height = max_y - min_y
    size = max(width, height)
    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Crop the image
    crop_left = int(center_x - size / 2)
    crop_upper = int(center_y - size / 2)
    crop_right = int(center_x + size / 2)
    crop_lower = int(center_y + size / 2)
    bird = image.crop((crop_left, crop_upper, crop_right, crop_lower)).convert('RGBA')

    # Adjust polygon to crop origin
    origin = [(center_x - size / 2, center_y - size / 2)] * len(points)
    points = tuple(tuple(a - b for a, b in zip(t1, t2)) for t1, t2 in zip(points, origin))

    # Create mask image
    mask_im = Image.new('L', bird.size, 0)
    ImageDraw.Draw(mask_im).polygon(points, outline=1, fill=1)
    mask = np.array(mask_im)

    # Combine image and mask
    new_im_array = np.array(bird)
    new_im_array[:, :, 3] = mask * 255

    # Create final image
    mask = Image.fromarray(new_im_array, 'RGBA')
    return mask

data = {"category": [], "R": [], "G": [], "B": [], "area": []}
root = "data/"
for entry in os.scandir(os.path.join(root,"consolidated")):
    if entry.path.endswith(".json"):
        print("Recording Patch Data From: ", entry.path)

        # Load the annotation
        annotation = json.load(open(entry.path))

        # Load the image
        imagepath = os.path.join(root, "consolidated", annotation["imagePath"])
        image = Image.open(imagepath)

        # Get GSD
        gsd = annotation["gsd"]

        # Check for polygons
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

            # Crop instances
            instance = crop_mask(image, points)
            
            # Calculate average R G B values
            R, G, B = [mean(instance.getdata(band)) for band in range(3)]

            # Get the label
            label = json.loads(shape["label"].replace("'", '"'))
            category = label["name"] + " - " + label ["age"]

            # Get number of pixels on bird and convert to area
            alpha_channel = instance.getchannel('A')
            non_zero_pixels = alpha_channel.getbbox()[2] * alpha_channel.getbbox()[3]

            # Convert to area
            area = non_zero_pixels * gsd**2

            # Save results
            data["category"].append(category)
            data["R"].append(R)
            data["G"].append(G)
            data["B"].append(B)
            data["area"].append(area)

# Average results
data = pd.DataFrame(data=data)
data_avg = data.groupby('category').agg('mean')

# Cluster
from sklearn.cluster import KMeans
# Number of clusters (k)
num_clusters = 5

# Perform k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data_avg[['R', 'G', 'B']])

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Get the labels for each pixel
data_avg['cluster'] = kmeans.labels_
for index, row in data_avg.iterrows():
    # category = row["category"]
    category = index
    R = row["R"]
    G = row["G"]
    B = row["B"]
    print("{}: \t{}, {}, {}".format(category, R, G, B))