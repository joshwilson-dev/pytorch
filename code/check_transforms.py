from PIL import Image

import torch
import torchvision.transforms as T
import json
import piexif
import random
import hashlib

torch.manual_seed(0)
dir = "../datasets/trial/"
orig_img = Image.open(dir + "0c3dab86ddf2ab59dad1241d1a2c218f.jpg")
# orig_img.show()
width, height = orig_img.size
if height > width: pad = [int((height - width) / 2), 0]
else: pad = [0, int((width - height)/2)]
# transformed_img = T.Pad(padding=pad)(orig_img)
exif_dict = piexif.load(orig_img.info['exif'])

# check if XPcomment tag already contains required metrics
comments = json.loads("".join(map(chr, [i for i in exif_dict["0th"][piexif.ImageIFD.XPComment] if i != 0])))
gsd = float(comments["gsd"])

gsd_range = [0, 0.01, 0.02]
for gsd_index in range(len(gsd_range) - 1):
    gsd_min = gsd_range[gsd_index]
    gsd_max = gsd_range[gsd_index + 1]
    if gsd < gsd_min:
        print("Downscale Image")
        gsd_req = random.uniform(gsd_min, gsd_max)
        # scale_req = gsd/gsd_req
        scale_req = 0.5
        print(scale_req)
        width, height = orig_img.size
        print(width, height)
        comments["gsd"] = str(gsd_req)
        comments["image_width"] = str(width*scale_req)
        comments["image_length"] = str(width*scale_req)
        exif_dict["0th"][piexif.ImageIFD.XPComment] = json.dumps(comments).encode('utf-16le')
        exif_bytes = piexif.dump(exif_dict)
        size = int(scale_req*min(width, height))
        print(size)
        transformed_img = T.Resize(size=size)(orig_img)
        annotation = json.load(open(dir + "0c3dab86ddf2ab59dad1241d1a2c218f.json"))
        for shape_index in range(len(annotation["shapes"])):
            for point_index in range(len(annotation["shapes"][shape_index]["points"])):
                annotation["shapes"][shape_index]["points"][point_index][0] *= scale_req
                annotation["shapes"][shape_index]["points"][point_index][1] *= scale_req
        md5hash = hashlib.md5(transformed_img.tobytes()).hexdigest()
        print(md5hash)
        transformed_img.save(dir + md5hash + ".JPG", exif = exif_bytes)
        annotation_output = dir + md5hash + ".json"
        annotation["imagePath"] = md5hash + ".JPG"
        with open(annotation_output, 'w') as new_annotation:
            json.dump(annotation, new_annotation, indent=2)

