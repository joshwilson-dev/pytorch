import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000
root = "C:/Users/uqjwil54/OneDrive - The University of Queensland/DBBS/Surveys/2024-03-12_1646_Adams Beach/T5"
name = "2024_03_12-1646-Adams_Beach-T5"
im = Image.open(os.path.join(root, name + ".tif")).convert("RGB")
print("Generating png for %s" % name)
im.save(os.path.join(root, name + ".png"), quality=100)