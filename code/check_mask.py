from PIL import Image
import numpy as np
import sys
# mask = Image.open('../../datasets/PennFudanPed/masks-1/FudanPed00001_mask.png')
mask = Image.open('../../datasets/seed-detector/mask/crops/1fdba3e15e1ce393a5e79fb7f5bd4d00-mask.png')
# mask = Image.open('../../datasets/seed-detector/mask/masks/Pan - 1-mask.png')

# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's add a color palette to the mask.
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
    138, 43, 226, # index 4 is purple
    # 127, 255, 0, # index 5 is green
])
mask.show()
# numpy_mask = np.array(mask)
# np.set_printoptions(threshold=sys.maxsize)
# print(np.unique(numpy_mask))