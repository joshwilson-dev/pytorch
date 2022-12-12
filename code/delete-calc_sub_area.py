import math
from PIL import Image, ImageDraw
from PIL import ImagePath 
  
xy = [[100,0], [200,200], [0,200]]  
xy = [tuple(point) for point in xy]
image = ImagePath.Path(xy).getbbox()  
size = list(map(int, map(math.ceil, image[2:])))
  
img = Image.new("P", size, 0) 
img1 = ImageDraw.Draw(img)  
img1.polygon(xy, fill = 255)
left = 0
right = 100
top = 0
bottom = 100
bright_count = sum(img.point(lambda pix: 1 if pix==255 else 0).getdata())
print(bright_count)
print(img.size)
img = img.crop((left, top, right, bottom))
print(img.size)
bright_count = sum(img.point(lambda pix: 1 if pix==255 else 0).getdata())
print(bright_count)
# img.show()