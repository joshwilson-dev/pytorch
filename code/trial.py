from PIL import Image
import random
instance = Image.open("./code/instance.png")

shadow = instance.copy()
width, height = shadow.size

shadow_data = []
for item in shadow.getdata():
    if item[3] == 255:
        shadow_data.append((0, 0, 0, int(200)))
    else:
        shadow_data.append(item)
shadow.putdata(shadow_data)
max_offset = 1.25
x_offset = random.randint(int(-width * max_offset), int(width * max_offset))
y_offset = random.randint(int(-height * max_offset), int(height * max_offset))
shadow = shadow.crop((min(x_offset, 0), min(y_offset, 0), max(width, width + x_offset), max(height, height + y_offset)))

shadow.paste(instance, (max(x_offset, 0), max(y_offset, 0)), instance)
instance = shadow.copy()
instance.show()