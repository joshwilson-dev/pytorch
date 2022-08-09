import onnxruntime
from PIL import Image, ImageDraw, ImageFont
import numpy
from matplotlib import pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ort_session = onnxruntime.InferenceSession("./models/seed-detector/model_final_state_dict.onnx")
ort_session = onnxruntime.InferenceSession("./models/bird-species-detector/model_final_state_dict.onnx")

# idx_to_class = {
#     1: "fertilised",
#     2: "unfertilised"
# }

# idx_to_colour = {
#     1: "orange",
#     2: "blue"
# }

idx_to_class = {
    1: "masked lapwing",
    2: "silver gull",
    3: "black swan",
    4: "bar-tailed godwit",
    5: "gull-billed tern",
    6: "australian white ibis",
    7: "pacific black duck",
    8: "australian wood duck",
    9: "great knot",
    10: "torresian crow",
    11: "australasian swamphen",
    12: "hardhead",
    13: "pied stilt",
    14: "muscovy duck",
    15: "australian pelican",
    16: "royal spoonbill",
    17: "pied oystercatcher"
}

idx_to_colour = {
    1: "maroon",
    2: "blue",
    3: "black",
    4: "purple",
    5: "yellow",
    6: "orange",
    7: "white",
    8: "red",
    9: "brown",
    10: "gold",
    11: "pink",
    12: "grey",
    13: "slateblue",
    14: "cyan",
    15: "lime",
    16: "green",
    17: "peru"
}

def prepare_image(image_path):
    PIL_image = Image.open(image_path).convert('RGB')
    CHW_image = numpy.array(PIL_image)
    tensor_image = numpy.array(numpy.expand_dims(CHW_image.transpose((2, 0, 1))/255,0), numpy.float32)
    return CHW_image, tensor_image

def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def draw_boxes(image, boxes, labels, colours, width):
    img_to_draw = Image.fromarray(image)
    draw = ImageDraw.Draw(img_to_draw)
    txt_font = ImageFont.load_default()
    for i, bbox in enumerate(boxes):
        colour = colours[i]
        draw.rectangle(bbox, width=width, outline=colour)
        margin = width + 1
        draw.text((bbox[0] + margin, bbox[1] + margin), labels[i], fill=colour, font=txt_font)
    return img_to_draw

def make_prediction(image_path):
    CHW_image, tensor_image = prepare_image(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name:tensor_image}
    prediction = ort_session.run(None, ort_inputs)
    boxes = prediction[0]
    labels = prediction[1]
    scores = prediction[2]
    string_scores = ['{0:.2f}'.format(score) for score in scores]
    named_labels = [idx_to_class[i] for i in labels]
    colours = [idx_to_colour[i] for i in labels]
    named_labels_with_scores = [named_labels[i] + ": " + string_scores[i] for i in range(len(scores))]
    visualise = draw_boxes(image = CHW_image, boxes = boxes, labels = named_labels_with_scores, colours = colours, width=5)
    show(visualise)

make_prediction("../dataset/bird-species-detector/test/test-1.JPG")