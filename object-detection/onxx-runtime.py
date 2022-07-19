import onnxruntime
from PIL import Image, ImageDraw, ImageFont
import numpy
from matplotlib import pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ort_session = onnxruntime.InferenceSession("models/model1/model_final_state_dict.onnx")
# ort_session = onnxruntime.InferenceSession("models/model1/model_final_state_dict_nms.onnx")

idx_to_class = {
    1: "fertilised",
    2: "unfertilised"
}

idx_to_colour = {
    1: "orange",
    2: "blue"
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

make_prediction("../dataset/trial/0adddc65d9ea2ce40b48a0442b63875d.JPG")