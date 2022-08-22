
import os
import onnx
from onnx import AttributeProto, TensorProto, GraphProto, ModelProto, OptionalProto, TypeProto, SequenceProto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add NMS
onnx_path ="models/model1/model_final_state_dict.onnx"
onnx_model = onnx.load(onnx_path)

# check names
output = onnx_model.graph.output
print(output)
# make constant tensors
score_threshold = onnx.helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.9])
iou_threshold = onnx.helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.3])
max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [200])

# create the NMS node
inputs=['output', '4440', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
outputs = ["selected_indices"]
nms_node = onnx.helper.make_node('NonMaxSuppression', inputs, ["selected_indices"],)

# add to the list of graph nodes
onnx_model.graph.node.append(nms_node)

# append to the output (now the outputs would be scores, bboxes, selected_indices)
output_value_info = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape=[])
onnx_model.graph.output.append(output_value_info)

# add to initializers - without this, onnx will not know where these came from, and complain that 
# they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated 
# as constants needed for the NMS op
onnx_model.graph.initializer.append(score_threshold)
onnx_model.graph.initializer.append(iou_threshold)
onnx_model.graph.initializer.append(max_output_boxes_per_class)

# check that it works and re-save
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "models/model1/model_final_state_dict_nms.onnx")