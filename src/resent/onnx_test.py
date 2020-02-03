import numpy as np
import onnx
import onnxruntime as rt

# create input data
input_data = np.ones((1, 3, 224, 224), dtype=np.float32)
# create runtime session
# model = onnx.load("../../models/onnx/resnet18.onnx")
# out = model(input_data)
#
#
sess = rt.InferenceSession("../../models/onnx/resnet18.onnx")
# get output name
output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
input_name = sess.get_inputs()[0].name
# forward model
res = sess.run([output_name], {"input.1": input_data})
out = np.array(res)
print(out)

