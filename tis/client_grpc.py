
import numpy as np
import cv2

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


np.random.seed(123)
palette = np.random.randint(0, 256, (100, 3))



url = '10.128.61.8:8001'
#  url = '127.0.0.1:8001'
model_name = 'bisenetv1'
model_version = '1'
inp_name = 'raw_img_bytes'
outp_name = 'preds'
inp_dtype = 'UINT8'
outp_dtype = np.int64
impth = '../example.png'
mean = [0.3257, 0.3690, 0.3223] # city, rgb
std = [0.2112, 0.2148, 0.2115]


## input data and mean/std
inp_data = np.fromfile(impth, dtype=np.uint8)[None, ...]
mean = np.array(mean, dtype=np.float32)[None, ...]
std = np.array(std, dtype=np.float32)[None, ...]
inputs = [service_pb2.ModelInferRequest().InferInputTensor() for _ in range(3)]
inputs[0].name = inp_name
inputs[0].datatype = inp_dtype
inputs[0].shape.extend(inp_data.shape)
inputs[1].name = 'channel_mean'
inputs[1].datatype = 'FP32'
inputs[1].shape.extend(mean.shape)
inputs[2].name = 'channel_std'
inputs[2].datatype = 'FP32'
inputs[2].shape.extend(std.shape)
inp_bytes = [inp_data.tobytes(), mean.tobytes(), std.tobytes()]


option = [
        ('grpc.max_receive_message_length', 1073741824),
        ('grpc.max_send_message_length', 1073741824),
        ]
channel = grpc.insecure_channel(url, options=option)
grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)


metadata_request = service_pb2.ModelMetadataRequest(
    name=model_name, version=model_version)
metadata_response = grpc_stub.ModelMetadata(metadata_request)
print(metadata_response)

config_request = service_pb2.ModelConfigRequest(
        name=model_name,
        version=model_version)
config_response = grpc_stub.ModelConfig(config_request)
print(config_response)


request = service_pb2.ModelInferRequest()
request.model_name = model_name
request.model_version = model_version

request.ClearField("inputs")
request.ClearField("raw_input_contents")
request.inputs.extend(inputs)
request.raw_input_contents.extend(inp_bytes)


# sync
#  resp = grpc_stub.ModelInfer(request)
# async
resp = grpc_stub.ModelInfer.future(request)
resp = resp.result()

outp_bytes = resp.raw_output_contents[0]
outp_shape = resp.outputs[0].shape

out = np.frombuffer(outp_bytes, dtype=outp_dtype).reshape(*outp_shape).squeeze()

out = palette[out]
cv2.imwrite('res.png', out)
