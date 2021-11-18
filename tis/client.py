
import numpy as np
import cv2

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


np.random.seed(123)
palette = np.random.randint(0, 256, (100, 3))



#  url = '10.128.61.7:8001'
url = '127.0.0.1:8001'
model_name = 'bisenetv2'
model_version = '1'
inp_name = 'input_image'
outp_name = 'preds'
inp_dtype = 'FP32'
outp_dtype = np.int64
inp_shape = [1, 3, 1024, 2048]
outp_shape = [1024, 2048]
impth = '../example.png'
mean = [0.3257, 0.3690, 0.3223] # city, rgb
std = [0.2112, 0.2148, 0.2115]


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

inp = service_pb2.ModelInferRequest().InferInputTensor()
inp.name = inp_name
inp.datatype = inp_dtype
inp.shape.extend(inp_shape)


mean = np.array(mean).reshape(1, 1, 3)
std = np.array(std).reshape(1, 1, 3)
im = cv2.imread(impth)[:, :, ::-1]
im = cv2.resize(im, dsize=tuple(inp_shape[-1:-3:-1]))
im = ((im / 255.) - mean) / std
im = im[None, ...].transpose(0, 3, 1, 2)
inp_bytes = im.astype(np.float32).tobytes()

request.ClearField("inputs")
request.ClearField("raw_input_contents")
request.inputs.extend([inp,])
request.raw_input_contents.extend([inp_bytes,])


outp = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
outp.name = outp_name
request.outputs.extend([outp,])

# sync
#  resp = grpc_stub.ModelInfer(request).raw_output_contents[0]
# async
resp = grpc_stub.ModelInfer.future(request)
resp = resp.result().raw_output_contents[0]

out = np.frombuffer(resp, dtype=outp_dtype).reshape(*outp_shape)

out = palette[out]
cv2.imwrite('res.png', out)
