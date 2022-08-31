

import argparse
import sys
import numpy as np
import cv2
import gevent.ssl

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


np.random.seed(123)
palette = np.random.randint(0, 256, (100, 3))


url = '10.128.61.8:8000'
#  url = '127.0.0.1:8000'
model_name = 'preprocess_cpp'
model_version = '1'
inp_name = 'raw_img_bytes'
outp_name = 'processed_img'
inp_dtype = 'UINT8'
impth = '../example.png'
mean = [0.3257, 0.3690, 0.3223] # city, rgb
std = [0.2112, 0.2148, 0.2115]


## prepare image and mean/std
inp_data = np.fromfile(impth, dtype=np.uint8)[None, ...]
mean = np.array(mean, dtype=np.float32)[None, ...]
std = np.array(std, dtype=np.float32)[None, ...]
inputs = []
inputs.append(httpclient.InferInput(inp_name, inp_data.shape, inp_dtype))
inputs.append(httpclient.InferInput('channel_mean', mean.shape, 'FP32'))
inputs.append(httpclient.InferInput('channel_std', std.shape, 'FP32'))
inputs[0].set_data_from_numpy(inp_data, binary_data=True)
inputs[1].set_data_from_numpy(mean, binary_data=True)
inputs[2].set_data_from_numpy(std, binary_data=True)

## client
triton_client = httpclient.InferenceServerClient(
        url=url, verbose=False, concurrency=32)

## infer
# sync
#  results = triton_client.infer(model_name, inputs)


# async
#  results = triton_client.async_infer(
#      model_name,
#      inputs,
#      outputs=None,
#      query_params=None,
#      headers=None,
#      request_compression_algorithm=None,
#      response_compression_algorithm=None)
#  results = results.get_result() # async infer only


## dynamic batching, this is not allowed, since different pictures has different raw size
results = []
for i in range(10):
    r = triton_client.async_infer(
        model_name,
        inputs,
        outputs=None,
        query_params=None,
        headers=None,
        request_compression_algorithm=None,
        response_compression_algorithm=None)
    results.append(r)
for i in range(10):
    results[i].get_result()
results = results[i]


# get output
outp = results.as_numpy(outp_name).squeeze()
print(outp.shape)
