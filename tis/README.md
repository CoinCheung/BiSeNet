

## A simple demo of using trition-inference-serving

### Platform

* ubuntu 18.04
* cmake-3.22.0
* 8 Tesla T4 gpu 


### Serving Model

#### 1. prepare model repository

We need to export our model to onnx and copy it to model repository:
```
$ cd BiSeNet
$ python tools/export_onnx.py --config configs/bisenetv1_city.py --weight-path /path/to/your/model.pth --outpath ./model.onnx 
$ cp -riv ./model.onnx tis/models/bisenetv1/1

$ python tools/export_onnx.py --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model.onnx 
$ cp -riv ./model.onnx tis/models/bisenetv2/1
```

#### 2. prepare the preprocessing backend
We can use either python backend or cpp backend for preprocessing in the server side.  
Firstly, we pull the docker image, and start a serving container:  
```
$ docker pull nvcr.io/nvidia/tritonserver:22.07-py3
$ docker run -it --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/BiSeNet/tis/models:/models -v /path/to/BiSeNet/:/BiSeNet nvcr.io/nvidia/tritonserver:21.10-py3 bash
```
From here on, we are in the container environment. Let's prepare the backends in the container:  
```
# ln -s /usr/local/bin/pip3.8 /usr/bin/pip3.8
# /usr/bin/python3 -m pip install pillow
# apt update && apt install rapidjson-dev libopencv-dev
```
Then we download cmake 3.22 and unzip in the container, we use this cmake 3.22 in the following operations.  
We compile c++ backends:   
```
# cp -riv /BiSeNet/tis/self_backend /opt/tritonserver/backends
# chmod 777 /opt/tritonserver/backends/self_backend
# cd /opt/tritonserver/backends/self_backend
# mkdir -p build && cd build
# cmake .. && make -j4
# mv -iuv libtriton_self_backend.so ..
```
Utils now, we should have backends prepared.



#### 3. start service
We start the server in the docker container, following the above steps:  
```
# tritonserver --model-repository=/models 
```
In general, the service would start now. You can check whether service has started by:  
```
$ curl -v localhost:8000/v2/health/ready
```

By default, we use gpu 0 and gpu 1, you can change configurations in the `config.pbtxt` file.


### Request with client

We call the model service with both python and c++ method.  

From here on, we are at the client machine, rather than the server docker container.  


#### 1. python method

Firstly, we need to install dependency package:  
```
$ python -m pip install tritonclient[all]==2.15.0
```

Then we can run the script for both http request and grpc request: 
```
$ cd BiSeNet/tis
$ python client_http.py  # if you want to use http client
$ python client_grpc.py  # if you want to use grpc client
```

This would generate a result file named `res.jpg` in `BiSeNet/tis` directory.


#### 2. c++ method

We need to compile c++ client library from source: 
```
$ apt install rapidjson-dev
$ mkdir -p /data/ $$ cd /data/
$ git clone https://github.com/triton-inference-server/client.git
$ cd client && git reset --hard da04158bc094925a56b
$ mkdir -p build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=/opt/triton_client -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=OFF -DTRITON_ENABLE_PYTHON_HTTP=OFF -DTRITON_ENABLE_PYTHON_GRPC=OFF -DTRITON_ENABLE_JAVA_HTTP=OFF -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=OFF -DTRITON_ENABLE_TESTS=ON ..
$ make cc-clients
```
The above commands are exactly what I used to compile the library. I learned these commands from the official document.

Also, We need to install `cmake` with version `3.22`.

Optionally, I compiled opencv from source and install it to `/opt/opencv`. You can first skip this and see whether you meet problems. If you have problems about opencv in the following steps, you can compile opencv as what I do.

After installing the dependencies, we can compile our c++ client:
```
$ cd BiSeNet/tis/cpp_client
$ mkdir -p build && cd build
$ cmake .. && make
```

Finally, we run the client and see a result file named `res.jpg` generated:
```
    ./client
```


### In the end

This is a simple demo with only basic function. There are many other features that is useful, such as shared memory and dynamic batching. If you have interests on this, you can learn more in the official document.
