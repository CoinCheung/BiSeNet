

## A demo of using openvino to deploy

Openvino is used to deploy model on intel cpus or "gpu inside cpu".  

My platform:  
* Ubuntu 18.04
* Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
* openvino_2021.4.689


### preparation

1.Train the model and export it to onnx  
```
$ cd BiSeNet/
$ python tools/export_onnx.py --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model_v2.onnx 
```
(Optional) 2.Install 'onnx-simplifier' to simplify the generated onnx model:
```
$ python -m pip install onnx-simplifier
$ python -m onnxsim model_v2.onnx model_v2_sim.onnx
```


### Install and configure openvino

1.pull docker image  
```
$ docker pull openvino/ubuntu18_dev
```

2.start a docker container and mount code into it  
```
$ docker run -itu root -v /path/to/BiSeNet:/BiSeNet openvino/ubuntu18_dev --device /dev/dri:/dev/dri bash

```
If your cpu does not have intel "gpu inside of cpu" or you do not want to use it, you can remove the option of `--device /dev/dri:/dev/dri`.  

After running the above command, you will be in the container.  

(optional) 3.install gpu dependencies  
If you want to use gpu, you also need to install some dependencies inside the container:
```
# mkdir -p /tmp/opencl && cd /tmp/opencl
# useradd -ms /bin/bash -G video,users openvino
# chown openvino -R /home/openvino
# apt update
# apt install -y --no-install-recommends ocl-icd-libopencl1
# curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb" --output "intel-gmmlib_19.3.2_amd64.deb" 
# curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb" --output "intel-igc-core_1.0.2597_amd64.deb" 
# curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb" --output "intel-igc-opencl_1.0.2597_amd64.deb"
# curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb" --output "intel-opencl_19.41.14441_amd64.deb" 
# curl -L "https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.41.14441_amd64.deb" --output "intel-ocloc_19.04.12237_amd64.deb" 
# dpkg -i /tmp/opencl/*.deb
# apt --fix-broken install
# ldconfig
```

I got the above commands from the official docs but I did not test it since my cpu does not have integrated gpu.  

You can check if your platform has intel gpu with this command:  
```
$ sudo lspci | grep -i vga
```

4.configure environment  
just run this script, and the environment would be ready:  
```
# source /opt/intel/openvino_2021.4.689/bin/setupvars.sh
```


### convert model and run demo

1.convert onnx to openvino IR  
In the docker container:  
```
# cd /opt/intel/openvino_2021.4.689/deployment_tools/model_optimizer
# python3 mo.py --input_model /BiSeNet/model_v2.onnx --output_dir /BiSeNet/openvino/output_v2
```

2.compile and run the demo
```
# cd /BiSeNet/openvino
# mkdir -p build && cd build
# cmake .. && make
# ./segment
```
After this, you will see a segmentation result image named `res.jpg` generated.



### Tipes

1. GPU support: openvino supports intel cpu and intel "gpu inside cpu". Until now(2021.11), other popular isolated gpus are not supported, such as nvidia/amd gpus. Also, other integrated gpus are not supported, such as aspeed graphics family.

2. About low-precision: precision is optimized automatically, and the model will be run in one or several precision mode. We can also manually enforce to use bf16, as long as our cpu have `avx512_bf16` supports. If cpu does not support bf16, it will use simulation which would slow down the inference. If neither native bf16 nor simulation is supported, an error would occur.
