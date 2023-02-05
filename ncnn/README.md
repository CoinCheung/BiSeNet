
### My platform  

* raspberry pi 3b
* 2022-04-04-raspios-bullseye-armhf-lite.img 
* cpu: 4 core armv8, memory: 1G 



### Install ncnn  

Just follow the ncnn official tutoral of [build-for-linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux) to install ncnn. Following steps are all carried out on my raspberry pi:  

**step 1:** install dependencies  
```
$ sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libopencv-dev
```

**step 2:** (optional) install vulkan  

**step 3:** build   
I am using commit `6869c81ed3e7170dc0`, and I have not tested over other commits.  
```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git reset --hard 6869c81ed3e7170dc0
$ git submodule update --init
$ mkdir -p build
$ cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake ..
$ make -j2
$ make install 
```

### Convert pytorch model to ncnn model   

#### 1. dependencies  
```
$ python -m pip install onnx-simplifier
```

#### 2. convert pytorch model to ncnn model via onnx  
On your training platform:
```
$ cd BiSeNet/
$ python tools/export_onnx.py --aux-mode eval --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model_v2.onnx 
$ python -m onnxsim model_v2.onnx model_v2_sim.onnx
```

Then copy your `model_v2_sim.onnx` from training platform to raspberry device.   

On raspberry device:  
```
$ /path/to/ncnn/build/tools/onnx/onnx2ncnn model_v2_sim.onnx model_v2_sim.param model_v2_sim.bin
```

You can optimize the ncnn model by fusing the layers and save the weights with fp16 datatype.   
On raspberry device:
```
$ /path/to/ncnn/build/tools/ncnnoptimize model_v2_sim.param model_v2_sim.bin model_v2_sim_opt.param model_v2_sim_opt.bin 65536
$ mv model_v2_sim_opt.param model_v2_sim.param
$ mv model_v2_sim_opt.bin model_v2_sim.bin
```

You can also quantize the model for int8 inference, following this [tutorial](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference). Make sure your device support int8 inference.  


### build and run the demo
#### 1. compile demo code  
On raspberry device:  
```
$ mkdir -p BiSeNet/ncnn/build
$ cd BiSeNet/ncnn/build
$ cmake .. -DNCNN_ROOT=/path/to/ncnn/build/install
$ make
```

#### 2. run demo  
```
./segment
```
