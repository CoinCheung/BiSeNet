
### My platform

* raspberry pi 3b
* armv8 4core cpu, 1G Memroy
* 2022-04-04-raspios-bullseye-armhf-lite.img 



### Install ncnn

#### 1. dependencies  
```
$ python -m pip install onnx-simplifier
```

#### 2. build ncnn  
Just follow the ncnn official tutoral of [build-for-linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux) to install ncnn. Following steps are all carried out on my raspberry pi:  

**step 1:** install dependencies  
```
$ sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libopencv-dev
```

**step 2:** (optional) install vulkan  

**step 3:** build   
I am using commit `5725c028c0980efd`, and I have not tested over other commits.  
```
$ git clone https://github.com/Tencent/ncnn.git
$ cd ncnn
$ git reset --hard 5725c028c0980efd
$ git submodule update --init
$ mkdir -p build
$ cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake ..
$ make -j2
$ make install 
```

### Convert model, build and run the demo

#### 1. convert pytorch model to ncnn model via onnx  
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
$ cd BiSeNet/ncnn/
$ mkdir -p models
$ mv model_v2_sim.param models/
$ mv model_v2_sim.bin models/
```

#### 2. compile demo code  
On raspberry device:  
```
$ mkdir -p BiSeNet/ncnn/build
$ cd BiSeNet/ncnn/build
$ cmake .. -DNCNN_ROOT=/path/to/ncnn/build/install
$ make
```

#### 3. run demo  
```
./segment
```
