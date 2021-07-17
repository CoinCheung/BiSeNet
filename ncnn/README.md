
### My platform

* ubuntu 18.04
* Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
* cmake 3.17.1
* opencv built from source

### NOTE

Though this demo runs on x86 platform, you can also use it on mobile platforms. NCNN is better optimized on mobile platforms.


### Install ncnn

1. dependencies
```
$ python -m pip install onnx-simplifier
```

2. build ncnn
Just following the ncnn official tutoral: [build-for-linux](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux) to install ncnn:

1) dependencies
```
# apt install build-essential git libprotobuf-dev protobuf-compiler 
```

2) (optional) install vulkan

3) install opencv from source

4) build 
```
## I am using commit 9391fae741a1fb8d58cdfdc92878a5e9800f8567, and I have not tested over newer commits
$ git clone https://github.com/Tencent/ncnn.git
$ $cd ncnn
$ git submodule update --init
$ mkdir -p build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/host.gcc.toolchain.cmake ..
$ make -j
$ make install 
```

### convert model, build and run the demo

1. convert pytorch model to ncnn model via onnx
```
$ cd BiSeNet/
$ python tools/export_onnx.py --aux-mode eval --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model_v2.onnx 
$ python -m onnxsim model_v2.onnx model_v2_sim.onnx
$ /path/to/ncnn/build/tools/onnx/onnx2ncnn model_v2_sim.onnx model_v2_sim.param model_v2_sim.bin
$ mv model_v2_sim.param ncnn/models 
$ mv model_v2_sim.bin ncnn/models 
```

2. compile demo code
```
cd ncnn/build
cmake .. -DNCNN_ROOT=/path/to/ncnn/build/install
make
```

3. run demo
```
./segment
```
