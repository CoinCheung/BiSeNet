

## Deploy with Tensorrt 

This works with tensorrt 11, where the network is strongly typed: the precision (fp32/fp16/int8) is decided when exporting the onnx model, not when compiling the tensorrt engine.  


### 1. Export onnx model

Install the dependencies first:  
```
$ python -m pip install --extra-index-url https://pypi.nvidia.com --upgrade tensorrt nvidia-modelopt[all]
$ python -m pip install onnx onnx_graphsurgeon onnxoptimizer
```

Then export the trained model:  
```
$ cd BiSeNet/
$ python tools/export_onnx.py --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model.onnx --input-size 1024 2048 --fuse-head
```

Options:  
* `--input-size H W`: the inference size, default is `1024 2048`. The input size is fixed from this step on, so you should decide it according to your application when you export the model. Input images of other sizes are resized to this size before inference.  
* `--fuse-head`: replace the final `upsample -> argmax` with one fused plugin, which is faster. The plugin source is in `tensorrt/plugins/fused_head`.  
* `--int8`: also export an int8 model, quantized with [modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer). By default it runs autotune to search for a faster quantization scheme, which takes about 3 hours and needs an idle gpu. Add `--int8-no-autotune` to skip it.  

The val set of the dataset in the config file is used for calibration, so make sure the dataset is prepared.  

With `--outpath ./model.onnx`, the following models are generated:  

| file | precision |
|---|---|
| `model_opt.onnx` | fp32 |
| `model_opt_fp16.onnx` | fp16 |
| `model_opt_int8.onnx` | int8, with `--int8` |
| `*_fused.onnx` | the above models with the fused head, with `--fuse-head` |

The model takes raw `uint8` images of shape `(N, H, W, 3)` in rgb order (`/255` and mean/std normalization are inside the model), and outputs `uint8` labels of shape `(N, H, W)`. The batch size `N` can be 1 to 4.  


### 2. Using C++

#### 1. My platform

* ubuntu 24.04
* nvidia L4 gpu, driver 595.91.07
* cuda 12.9
* cmake 3.28.3
* opencv
* tensorrt 11.3.0.99


#### 2. Build with source code
Just use the standard cmake build method:  
```
$ mkdir -p tensorrt/build
$ cd tensorrt/build
$ cmake ..
$ make
```
This would generate a `./segment` in the `tensorrt/build` directory, as well as `plugins/fused_head/libfused_head_plugin.so`. The fused head plugin is compiled into `./segment`, so you only need the `.so` file when you use other tools such as python or `trtexec --staticPlugins=/path/to/libfused_head_plugin.so`.  


#### 3. Convert onnx to tensorrt model
Compile the exported onnx model into a tensorrt engine:  
```
$ ./segment compile /path/to/model_opt_fp16_fused.onnx /path/to/saved_model.trt
```
There is no precision option, the precision of the engine is the same as the onnx model. You can append an optional batch size (1 to 4) which the engine is optimized for, default is 1:  
```
$ ./segment compile /path/to/model_opt_fp16_fused.onnx /path/to/saved_model.trt 2
```


#### 4. Infer with one single image
Run inference like this:   
```
$ ./segment run /path/to/saved_model.trt /path/to/input/image.jpg /path/to/saved_img.jpg
```


#### 5. Test speed  
The speed depends on the specific gpu platform you are working on, you can test the fps on your gpu like this:  
```
$ ./segment test /path/to/saved_model.trt
```
This prints the fps twice: without and with the host-device copy of input and output. You can also append a batch size (1 to 4), like `./segment test /path/to/saved_model.trt 4`.  


#### 6. Tips:  

1. The speed(fps) is tested with `batchsize=1` and `cropsize=(1024,2048)`, which might be different from your platform and settings. You should evaluate the speed considering your own platform and cropsize. Also note that the performance would be affected if your gpu is concurrently working on other tasks. Please make sure no other program is running on your gpu when you test the speed.  

2. Cuda graph and pinned host memory are used by default. You can disable them with environment variables `TRT_NO_CUDA_GRAPH=1` and `TRT_NO_PINNED=1`.  



### 3. Using python

You can also use python script to compile and run inference of your model. Besides tensorrt, it requires `cuda-python` and `opencv-python`:  
```
$ python -m pip install cuda-python opencv-python
```

The script loads the fused head plugin from `tensorrt/build/plugins/fused_head/libfused_head_plugin.so` by default, so build it as in the C++ part first, or set its path with `--plugin /path/to/libfused_head_plugin.so` (before the sub command).  


#### 1. Compile onnx to tensorrt engine

With this command: 
```
$ cd BiSeNet/tensorrt
$ python segment.py compile --onnx /path/to/model_opt_fp16_fused.onnx --savepth ./model.trt
```

This will compile onnx model into tensorrt serialized engine, and save it to `./model.trt`. Same as C++, the precision comes from the onnx model, and you can use `--opt-batch` to set the optimized batch size.  


#### 2. Inference with Tensorrt

Run Inference like this:  
```
$ python segment.py run --mdpth ./model.trt --impth ../example.png --outpth ./res.png
```

This will use the tensorrt model compiled above, and run inference with the example image.


#### 3. Test speed

```
$ python segment.py test --mdpth ./model.trt --batch 1
```
