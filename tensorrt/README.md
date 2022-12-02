

## Deploy with Tensorrt 

Firstly, We should export our trained model to onnx model:  
```
$ cd BiSeNet/
$ python tools/export_onnx.py --config configs/bisenetv2_city.py --weight-path /path/to/your/model.pth --outpath ./model.onnx --aux-mode eval
```

**NOTE:** I use cropsize of `1024x2048` here in my example, you should change it according to your specific application. The inference cropsize is fixed from this step on, so you should decide the inference cropsize when you export the model here.  

Then we can use either c++ or python to compile the model and run inference.  


### Using C++

#### 1. My platform

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver newer than 450.80
* cuda 11.3, cudnn 8
* cmake 3.22.0
* opencv built from source
* tensorrt 8.2.5.1



#### 2. Build with source code
Just use the standard cmake build method:  
```
mkdir -p tensorrt/build
cd tensorrt/build
cmake ..
make
```
This would generate a `./segment` in the `tensorrt/build` directory.


#### 3. Convert onnx to tensorrt model
If you can successfully compile the source code, you can parse the onnx model to tensorrt model with one of the following commands.   
For fp32, command is:
```
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt
```
If your gpu support acceleration with fp16 inferenece, you can add a `--fp16` option to in this step:  
```
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --fp16
```
Building an int8 engine is also supported. Firstly, you should make sure your gpu support int8 inference, or you model will not be faster than fp16/fp32. Then you should prepare certain amount of images for int8 calibration. In this example, I use train set of cityscapes for calibration. The command is like this:  
```
$ rm calibrate_int8 # delete this if exists
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --int8 /path/to/BiSeNet/datasets/cityscapes /path/to/BiSeNet/datasets/cityscapes/train.txt
```
With the above commands, we will have an tensorrt engine named `saved_model.trt` generated.  

Note that I use the simplest method to parse the command line args, so please do **Not** change the order of the args in above command.  


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


#### 6. Tips:  
1. ~Since tensorrt 7.0.0 cannot parse well the `bilinear interpolation` op exported from pytorch, I replace them with pytorch `nn.PixelShuffle`, which would bring some performance overhead(more flops and parameters), and make inference a bit slower. Also due to the `nn.PixelShuffle` op, you **must** export the onnx model with input size to be *n* times of 32.~   
If you are using 7.2.3.4 or newer versions, you should not have problem with `interpolate` anymore.

2. ~There would be some problem for tensorrt 7.0.0 to parse the `nn.AvgPool2d` op from pytorch with onnx opset11. So I use opset10 to export the model.~  
Likewise, you do not need to worry about this anymore with version newer than 7.2.3.4.

3. The speed(fps) is tested on a single nvidia Tesla T4 gpu with `batchsize=1` and `cropsize=(1024,2048)`. Please note that T4 gpu is almost 2 times slower than 2080ti, you should evaluate the speed considering your own platform and cropsize. Also note that the performance would be affected if your gpu is concurrently working on other tasks. Please make sure no other program is running on your gpu when you test the speed.

4. On my platform, after compiling with tensorrt, the model size of bisenetv1 is 29Mb(fp16) and 128Mb(fp32), and the size of bisenetv2 is 16Mb(fp16) and 42Mb(fp32). However, the fps of bisenetv1 is 68(fp16) and 23(fp32), while the fps of bisenetv2 is 59(fp16) and 21(fp32). It is obvious that bisenetv2 has fewer parameters than bisenetv1, but the speed is otherwise. I am not sure whether it is because tensorrt has worse optimization strategy in some ops used in bisenetv2(such as depthwise convolution) or because of the limitation of the gpu on different ops. Please tell me if you have better idea on this.  

5. int8 mode is not always greatly faster than fp16 mode. For example, I tested with bisenetv1-cityscapes and tensorrt 8.2.5.1. With v100 gpu and driver 515.65, the fp16/int8 fps is 185.89/186.85, while with t4 gpu and driver 450.80, it is 78.77/142.31. 


### Using python

You can also use python script to compile and run inference of your model.  


#### 1. Compile model to onnx

With this command: 
```
$ cd BiSeNet/tensorrt
$ python segment.py compile --onnx /path/to/model.onnx --savepth ./model.trt --quant fp16/fp32
```

This will compile onnx model into tensorrt serialized engine, save save to `./model.trt`.  


#### 2. Inference with Tensorrt

Run Inference like this:  
```
$ python segment.py run --mdpth ./model.trt --impth ../example.png --outpth ./res.png
```

This will use the tensorrt model compiled above, and run inference with the example image.  

