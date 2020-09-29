
### My platform

* ubuntu 18.04
* nvidia Tesla T4 gpu, driver 450.51.05
* cuda 10.2, cudnn 7
* cmake 3.10.2
* opencv built from source
* tensorrt 7.0.0


### Export model to onnx
I export the model like this:  
```
$ python tools/export_onnx.py --model bisenetv1 --weight-path /path/to/your/model.pth --outpath ./model.onnx 
```

**NOTE:** I use cropsize of `1024x2048` here in my example, you should change it according to your specific application. The inference cropsize is fixed from this step on, so you should decide the inference cropsize when you export the model here.

### Build with source code
Just use the standard cmake build method:  
```
mkdir -p tensorrt/build
cd tensorrt/build
cmake ..
make
```
This would generate a `./segment` in the `tensorrt/build` directory.


### Convert onnx to tensorrt model
If you can successfully compile the source code, you can parse the onnx model to tensorrt model like this:  
```
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt
```
If your gpu support acceleration with fp16 inferenece, you can add a `--fp16` option to in this step:  
```
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --fp16
```
Note that I use the simplest method to parse the command line args, so please do **Not** change the order of the above command.  


### Infer with one single image
Run inference like this:   
```
$ ./segment run /path/to/saved_model.trt /path/to/input/image.jpg /path/to/saved_img.jpg
```

### Test speed  
The speed depends on the specific gpu platform you are working on, you can test the fps on your gpu like this:  
```
$ ./segment test /path/to/saved_model.trt
```


## Tips:  
1. Since tensorrt 7.0.0 cannot parse well the `bilinear interpolation` op exported from pytorch, I replace them with pytorch `nn.PixelShuffle`, which would bring some performance overhead(more flops and parameters), and make inference a bit slower. Also due to the `nn.PixelShuffle` op, you **must** export the onnx model with input size to be *n* times of 32.

2. There would be some problem for tensorrt 7.0.0 to parse the `nn.AvgPool2d` op from pytorch with onnx opset11. So I use opset10 to export the model. 

4. The speed(fps) is tested on a single nvidia Tesla T4 gpu with `batchsize=1` and `cropsize=(1024,2048)`. Please note that T4 gpu is almost 2 times slower than 2080ti, you should evaluate the speed considering your own platform and cropsize. Also note that the performance would be affected if your gpu is concurrently working on other tasks. Please make sure no other program is running on your gpu when you test the speed.

5. On my platform, after compiling with tensorrt, the model size of bisenetv1 is 33Mb(fp16) and 133Mb(fp32), and the size of bisenetv2 is 29Mb(fp16) and 54Mb(fp32). However, the fps of bisenetv1 is 60(fp16) and 19(fp32), while the fps of bisenetv2 is 50(fp16) and 16(fp32). It is obvious that bisenetv2 has fewer parameters than bisenetv1, but the speed is otherwise. I am not sure whether it is because tensorrt has worse optimization strategy in some ops used in bisenetv2(such as depthwise convolution) or because of the limitation of the gpu on different ops. Please tell me if you have better idea on this.

