

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

* ubuntu 22.04
* nvidia A40 gpu, driver newer than 555.42.06
* cuda 12.1, cudnn 8
* cmake 3.22.1
* opencv built from source
* tensorrt 10.3.0.26



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
For fp32/fp16/bf16, command is:
```
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --fp32
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --fp16
$ ./segment compile /path/to/onnx.model /path/to/saved_model.trt --bf16
```
Make sure that your gpu support acceleration with fp16/bf16 inferenece when you set these options.<br>

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

The speed(fps) is tested on a single nvidia A40 gpu with `batchsize=1` and `cropsize=(1024,2048)`, which might be different from your platform and settings. You should evaluate the speed considering your own platform and cropsize. Also note that the performance would be affected if your gpu is concurrently working on other tasks. Please make sure no other program is running on your gpu when you test the speed.



### Using python (this is not updated to tensorrt 10.3)

You can also use python script to compile and run inference of your model. <br>

Following is still the usage method of tensorrt 8.2.<br>


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

