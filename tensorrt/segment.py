
import os
import os.path as osp
import cv2
import numpy as np
import logging
import argparse

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")
compile_parser = subparsers.add_parser('compile')
compile_parser.add_argument('--onnx')
compile_parser.add_argument('--quant', default='fp32')
compile_parser.add_argument('--savepth', default='./model.trt')
run_parser = subparsers.add_parser('run')
run_parser.add_argument('--mdpth')
run_parser.add_argument('--impth')
run_parser.add_argument('--outpth', default='./res.png')
args = parser.parse_args()


np.random.seed(123)
in_datatype = trt.nptype(trt.float32)
out_datatype = trt.nptype(trt.int32)
palette = np.random.randint(0, 256, (256, 3)).astype(np.uint8)

ctx = pycuda.autoinit.context
trt.init_libnvinfer_plugins(None, "")
TRT_LOGGER = trt.Logger()



def get_image(impth, size):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    var = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    iH, iW = size[0], size[1]
    img = cv2.imread(impth)[:, :, ::-1]
    orgH, orgW, _ = img.shape
    img = cv2.resize(img, (iW, iH)).astype(np.float32)
    img = img.transpose(2, 0, 1) / 255.
    img = (img - mean) / var
    return img, (orgH, orgW)



def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(
            trt.volume(engine.get_binding_shape(0)), dtype=in_datatype)
    print(engine.get_binding_shape(0))
    d_input = cuda.mem_alloc(h_input.nbytes)
    h_outputs, d_outputs = [], []
    n_outs = 1
    for i in range(n_outs):
        h_output = cuda.pagelocked_empty(
                trt.volume(engine.get_binding_shape(i+1)),
                dtype=out_datatype)
        d_output = cuda.mem_alloc(h_output.nbytes)
        h_outputs.append(h_output)
        d_outputs.append(d_output)
    stream = cuda.Stream()
    return (
        stream,
        h_input,
        d_input,
        h_outputs,
        d_outputs,
    )


def build_engine_from_onnx(onnx_file_path):
    engine = None ## add this to avoid return deleted engine
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:

        # Parse model file
        print(f'Loading ONNX file from path {onnx_file_path}...')
        assert os.path.exists(onnx_file_path), f'cannot find {onnx_file_path}'
        with open(onnx_file_path, 'rb') as fr:
            if not parser.parse(fr.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                assert False

        # build settings
        builder.max_batch_size = 128
        config.max_workspace_size = 1 << 30 # 1G
        if args.quant == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)

        print("Start to build Engine")
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
    return engine


def serialize_engine_to_file(engine, savepth):
    plan = engine.serialize()
    with open(savepth, "wb") as fw:
        fw.write(plan)


def deserialize_engine_from_file(savepth):
    with open(savepth, 'rb') as fr, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(fr.read())
    return engine


def main():
    if args.command == 'compile':
        engine = build_engine_from_onnx(args.onnx)
        serialize_engine_to_file(engine, args.savepth)

    elif args.command == 'run':
        engine = deserialize_engine_from_file(args.mdpth)

        ishape = engine.get_binding_shape(0)
        img, (orgH, orgW) = get_image(args.impth, ishape[2:])

        ## create engine and allocate bffers
        (
            stream,
            h_input,
            d_input,
            h_outputs,
            d_outputs,
        ) = allocate_buffers(engine)
        ctx.push()
        context = engine.create_execution_context()
        ctx.pop()
        bds = [int(d_input), ] + [int(el) for el in d_outputs]

        h_input = np.ascontiguousarray(img)
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async(
            bindings=bds, stream_handle=stream.handle)
        for h_output, d_output in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        oshape = engine.get_binding_shape(1)
        pred = np.argmax(h_outputs[0].reshape(oshape), axis=1)
        out = palette[pred]
        out = out.reshape(*oshape[2:], 3)
        out = cv2.resize(out, (orgW, orgH))
        cv2.imwrite(args.outpth, out)



if __name__ == '__main__':
    main()

