import argparse
import ctypes
import os
import os.path as osp
import time

import cv2
import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


MAX_BATCH = 4
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
DEFAULT_PLUGIN = osp.join(osp.dirname(osp.abspath(__file__)),
        'build', 'plugins', 'fused_head', 'libfused_head_plugin.so')
LABEL_DTYPES = (np.int32, np.uint8, np.int64)


def check(ret):
    if not isinstance(ret, tuple):
        ret = (ret, )
    err, rest = ret[0], ret[1:]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(err)[1].decode())
    if len(rest) == 0:
        return None
    return rest[0] if len(rest) == 1 else rest


def load_plugin(path):
    if path and osp.exists(path):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    elif path:
        print(f'plugin not found: {path}, models with the fused head will not load')
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def network_flags():
    flag = getattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED', None)
    return 0 if flag is None else 1 << int(flag)


def compile_onnx(onnx_path, savepth, opt_bsize):
    if not 1 <= opt_bsize <= MAX_BATCH:
        raise ValueError(f'opt batch size must be in [1, {MAX_BATCH}]')
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(network_flags())
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f'parse onnx failed: {onnx_path}')
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError('expect a model with one input and one output')

    inp = network.get_input(0)
    dims = tuple(inp.shape[1:])
    if inp.shape[0] > 0:
        lo = opt = hi = inp.shape[0]
    else:
        lo, opt, hi = 1, opt_bsize, MAX_BATCH
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 34)
    config.builder_optimization_level = 5
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (lo, ) + dims, (opt, ) + dims, (hi, ) + dims)
    config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('build engine failed')
    with open(savepth, 'wb') as fw:
        fw.write(plan)


class SegmentTrt(object):

    def __init__(self, mdpth, batch=1):
        self.runtime = trt.Runtime(TRT_LOGGER)
        with open(mdpth, 'rb') as fr:
            self.engine = self.runtime.deserialize_cuda_engine(fr.read())
        if self.engine is None:
            raise RuntimeError(f'deserialize engine failed: {mdpth}')

        names = [self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError('expect an engine with one input and one output')
        self.in_name, self.out_name = ins[0], outs[0]

        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        if any(d < 0 for d in in_shape[1:]):
            raise RuntimeError('only the batch dimension may be dynamic')
        if in_shape[0] < 0:
            lo, _, hi = self.engine.get_tensor_profile_shape(self.in_name, 0)
            if not lo[0] <= batch <= hi[0]:
                raise RuntimeError(f'batch {batch} is outside this engine, [{lo[0]}, {hi[0]}]')
        elif in_shape[0] != batch:
            raise RuntimeError(f'batch {batch} is outside this engine, [{in_shape[0]}, {in_shape[0]}]')
        self.batch = batch
        self.in_shape = (batch, ) + in_shape[1:]
        self.in_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.in_name)))
        self.out_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.out_name)))
        if self.out_dtype not in [np.dtype(t) for t in LABEL_DTYPES]:
            raise RuntimeError(f'unsupported output dtype: {self.out_dtype}')

        self.context = self.engine.create_execution_context()
        self.context.set_input_shape(self.in_name, self.in_shape)
        self.out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        self.stream = check(cudart.cudaStreamCreate())
        self.pinned = os.environ.get('TRT_NO_PINNED') != '1'
        self.h_in, self.h_in_ptr = self._host(self.in_shape, self.in_dtype)
        self.h_out, self.h_out_ptr = self._host(self.out_shape, self.out_dtype)
        self.in_nbytes = self.h_in.nbytes
        self.out_nbytes = self.h_out.nbytes
        self.d_in = check(cudart.cudaMalloc(self.in_nbytes))
        self.d_out = check(cudart.cudaMalloc(self.out_nbytes))
        check(cudart.cudaMemsetAsync(self.d_in, 0, self.in_nbytes, self.stream))
        check(cudart.cudaMemsetAsync(self.d_out, 0, self.out_nbytes, self.stream))
        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))

        for _ in range(3):
            self._enqueue()
        check(cudart.cudaStreamSynchronize(self.stream))
        self.graph_exec = None
        if os.environ.get('TRT_NO_CUDA_GRAPH') != '1':
            self._capture()

    def _host(self, shape, dtype):
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if not self.pinned:
            return np.zeros(shape, dtype), None
        ptr = check(cudart.cudaMallocHost(nbytes))
        buf = (ctypes.c_uint8 * nbytes).from_address(int(ptr))
        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
        arr[...] = 0
        return arr, ptr

    def _enqueue(self):
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError('enqueue failed')

    def _capture(self):
        mode = cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal
        check(cudart.cudaStreamBeginCapture(self.stream, mode))
        ok = self.context.execute_async_v3(int(self.stream))
        err, graph = cudart.cudaStreamEndCapture(self.stream)
        if err == cudart.cudaError_t.cudaSuccess:
            if ok:
                err_inst, graph_exec = cudart.cudaGraphInstantiate(graph, 0)
                if err_inst == cudart.cudaError_t.cudaSuccess:
                    self.graph_exec = graph_exec
            cudart.cudaGraphDestroy(graph)
        cudart.cudaGetLastError()
        if self.graph_exec is None:
            print('WARNING: cuda graph capture failed, running without it')

    def launch(self):
        if self.graph_exec is not None:
            check(cudart.cudaGraphLaunch(self.graph_exec, self.stream))
        else:
            self._enqueue()

    def infer(self, with_transfer=True):
        kind_h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        kind_d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        if with_transfer:
            check(cudart.cudaMemcpyAsync(self.d_in, self.h_in.ctypes.data,
                self.in_nbytes, kind_h2d, self.stream))
        self.launch()
        if with_transfer:
            check(cudart.cudaMemcpyAsync(self.h_out.ctypes.data, self.d_out,
                self.out_nbytes, kind_d2h, self.stream))
        check(cudart.cudaStreamSynchronize(self.stream))
        return self.h_out

    def close(self):
        if self.graph_exec is not None:
            cudart.cudaGraphExecDestroy(self.graph_exec)
            self.graph_exec = None
        for ptr in (self.h_in_ptr, self.h_out_ptr):
            if ptr is not None:
                cudart.cudaFreeHost(ptr)
        cudart.cudaFree(self.d_in)
        cudart.cudaFree(self.d_out)
        cudart.cudaStreamDestroy(self.stream)
        self.h_in_ptr = self.h_out_ptr = None


def get_color_map():
    scaling = (2147483646 - 1) // 256
    past = 256 * scaling
    state = 123
    out = np.empty(256 * 3, dtype=np.uint8)
    for i in range(out.size):
        while True:
            state = state * 48271 % 2147483647
            ret = state - 1
            if ret < past:
                break
        out[i] = ret // scaling
    return out.reshape(256, 3)


def read_image(impth, iH, iW):
    im = cv2.imread(impth)
    if im is None:
        raise RuntimeError(f'cannot read image: {impth}')
    orgH, orgW = im.shape[:2]
    if (orgH, orgW) != (iH, iW):
        print(f'resize orignal image of ({orgH},{orgW}) to ({iH}, {iW}) according to model require')
        im = cv2.resize(im, (iW, iH), interpolation=cv2.INTER_LINEAR)
    return im[:, :, ::-1], (orgH, orgW)


def run(mdpth, impth, outpth):
    seg = SegmentTrt(mdpth, batch=1)
    try:
        _, iH, iW, _ = seg.in_shape
        img, (orgH, orgW) = read_image(impth, iH, iW)
        seg.h_in[0] = img
        labels = seg.infer()[0]
        pred = get_color_map()[labels]
        oH, oW = pred.shape[:2]
        if (orgH, orgW) != (oH, oW):
            pred = cv2.resize(pred, (orgW, orgH), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(outpth, pred)
    finally:
        seg.close()


def test_speed(mdpth, batch, n_loops=2000):
    seg = SegmentTrt(mdpth, batch=batch)
    try:
        _, iH, iW, _ = seg.in_shape
        print(f'test with cropsize of ({iH}, {iW}), and batch size of {batch} ...')
        print(f'output dtype: {seg.out_dtype}, cuda graph: {"on" if seg.graph_exec is not None else "off"}')
        for with_transfer in (False, True):
            for _ in range(20):
                seg.infer(with_transfer)
            t0 = time.perf_counter()
            for _ in range(n_loops):
                seg.infer(with_transfer)
            duration = time.perf_counter() - t0
            print(f'host transfers (H2D+D2H): {"included" if with_transfer else "excluded"}')
            print(f'fps is: {n_loops * batch / duration}')
            print(f'latency per batch: {duration * 1000. / n_loops} ms')
            print(f'latency per frame: {duration * 1000. / (n_loops * batch)} ms')
    finally:
        seg.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', default=DEFAULT_PLUGIN)
    subparsers = parser.add_subparsers(dest='command', required=True)
    compile_parser = subparsers.add_parser('compile')
    compile_parser.add_argument('--onnx', required=True)
    compile_parser.add_argument('--savepth', default='./model.trt')
    compile_parser.add_argument('--opt-batch', type=int, default=1)
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--mdpth', required=True)
    run_parser.add_argument('--impth', required=True)
    run_parser.add_argument('--outpth', default='./res.png')
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--mdpth', required=True)
    test_parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    load_plugin(args.plugin)
    if args.command == 'compile':
        compile_onnx(args.onnx, args.savepth, args.opt_batch)
    elif args.command == 'run':
        run(args.mdpth, args.impth, args.outpth)
    elif args.command == 'test':
        test_speed(args.mdpth, args.batch)


if __name__ == '__main__':
    main()
