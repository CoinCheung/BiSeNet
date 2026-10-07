import argparse
import os
import os.path as osp
import sys
# repo root derived from __file__, so this works from any cwd
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import copy

import torch
import torch.nn as nn
from torch.export import Dim

import numpy as np
from modelopt.onnx.quantization import quantize
import modelopt.onnx.autocast as autocast

import onnx
import onnx_graphsurgeon as gs
import onnxoptimizer


from lib.models import model_factory
from lib.data import get_data_loader
from configs import set_cfg_from_file

'''
    Install tensorrt and modelopt:
        python -m pip install --extra-index-url https://pypi.nvidia.com --upgrade tensorrt nvidia-modelopt[all]
        python -m pip install onnx onnx_graphsurgeon onnxoptimizer

    Command:
        python tools/export_onnx.py \
                 --config configs/bisenetv2_city.py \
                 --weight-path /path/to/model_final.pth \
                 --outpath ./model.onnx \
                 --input-size 1024 2048 \
                 --fuse-head --int8 --int8-no-autotune
'''

torch.set_grad_enabled(False)


parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='model_final.pth')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='model.onnx')
# h w of the exported input
parse.add_argument('--input-size', dest='input_size', type=int, nargs=2,
        default=[1024, 2048])
parse.add_argument('--int8', action='store_true')
# autotune searches per-region quantization schemes; ~3h and needs an idle GPU
parse.add_argument('--int8-no-autotune', action='store_true')
# replace the Resize->ArgMax->Cast tail with the fused head plugin; run this
# last, after quantization, so autotune never sees a custom op
parse.add_argument('--fuse-head', action='store_true')
args = parse.parse_args()



class ModelWrapper(nn.Module):

    def __init__(self, cfg, weight_pth, img_mean, img_std):
        super(ModelWrapper, self).__init__()
        assert cfg.n_cats <= 256
        self.register_buffer('img_mean', torch.tensor(img_mean).reshape(1, 1, 1, -1))
        self.register_buffer('img_rstd', 1. / torch.tensor(img_std).reshape(1, 1, 1, -1))
        net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
        net.load_state_dict(torch.load(weight_pth, map_location='cpu'), strict=False)
        net.eval()
        if hasattr(net, 'deploy'): net.deploy()
        self.net = net

    def forward(self, x):
        '''
        x: 1hw3, uint8
        '''
        x = x.float().div(255.).sub(self.img_mean).mul(self.img_rstd)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.net(x).to(torch.uint8)




def get_calib_batches(cfg, n_batches=100):
    """Raw HWC uint8 images; /255 and mean/std happen inside the graph."""
    cfg = copy.deepcopy(cfg)
    dataloader = get_data_loader(cfg, mode='raw')
    batches = []
    for ind, (img, lab) in enumerate(dataloader):
        if ind >= n_batches: break
        batches.append(img)
    return torch.cat(batches, dim=0).numpy()


def optimize_onnx_onnxopt(onnx_path):

    print(f"Optimizing ONNX with onnxoptimizer: {onnx_path}")

    model = onnx.load(onnx_path)

    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_unused_initializer',
        'fuse_bn_into_conv',
        'fuse_add_bias_into_conv',
        'fuse_pad_into_conv',
        'eliminate_nop_reshape',
        'eliminate_nop_transpose',
    ]
    model_simp = onnxoptimizer.optimize(model, passes)

    base, ext = osp.splitext(onnx_path)
    output_path = f"{base}_opt{ext}"
    onnx.save(model_simp, output_path)
    print(f"Optimized ONNX saved: {onnx_path}")

    return output_path


def quantize_model(onnx_path, cfg, img_mean, img_std, autotune=True):

    base, ext = osp.splitext(onnx_path)
    output_path = f"{base}_int8{ext}"

    kw = dict(
        calibration_data={'input_image': get_calib_batches(cfg, n_batches=100)},
        quantize_mode="int8", # or fp8
        output_path=output_path,
        #  op_types_to_quantize=['Conv', 'MatMul'], # default is all ops
        calibration_method='entropy',
        #  log_level='DEBUG',
    )

    if autotune:
        # A fresh directory per run is mandatory: the timing cache is keyed on
        # layer descriptors and ignores weights, so two graphs of the same shape
        # collide and the second run silently reuses the first one's latencies.
        at_dir = f"{base}_autotune"
        os.makedirs(at_dir, exist_ok=True)
        kw.update(
            autotune=True,
            autotune_use_trtexec=True,
            autotune_output_dir=at_dir,
            autotune_state_file=f"{at_dir}/state.json",
            autotune_timing_cache=f"{at_dir}/timing.cache",
            autotune_pattern_cache_file=f"{at_dir}/state_pattern_cache.json",
            autotune_num_schemes_per_region=50,
            autotune_warmup_runs=50,
            autotune_timing_runs=100,
            autotune_trtexec_args="--noDataTransfers --useCudaGraph",
            autotune_verbose=True,
        )
        print(f"autotune enabled, artifacts in {at_dir} (expect ~3h)", flush=True)

    quantize(onnx_path, **kw)

    ## we do not optimizer after quant, no need
    #  optimize_onnx_onnxopt(onnx_path)
    #  print(f"Quantized model saved to: {output_path}")

    return output_path



def fuse_head(onnx_path):
    """Replace Resize(bilinear,half_pixel) -> ArgMax(axis=1) with a single
    FusedUpsampleArgMax node served by libfused_head_plugin.so.

    The plugin emits int32, so a Cast(int32) after the ArgMax is absorbed and the
    graph is exactly what an int32 export always produced.  Other Casts are kept
    and re-wired to the plugin; any other consumer (or a graph output) gets a
    Cast(int64) to preserve ArgMax's output type.

    Run this after quantization, so autotune never has to deal with a custom op.
    Q/DQ pairs between the Resize and the ArgMax are bypassed; the rest of the
    autotune result is kept.  The source file is not modified.
    """
    base, ext = osp.splitext(onnx_path)
    output_path = f"{base}_fused{ext}"

    g = gs.import_onnx(onnx.load(onnx_path))

    argmaxes = [n for n in g.nodes if n.op == "ArgMax"]
    # next() would silently take the first one in graph order, which need not be
    # the main output head on a multi-task graph
    assert len(argmaxes) == 1, f"graph has {len(argmaxes)} ArgMax nodes"
    argmax = argmaxes[0]
    # Q -> DQ pairs between the Resize and the ArgMax (autotune may place them) are bypassed:
    # the plugin takes the argmax of the interpolated values, not of their int8-rounded copy
    bypassed, head = [], argmax
    while head.inputs[0].inputs and head.inputs[0].inputs[0].op == "DequantizeLinear":
        dq = head.inputs[0].inputs[0]
        q = dq.inputs[0].inputs[0] if dq.inputs[0].inputs else None
        assert q is not None and q.op == "QuantizeLinear", "DequantizeLinear before the ArgMax without a QuantizeLinear"
        for t, user in ((dq.outputs[0], head), (q.outputs[0], dq)):
            assert t not in g.outputs and [n for n in g.nodes if t in n.inputs] == [user], \
                "Q/DQ between the Resize and the ArgMax has other consumers"
        bypassed += [dq, q]
        head = q
    resize = head.inputs[0].inputs[0]
    assert resize.op == "Resize", resize.op
    # the Resize is removed, so nothing else may read its output
    rz_out = resize.outputs[0]
    assert rz_out not in g.outputs and [n for n in g.nodes if rz_out in n.inputs] == [head], \
        "Resize output has consumers other than the ArgMax"

    # the plugin hard-codes these semantics; refuse to fuse if they ever change
    # (omitted attributes take their ONNX defaults)
    a = resize.attrs
    assert a.get("mode", "nearest") == "linear", a.get("mode", "nearest")
    assert a.get("coordinate_transformation_mode", "half_pixel") == "half_pixel", a["coordinate_transformation_mode"]
    assert int(a.get("antialias", 0)) == 0
    assert int(a.get("exclude_outside", 0)) == 0
    assert a.get("keep_aspect_ratio_policy", "stretch") == "stretch" and "axes" not in a, \
        "Resize with keep_aspect_ratio_policy / axes is not supported"
    ins = resize.inputs
    if len(ins) > 2 and isinstance(ins[2], gs.Constant) and ins[2].values.size:
        scales = [float(v) for v in ins[2].values]
    else:   # sizes: needs a static input shape to recover the scales
        x_shape = ins[0].shape
        assert len(ins) > 3 and isinstance(ins[3], gs.Constant) and x_shape is not None \
            and all(isinstance(d, int) for d in x_shape[1:]), "Resize needs constant scales, or constant sizes and a static input shape"
        scales = [1.0, 1.0] + [float(o) / float(i) for o, i in zip(ins[3].values[2:], x_shape[2:])]
        assert list(ins[3].values[1:2]) == list(x_shape[1:2]), "Resize sizes must keep the channel count"
        assert not isinstance(x_shape[0], int) or int(ins[3].values[0]) == x_shape[0], \
            "Resize sizes must keep the batch size"
    assert len(scales) == 4 and scales[0] == 1.0 and scales[1] == 1.0, scales
    sh, sw = int(scales[2]), int(scales[3])
    assert float(sh) == scales[2] and float(sw) == scales[3], scales
    assert sh >= 2 and sw >= 2, scales
    assert int(argmax.attrs.get("axis", 0)) == 1, "ArgMax must reduce the channel axis (axis=1)"
    assert int(argmax.attrs.get("keepdims", 1)) == 0, "ArgMax must drop the channel axis (keepdims=0)"
    assert int(argmax.attrs.get("select_last_index", 0)) == 0, "ArgMax must pick the first maximum"

    src, idx = resize.inputs[0], argmax.outputs[0]
    casts = [n for n in g.nodes if idx in n.inputs and n.op == "Cast"]
    others = [n for n in g.nodes if idx in n.inputs and n.op != "Cast"]
    keep_int64 = bool(others) or idx in g.outputs

    i32 = next((n for n in casts if int(n.attrs["to"]) == onnx.TensorProto.INT32), None)
    if i32 is not None:
        casts.remove(i32)
        fused = i32.outputs[0]
        i32.inputs.clear()
        i32.outputs.clear()
    else:
        name = f"{idx.name}_int32"
        while name in g.tensors():
            name += "_"
        fused = gs.Variable(name, dtype=np.int32, shape=idx.shape)
    g.nodes.append(gs.Node(
        op="FusedUpsampleArgMax", name="fused_upsample_argmax", domain="custom",
        attrs={"scale_h": sh, "scale_w": sw,
               "plugin_version": "1", "plugin_namespace": "custom"},
        inputs=[src], outputs=[fused]))
    for n in casts:
        n.inputs[n.inputs.index(idx)] = fused
    for n in [resize, argmax] + bypassed:
        n.inputs.clear()
        n.outputs.clear()
    if keep_int64:
        g.nodes.append(gs.Node(op="Cast", name="fused_upsample_argmax_int64",
                               attrs={"to": onnx.TensorProto.INT64}, inputs=[fused], outputs=[idx]))

    tail = [f"Cast(to={onnx.TensorProto.DataType.Name(int(n.attrs['to']))})" for n in casts]
    tail += ["Cast(to=INT64)"] if keep_int64 else []
    print(f"fusing {src.name} {src.shape} -> FusedUpsampleArgMax(int32) -> {', '.join(tail) or 'nothing'}, "
          f"scales=({sh},{sw})" + (f", bypassed {len(bypassed) // 2} Q/DQ pair(s) after the Resize" if bypassed else ""))

    g.cleanup().toposort()
    m = gs.export_onnx(g)
    if not any(o.domain == "custom" for o in m.opset_import):
        op = m.opset_import.add()
        op.domain, op.version = "custom", 1
    onnx.checker.check_model(m)
    onnx.save(m, output_path)
    print(f"fused model saved to: {output_path}")
    return output_path


def convert_fp16(onnx_path, cfg):
    """Write an fp16 mixed-precision copy of onnx_path.

    TensorRT 11 removed the per-precision builder flags, so the precision has
    to live in the graph instead of being selected at build time.
    """
    base, ext = osp.splitext(onnx_path)
    output_path = f"{base}_fp16{ext}"

    calib_npz_path = f"{base}_autocast_calib.npz"
    np.savez(calib_npz_path, input_image=get_calib_batches(cfg, n_batches=4))

    model = autocast.convert_to_mixed_precision(
        onnx_path,
        calibration_data=calib_npz_path,
        low_precision_type='fp16',
        data_max=32768,
        keep_io_types=True,     # input stays uint8, output keeps its integer type
    )
    onnx.save(model, output_path)
    print(f"fp16 model saved to: {output_path}")
    return output_path


def export_func_dynamo(net, crop_size, out_pth):
    dummy_input = torch.randint(0, 256, (1, *crop_size, 3), dtype=torch.uint8)
    input_names = ['input_image',]
    output_names = ['preds',]
    batch_size = Dim('batch', min=1, max=4)
    dynamic_shapes = ({0: batch_size},)

    out = net(dummy_input)

    torch.onnx.export(net, dummy_input, out_pth,
        input_names=input_names, output_names=output_names,
        do_constant_folding=True,
        verbose=False, opset_version=18,
        external_data=False,
        dynamic_shapes=dynamic_shapes)


def export_bisenet():
    cfg = set_cfg_from_file(args.config)
    if cfg.use_sync_bn: cfg.use_sync_bn = False

    img_mean, img_std = cfg.img_mean, cfg.img_std

    net = ModelWrapper(cfg, args.weight_pth, img_mean, img_std)
    net.eval()

    cfg.cropsize = args.input_size

    export_func_dynamo(net, cfg.cropsize, args.out_pth)
    opt_out_pth = optimize_onnx_onnxopt(args.out_pth)

    fp16_out_pth = convert_fp16(opt_out_pth, cfg)

    if args.fuse_head:
        fuse_head(opt_out_pth)
        fuse_head(fp16_out_pth)

    if args.int8:
        autotune = not args.int8_no_autotune
        int8_pth = quantize_model(opt_out_pth, cfg, img_mean, img_std, autotune=autotune)
        if args.fuse_head:
            fuse_head(int8_pth)


export_bisenet()
