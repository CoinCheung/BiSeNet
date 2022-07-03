
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


torch.set_grad_enabled(False)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--input', dest='input', type=str, default='./example.mp4',)
parse.add_argument('--output', dest='output', type=str, default='./res.mp4',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)



# define model
def get_model():
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    net.eval()
    net.cuda()
    return net


# fetch frames
def get_func(inpth, in_q, done):
    cap = cv2.VideoCapture(args.input)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)

    to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = frame[:, :, ::-1]
        frame = to_tensor(dict(im=frame, lb=None))['im'].unsqueeze(0)
        in_q.put(frame)

    in_q.put('quit')
    done.wait()

    cap.release()
    time.sleep(1)
    print('input queue done')


# save to video
def save_func(inpth, outpth, out_q):
    np.random.seed(123)
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(args.input)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_writer = cv2.VideoWriter(outpth,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (int(width), int(height)))

    while True:
        out = out_q.get()
        if out == 'quit': break
        out = out.numpy()
        preds = palette[out]
        for pred in preds:
            video_writer.write(pred)
    video_writer.release()
    print('output queue done')


# inference a list of frames
def infer_batch(frames):
    frames = torch.cat(frames, dim=0).cuda()
    H, W = frames.size()[2:]
    frames = F.interpolate(frames, size=(768, 768), mode='bilinear',
            align_corners=False) # must be divisible by 32
    out = net(frames)[0]
    out = F.interpolate(out, size=(H, W), mode='bilinear',
            align_corners=False).argmax(dim=1).detach().cpu()
    out_q.put(out)



if __name__ == '__main__':
    mp.set_start_method('spawn')

    in_q = mp.Queue(1024)
    out_q = mp.Queue(1024)
    done = mp.Event()

    in_worker = mp.Process(target=get_func,
            args=(args.input, in_q, done))
    out_worker = mp.Process(target=save_func,
            args=(args.input, args.output, out_q))

    in_worker.start()
    out_worker.start()

    net = get_model()

    frames = []
    while True:
        frame = in_q.get()
        if frame == 'quit': break

        frames.append(frame)
        if len(frames) == 8:
            infer_batch(frames)
            frames = []
    if len(frames) > 0:
        infer_batch(frames)

    out_q.put('quit')
    done.set()

    out_worker.join()
    in_worker.join()
