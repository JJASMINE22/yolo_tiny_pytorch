import os
import onnx
import torch
import numpy as np
import onnxruntime
from torch import nn
from torch.nn import functional as F
from net.network import YoloBody
from configure import config as cfg

# ===generate onnx===
model = YoloBody(class_num=cfg.class_names.__len__(),
                 anchor_num=cfg.anchors.__len__()//2).to(cfg.device)
model.eval()

ckpt = torch.load(os.path.join(cfg.ckpt_path, '模型文件'))
model.load_state_dict(ckpt['state_dict'])
print("model successfully loaded, loss is {:.3f} acc is {:.3f}".format(ckpt['loss'],
                                                                       ckpt['acc']))

input_names = ["input"]
output_names = ["output_1", "output_2"]
for named_param in model.named_parameters():
    name, param = named_param
    if param.requires_grad:
        input_names.append(name)

inputs = torch.randn(size=(1, 3, *cfg.input_size), device=cfg.device)
torch.onnx.export(model, inputs, os.path.join(".\\", "yolo.onnx"),
                  verbose=True, input_names=input_names, output_names=output_names,
                  opset_version=11, dynamic_axes=None)

# ===onnx usage===
# ort_session = onnxruntime.InferenceSession(os.path.join(".\\", "yolo.onnx"),
#                                            providers=['CUDAExecutionProvider',
#                                                       'CPUExecutionProvider'])
#
# outputs = ort_session.run(
#     [_.name for _ in ort_session.get_outputs()],
#     {ort_session.get_inputs()[0].name: np.random.randn(1, 3, 416, 416).astype(np.float32)})
