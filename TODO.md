- [ ] RGBD training

- [ ] /home/matteo/miniconda3/envs/genesis/lib/python3.12/site-packages/torch/__init__.py:1551: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)

- [ ] Check that DinoV3 was still resizing your 480×640 RGB frames down to its default resolution,

- [ ] verify CLS token handling

- [ ] verify image features extracted before feeding to dino, check why non funziona neanche se evito quel passaggio

- [ ] Geodesic loss and rotation representation is important
