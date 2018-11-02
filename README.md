# spectral normalization and projection discriminator(Pytorch)
This project attempts to reproduce the results from the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. The Official Chainer implementation [**link**](https://github.com/pfnet-research/sngan_projection)

### Setup:
`pip install pytorch pyyaml`

### Training(cifar10):
python train.py --config_path configs/sn_cifar10_conditional.yml --batch_size 128

### Evaluation:
to be implement (inception score, fid)

### 64X64 Imagenet Dog Samples
![](img/250000_img.png)

### References
- Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida. *Spectral Normalization for Generative Adversarial Networks*. ICLR2018. [OpenReview][sngans]
- Takeru Miyato, Masanori Koyama. *cGANs with Projection Discriminator*. ICLR2018. [OpenReview][pcgans]

