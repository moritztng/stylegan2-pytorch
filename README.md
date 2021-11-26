## Installation
```bash
pip install git+https://github.com/moritztng/stylegan2-pytorch.git
```

## Quickstart
### Train
Downloads Flickr Faces HQ Dataset into `data` and creates checkpoint
```bash
stylegan2 learn --path-ffhq data
```
### Generate
Loads checkpoint and creates `images.png`
```bash
stylegan2 generate
```

## Python Object
```python
import torch
from torchvision.utils import save_image
from stylegan2.learn import StyleGAN

# Train
stylegan = StyleGAN(path_ffhq='data')
stylegan('data')

# Generate
generator = stylegan.generator_average
latent = torch.randn(16, 512, device=stylegan.device)
images = generator(latent, weight_truncation=0.7)
save_image(images * 0.5 + 0.5, 'images.png')
```

## Documentation
### Train
```bash
$ stylegan2 learn --help
Usage: stylegan2 learn [OPTIONS]

Options:
  --data TEXT
  --n-epochs INTEGER           [default: 1]
  --batch-size INTEGER         [default: 16]
  --n-batch-log INTEGER        [default: 100]
  --n-log-checkpoint INTEGER   [default: 1]
  --path-checkpoint-save TEXT  [default: checkpoint.pth]
  --lr FLOAT                   [default: 0.001]
  --scale-lr-mapper FLOAT      [default: 0.01]
  --equal-lr / --no-equal-lr   [default: True]
  --betas <FLOAT FLOAT>...     [default: 0.0, 0.99]
  --scale-r1 FLOAT             [default: 10.0]
  --resolution INTEGER         [default: 128]
  --n-feature-maps TEXT        [default: [512, 512, 512, 512, 512, 256]]
  --n-conv-d INTEGER           [default: 2]
  --n-layers-mapper-g INTEGER  [default: 8]
  --n-dim-mapper-g INTEGER     [default: 512]
  --n-conv-blocks-g INTEGER    [default: 2]
  --n-dim-const-g INTEGER      [default: 4]
  --decay-generator-avg FLOAT  [default: 0.999]
  --path-checkpoint-init TEXT
  --path-ffhq TEXT
  --device TEXT
  --help                       Show this message and exit.
```
### Generate
```bash
$ stylegan2 generate --help
Usage: stylegan2 generate [OPTIONS]

Options:
  --path-checkpoint TEXT      [default: checkpoint.pth]
  --path-images TEXT          [default: images.png]
  --n-images INTEGER          [default: 16]
  --n-layers-mapper INTEGER   [default: 8]
  --n-dim-mapper INTEGER      [default: 512]
  --n-conv-blocks INTEGER     [default: 2]
  --n-feature-maps TEXT       [default: [512, 512, 512, 512, 512, 256]]
  --n-dim-const INTEGER       [default: 4]
  --equal-lr / --no-equal-lr  [default: True]
  --weight-truncation FLOAT   [default: 0.7]
  --device TEXT
  --help                      Show this message and exit.
```

## References
* [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)
* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
* [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
