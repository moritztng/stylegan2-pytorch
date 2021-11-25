import click, torch
from ast import literal_eval
from torchvision.utils import save_image
from .learn import StyleGAN
from .networks import Generator

@click.group()
def stylegan2():
    pass

@stylegan2.command()
@click.option('--data')
@click.option('--n-epochs', default=1, show_default=True)
@click.option('--batch-size', default=16, show_default=True)
@click.option('--n-batch-log', default=100, show_default=True)
@click.option('--n-log-checkpoint', default=1, show_default=True)
@click.option('--path-checkpoint-save', default='checkpoint.pth', show_default=True)
@click.option('--lr', default=0.001, show_default=True)
@click.option('--scale-lr-mapper', default=0.01, show_default=True)
@click.option('--equal-lr/--no-equal-lr', default=True, show_default=True)
@click.option('--betas', default=(0., 0.99), nargs=2, show_default=True)
@click.option('--scale-r1', default=10., show_default=True)
@click.option('--resolution', default=128, show_default=True)
@click.option('--n-feature-maps', default='[512, 512, 512, 512, 512, 256]', show_default=True)
@click.option('--n-conv-d', default=2, show_default=True)
@click.option('--n-layers-mapper-g', default=8, show_default=True)
@click.option('--n-dim-mapper-g', default=512, show_default=True)
@click.option('--n-conv-blocks-g', default=2, show_default=True)
@click.option('--n-dim-const-g', default=4, show_default=True)
@click.option('--decay-generator-avg', default=0.999, show_default=True)
@click.option('--path-checkpoint-init')
@click.option('--path-ffhq')
@click.option('--device')
def learn(data, n_epochs, batch_size, n_batch_log, n_log_checkpoint, path_checkpoint_save, lr, scale_lr_mapper, equal_lr, betas, scale_r1, resolution, n_feature_maps, n_conv_d, n_layers_mapper_g, n_dim_mapper_g, n_conv_blocks_g, n_dim_const_g, decay_generator_avg, path_checkpoint_init, path_ffhq, device):
    stylegan = StyleGAN(lr, scale_lr_mapper, equal_lr, betas, scale_r1, resolution, literal_eval(n_feature_maps),
                        {'n_conv': n_conv_d},
                        {'n_layers_mapper': n_layers_mapper_g,
                         'n_dim_mapper': n_dim_mapper_g, 
                         'n_conv_blocks': n_conv_blocks_g,
                         'n_dim_const': n_dim_const_g},
                        decay_generator_avg, path_checkpoint_init, path_ffhq, device)
    stylegan(data if data else path_ffhq, n_epochs, batch_size, n_batch_log, n_log_checkpoint, path_checkpoint_save)

@stylegan2.command()
@click.option('--path-checkpoint', default='checkpoint.pth', show_default=True)
@click.option('--path-images', default='images.png', show_default=True)
@click.option('--n-images', default=16, show_default=True)
@click.option('--n-layers-mapper', default=8, show_default=True)
@click.option('--n-dim-mapper', default=512, show_default=True)
@click.option('--n-conv-blocks', default=2, show_default=True)
@click.option('--n-feature-maps', default='[512, 512, 512, 512, 512, 256]', show_default=True)
@click.option('--n-dim-const', default=4, show_default=True)
@click.option('--equal-lr/--no-equal-lr', default=True, show_default=True)
@click.option('--weight-truncation', default=0.7, show_default=True)
@click.option('--device')
def generate(path_checkpoint, path_images, n_images, n_layers_mapper, n_dim_mapper, n_conv_blocks, n_feature_maps, n_dim_const, equal_lr, weight_truncation, device):
    device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = Generator(n_layers_mapper, n_dim_mapper, n_conv_blocks, literal_eval(n_feature_maps), n_dim_const, equal_lr).to(device).eval()
    generator.load_state_dict(torch.load(path_checkpoint)['state_dict_g_avg'])
    with torch.no_grad():
        images = generator(torch.randn(n_images, n_dim_mapper, device=device), weight_truncation)
        save_image(images * 0.5 + 0.5, path_images)

if __name__ == '__main__':
    stylegan2()
