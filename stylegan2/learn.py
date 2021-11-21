import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, RandomHorizontalFlip, ToTensor, Normalize, Compose
from itertools import islice
from .networks import Discriminator, Generator
from .view import View
from .utils import set_requires_grad

class StyleGAN():
    def __init__(self, lr=0.001, scale_lr_mapper=0.01, equal_lr=True, betas=(0, 0.99), scale_r1=10, resolution=1024,
                 n_feature_maps=[512, 512, 512, 512, 512, 256, 128, 64, 32],
                 params_discriminator={'n_conv': 2},
                 params_generator={'n_layers_mapper': 8,
                                   'n_dim_mapper': 512, 
                                   'n_conv_blocks': 2,
                                   'n_dim_const': 4},
                 path_checkpoint_init=None, device=None):
        self.scale_r1 = scale_r1
        self.resolution = resolution
        self.n_dim_z = params_generator['n_dim_mapper']
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.discriminator = Discriminator(params_discriminator['n_conv'], n_feature_maps[::-1], equal_lr).to(self.device)
        self.generator = Generator(params_generator['n_layers_mapper'], params_generator['n_dim_mapper'], params_generator['n_conv_blocks'], n_feature_maps, params_generator['n_dim_const'], equal_lr).to(self.device)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.optimizer_g = Adam([{'params': self.generator.mapper.parameters(), 'lr': scale_lr_mapper * lr}, {'params': self.generator.synthesizer.parameters(), 'lr': lr}], betas=betas)
        self.epoch = 0
        self.batch = 0
        self.mean_loss_d = [0] 
        self.mean_loss_g = [0]
        self.view = View()

        if path_checkpoint_init:
            checkpoint = torch.load(path_checkpoint_init)
            self.discriminator.load_state_dict(checkpoint['state_dict_d'])
            self.generator.load_state_dict(checkpoint['state_dict_g'])
            self.optimizer_d.load_state_dict(checkpoint['state_dict_optim_d'])
            self.optimizer_g.load_state_dict(checkpoint['state_dict_optim_g'])
            self.epoch = checkpoint['epoch']
            self.batch = checkpoint['batch'] + 1
            self.mean_loss_d = checkpoint['mean_loss_d']
            self.mean_loss_g = checkpoint['mean_loss_g']

    def __call__(self, data, n_epochs, batch_size=16, n_batch_log=100, n_log_checkpoint=None, path_checkpoint_save='checkpoint.pth'):
        dataset = ImageFolder(data, transform=Compose([Resize(self.resolution), RandomHorizontalFlip(), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(self.epoch, self.epoch + n_epochs):
            self.epoch = epoch
            for batch, (images_real, _) in enumerate(islice(dataloader, self.batch, None), self.batch):
                self.batch = batch

                self.optimizer_d.zero_grad()

                images_real = images_real.to(self.device)
                images_real.requires_grad = True
                pred_real = self.discriminator(images_real)
                loss_d_real = nn.functional.softplus(-pred_real).mean()
                loss_d_real.backward(retain_graph=True)

                grad_d_real = torch.autograd.grad(pred_real.sum(), images_real, create_graph=True)[0]
                r1 = self.scale_r1 / 2 * (grad_d_real.view(grad_d_real.size(0), -1).norm(p=2, dim=1)**2).mean()
                r1.backward()

                with torch.no_grad():
                    images_fake = self.generator(torch.randn(images_real.size(0), self.n_dim_z, device=self.device))
                pred_fake = self.discriminator(images_fake)
                loss_d_fake = nn.functional.softplus(pred_fake).mean()
                loss_d_fake.backward()
                
                self.optimizer_d.step()

                self.mean_loss_d[-1] += (loss_d_real.item() + loss_d_fake.item()) / 2

                self.optimizer_g.zero_grad()
                
                images_fake = self.generator(torch.randn(images_real.size(0), self.n_dim_z, device=self.device))
                set_requires_grad(self.discriminator, False)
                pred_fake = self.discriminator(images_fake)
                set_requires_grad(self.discriminator, True)
                loss_g = nn.functional.softplus(-pred_fake).mean()
                loss_g.backward()

                self.optimizer_g.step()

                self.mean_loss_g[-1] += loss_g.item()
                
                if (batch + 1) % n_batch_log == 0:
                    self.mean_loss_d[-1] /= n_batch_log
                    self.mean_loss_g[-1] /= n_batch_log
                    self.view(epoch, batch + 1, self.mean_loss_d, self.mean_loss_g, images_fake.detach() * 0.5 + 0.5)
                    self.mean_loss_d.append(0)
                    self.mean_loss_g.append(0)
                    
                    if n_log_checkpoint and (batch + 1) % (n_log_checkpoint * n_batch_log) == 0:
                        torch.save({'state_dict_d': self.discriminator.state_dict(),
                                    'state_dict_g': self.generator.state_dict(),
                                    'state_dict_optim_d': self.optimizer_d.state_dict(),
                                    'state_dict_optim_g': self.optimizer_g.state_dict(),
                                    'epoch': epoch,
                                    'batch': batch,
                                    'mean_loss_d': self.mean_loss_d,
                                    'mean_loss_g': self.mean_loss_g}, path_checkpoint_save)
            
            self.batch = 0
            self.mean_loss_d[-1] = 0
            self.mean_loss_g[-1] = 0
