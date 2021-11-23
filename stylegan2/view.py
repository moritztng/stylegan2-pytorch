from jinja2 import Environment, PackageLoader, select_autoescape
from torchvision.utils import save_image

class View():
    def __init__(self):
        self.env = Environment(loader=PackageLoader('stylegan2'), autoescape=select_autoescape())
        self.template = self.env.get_template('template.html')
        self.mean_abs_grad_d = {}
        self.mean_abs_grad_g = {}

    def __call__(self, epoch, batch, mean_loss_d, mean_loss_g, images_real, images_fake, discriminator, generator):
        save_image(images_real, 'images_real.png')
        save_image(images_fake, 'images_fake.png')
        self._append_grad(self.mean_abs_grad_d, discriminator)
        self._append_grad(self.mean_abs_grad_g, generator)
        with open('view.html', 'w') as f:
            f.write(self.template.render(epoch=epoch, batch=batch, labels=list(range(len(mean_loss_d))), mean_loss_d=mean_loss_d, mean_loss_g=mean_loss_g, mean_abs_grad_d=self.mean_abs_grad_d, mean_abs_grad_g=self.mean_abs_grad_g))
    
    def _append_grad(self, grad, network):
        for name, param in network.named_parameters():
            if 'bias' not in name:
                name = name.replace('.', '_')
                mean_abs_grad = round(param.grad.abs().mean().item(), 5)
                if name in grad:
                    grad[name].append(mean_abs_grad)
                else:
                    grad[name] = [mean_abs_grad]
