from jinja2 import Environment, PackageLoader, select_autoescape
from torchvision.utils import save_image

class View():
    def __init__(self):
        self.env = Environment(loader=PackageLoader('stylegan2'), autoescape=select_autoescape())
        self.template = self.env.get_template('template.html')
   
    def __call__(self, epoch, batch, mean_loss_d, mean_loss_g, images):
        save_image(images, 'images.png')
        with open('view.html', 'w') as f:
            f.write(self.template.render(epoch=epoch, batch=batch, labels=list(range(len(mean_loss_d))), mean_loss_d=mean_loss_d, mean_loss_g=mean_loss_g, images=images))
