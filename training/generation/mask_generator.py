import torch
from PIL import Image
import numpy as np


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


class EyeMaskGANGenerator(object):
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.generator = torch.jit.load(model_path, map_location=device)
        self.generator.eval()
        self.device = device

    def __call__(self) -> Image:
        z = torch.FloatTensor(np.random.normal(0, 1, (1, 100, 1, 1))).to(self.device)
        with torch.no_grad():
            img_tensor = self.generator(z).to('cpu')[0]

        norm_range(img_tensor, None)
        img = img_tensor.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()[0]

        return Image.fromarray(img).convert('L')


if __name__ == '__main__':
    model = EyeMaskGANGenerator('mask_generator.pt')
    img = model()
    img.show()

    from tqdm import tqdm
    import os

    target_folder = './data_generation/masks/'
    os.makedirs(target_folder, exist_ok=True)
    size = 1000
    for i in tqdm(range(size)):
        model().save(
            os.path.join(target_folder, 'mask_{}.png'.format(i + 1))
        )
