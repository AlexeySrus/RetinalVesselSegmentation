import numpy as np
import torch
import torchvision
from PIL import Image


class UGATITModelInference(object):
    def __init__(self,
                 model_path: str = None,
                 img_size: int = 256,
                 device: str = 'cpu'):
        self.device = device
        self.style_model = torch.jit.load(model_path, map_location=device)
        _ = self.style_model.eval()

        mean_std = (0.5, 0.5, 0.5)

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=mean_std, std=mean_std)
        ])

    def apply_styler(self, inp_image: Image) -> Image:
        input_tensor = self.preprocess(inp_image).unsqueeze(dim=0)
        input_tensor = input_tensor.to(self.device).mean(dim=1).unsqueeze(1)
        out, _, _ = self.style_model(input_tensor)
        out = out.detach().to('cpu')[0].permute(1, 2, 0) * 0.5 + 0.5
        out = (np.clip(out.numpy(), 0, 1) * 255.0).astype(np.uint8)

        res_img = Image.fromarray(out)
        return res_img

    def __call__(self, img: Image) -> Image:
        """
        Inference method
        Args:
            img: PIL image instance

        Returns:
            PIL image instance
        """
        return self.apply_styler(img.convert('RGB')).convert('RGB')


if __name__ == '__main__':
    model = UGATITModelInference('traced_mask2eye.pt')


    from tqdm import tqdm
    import os

    input_folder = './data_generation/images/'
    target_folder = './data_generation/masks/'
    os.makedirs(target_folder, exist_ok=True)
    size = 1000
    for i in tqdm(range(size)):
        mask_p = os.path.join(input_folder, 'mask_{}.png'.format(i + 1))
        model(Image.open(mask_p)).save(
            os.path.join(target_folder, 'eye/', 'mask_{}.png'.format(i + 1))
        )
