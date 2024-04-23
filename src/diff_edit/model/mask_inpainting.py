import torch
from diff_edit.model.constants import TORCH_SEED

class Inpainter:
    def __init__(self, inpainting, torch_device):
        self.inpainting = inpainting
        self.torch_device = torch_device

    def inpaint_mask(self, img, mask, prompt, seed=TORCH_SEED):
        """
            This method performs the inpainting of the image using the provided mask.

            img (PIL.Image): The image to inpaint.
            mask (PIL.Image): The mask to use for inpainting.
            prompt (str): The prompt to use for inpainting.
            seed (int): The seed to use for reproducibility.

            return (PIL.Image): The inpainted image.
        """
        return self.inpainting(prompt=[prompt], image=img, mask_image=mask,
                               generator=torch.Generator(self.torch_device).manual_seed(seed)).images[0]