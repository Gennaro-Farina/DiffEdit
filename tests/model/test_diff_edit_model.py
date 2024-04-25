import pytest

from diffusers import LMSDiscreteScheduler
from diff_edit.model.model_composer import ModelComposer


# all right reserved to the owner. Image from wikipedia
@pytest.mark.parametrize("image_path, p, n, seed", [
    ("tests/images/wikipedia_440px-BlkStdSchnauzer2.jpg", ["dog", "cat"], 2, 42),
])
def test_diff_edit_model(image_path, p, n, seed):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)
    diff_edit_model = ModelComposer("stabilityai/sd-vae-ft-ema",
                                    "openai/clip-vit-large-patch14",
                                    "openai/clip-vit-large-patch14",
                                    "CompVis/stable-diffusion-v1-4",
                                    "runwayml/stable-diffusion-inpainting",
                                    scheduler,
                                    torch_dev="best").compose()

    assert diff_edit_model.tokenizer is not None, "Tokenizer is None"
    assert diff_edit_model.text_encoder is not None, "Text encoder is None"
    assert diff_edit_model.vae is not None, "VAE is None"
    assert diff_edit_model.unet is not None, "UNet is None"
    assert diff_edit_model.inpainting is not None, "Inpainting is None"
    assert diff_edit_model.scheduler is not None, "Scheduler is None"

    # This test is taking too long to run on the Github Actions CI workflow. Commenting out for now, runnable locally.
    # out = diff_edit_model.demo_diffedit(image_path, *p, n=2, seed=42)
    # assert out is not None, "Output is None"


