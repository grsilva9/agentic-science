from pathlib import Path
from schemas import GenerateInput, GenerateOutput
from models.ddpm import DDPM
import torch

def generate_task(input: GenerateInput) -> GenerateOutput:
    """
    1. Loads trained model.
    2. Generates synthetic images.
    3. Stores images in a specified folder. 
    """
    #1. Loads trained model. 
    model_checkpoint_path: Path = input.model_checkpoint_path
    model = DDPM.load_from_checkpoint(str(model_checkpoint_path))

    #2. Generates samples.
    samples = model.gen_sample(N = input.n_samples)

    #3. Stores samples.
    parent_path = model_checkpoint_path.parent.parent
    gen_images_dir = parent_path / "generated_samples"

    gen_images_dir.mkdir(parents= True, exist_ok= True)
    gen_images_path = gen_images_dir / "{}.pt".format(input.samples_name)

    torch.save(samples, str(gen_images_path))
    return GenerateOutput(gen_images_path = gen_images_path)