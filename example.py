import os
import json
import torch
import typing
from diffusers import FluxPipeline

from model import T5EncoderModel, FluxTransformer2DModel

def run_flux_pipeline_batch(
    prompts: typing.List[str],
    output_names: typing.List[str] = None,
    output_folder: str = "output",
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 50,
    max_sequence_length: int = 512,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    quant_model_id: str = "HighCWu/FLUX.1-dev-4bit",
    seed: int = None,
    metadata: str = 'metadata',
) -> None:
    """
    Run the Flux pipeline on a batch of prompts and save the results to disk.

    Args:
        prompts: A list of prompts to generate images for.
        output_names: A list of output file names to save the images as. If None, the images will be named
            "flux-dev-4bit_<n>.png" where n is the number of images saved so far.
        output_folder: The folder to save the images in. If None, the images will be saved in the current working directory.
        height: The height of the generated images in pixels.
        width: The width of the generated images in pixels.
        guidance_scale: The guidance scale for the text-to-image model.
        num_inference_steps: The number of inference steps to take for each prompt.
        max_sequence_length: The maximum sequence length to use for the text-to-image model.
        model_id: The ID of the base model to use for the pipeline.
        quant_model_id: The ID of the quantized model to use for the pipeline.
        seed: An optional seed to use for the PyTorch random number generator. If None, no seed will be used.
        metadata: An optional folder to save metadata about the generated images in. If None, no metadata will be saved.

    Returns:
        None
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if metadata and not os.path.exists(metadata):
        os.makedirs(metadata)

    # Load the model components once
    text_encoder_2 = T5EncoderModel.from_pretrained(
        quant_model_id,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        quant_model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=text_encoder_2,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.remove_all_hooks()
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    # Ensure output_names matches prompts in length
    if output_names is None:
        output_names = [None] * len(prompts)

    for prompt, output_name in zip(prompts, output_names):
        print(f"Processing prompt: {prompt[:150]}...")  # Display part of the prompt for context
        
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed) if isinstance(seed, int) else None,
        ).images[0]

        if output_name:
            output_path = os.path.join(output_folder, output_name)
        else:
            output_files = len([f for f in os.listdir(output_folder) if f.endswith(".png")])
            output_name = f"flux-dev-4bit_{output_files}.png"
            output_path = os.path.join(output_folder, output_name)

        image.save(output_path)
        print(f"Saved to: {output_path}")

        # Save metadata
        if metadata:
            metadata_path = os.path.join(metadata, output_name.replace(".png", ".json"))
            metadata_dict = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "max_sequence_length": max_sequence_length,
                "model_id": model_id,
                "quant_model_id": quant_model_id,
                "seed": seed,
                "gpu": torch.cuda.get_device_name(0),
                'output_path': output_path,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=4)

    # Release GPU memory after processing all prompts
    del text_encoder_2, transformer, pipe, image  # Delete model and image variables
    torch.cuda.empty_cache()  # Clear PyTorch CUDA cache

prompts = [
    "old man with glasses portrait, photo, 50mm, f1.4, natural light, Pathéchrome",
    "A realistic, best quality, extremely detailed, ray tracing, photorealistic, A blue cat holding a sign that says hello world",
    "A playful and photorealistic image of a golden retriever wearing a chef's hat and apron, standing at a kitchen counter making pizza. The counter is covered in flour, and the dog's paw is lifting a slice with melted cheese stretching. Include realistic fur texture and playful eyes.",
    "A beautiful, anime-style portrait of a young girl with expressive eyes, wearing a futuristic astronaut helmet, vibrant orange and pink flowers surrounding the helmet, surreal composition",
    "A girl astronaut exploring the cosmos, floating among planets and stars, high quality detail, anime screencap, studio Ghibli style, illustration, high contrast, masterpiece, best quality",
    "Masterpiece, best quality, girl, collarbone, wavy hair, looking at viewer, blurry foreground, upper body, necklace, contemporary, plain pants, intricate, print, pattern, ponytail, freckles, red hair, dappled sunlight, smile, happy",
]

run_flux_pipeline_batch(prompts=prompts)