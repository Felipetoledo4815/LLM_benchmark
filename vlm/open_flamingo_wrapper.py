from typing import List, Tuple
from PIL import Image
from time import time
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from vlm.vlm_interface import VLMInterface


class OpenFlamingo(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            cross_attn_every_n_layers=1
        )
        checkpoint_path = hf_hub_download(model_name, "checkpoint.pt", local_dir=cache_dir)
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:

        start = time()

        # Step 1: Load images

        image = Image.open(images[0])

        # Step 2: Preprocessing images
        # Details: For OpenFlamingo, we expect the image to be a torch tensor of shape
        # batch_size x num_media x num_frames x channels x height x width.
        # In this case batch_size = 1, num_media = 3, num_frames = 1,
        # channels = 3, height = 224, width = 224.

        # vision_x = [self.image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
        vision_x = [self.image_processor(image).unsqueeze(0)]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        # Step 3: Preprocessing text
        # Details: In the text we expect an <image> special token to indicate where an image is.
        # We also expect an <|endofchunk|> special token to indicate the end of the text
        # portion associated with an image.

        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            # ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
            [f"<image>{prompts[0]}<|endofchunk|>"],
            return_tensors="pt",
        )

        # Step 4: Generate text

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.device),
            lang_x=lang_x["input_ids"].to(self.device),
            attention_mask=lang_x["attention_mask"].to(self.device),
            max_new_tokens=150,
            num_beams=3,
        )
        output_str = self.tokenizer.decode(generated_text[0])
        response_str = output_str.split("<|endoftext|>")[1]

        end = time()

        return response_str, end - start
