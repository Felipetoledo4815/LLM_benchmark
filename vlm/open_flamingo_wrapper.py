from typing import List, Tuple
from time import time
from PIL import Image
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from torchvision.transforms import Compose
import torch
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import flamingo_formatter


class OpenFlamingo(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.model, image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
            cross_attn_every_n_layers=1
        )
        if isinstance(image_processor, Compose):
            self.image_processor = image_processor
        else:
            raise ValueError("Image processor is not an instance of Compose.")
        checkpoint_path = hf_hub_download(model_name, "checkpoint.pt", local_dir=cache_dir)
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:

        message, images_to_load = self.parse_prompt(prompt, images)

        # Step 1: Load images
        pil_images = []

        for img in images_to_load:
            pil_images.append(Image.open(img))

        for img in images:
            pil_images.append(Image.open(img))

        start = time()
        # Step 2: Preprocessing images
        # Details: For OpenFlamingo, we expect the image to be a torch tensor of shape
        # batch_size x num_media x num_frames x channels x height x width.
        # In this case batch_size = 1, num_media = 3, num_frames = 1, channels = 3, height = 224, width = 224.
        vision_x = []
        for img in pil_images:
            vision_x.append(self.image_processor(img).unsqueeze(0))
        vision_x = torch.cat(vision_x, dim=0) # pylint: disable=no-member
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        # Step 3: Preprocessing text
        # Details: In the text we expect an <image> special token to indicate where an image is.
        # We also expect an <|endofchunk|> special token to indicate the end of the text portion associated with image.
        self.tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
        lang_x = self.tokenizer(
            [message],
            return_tensors="pt",
        )
        input_ids = lang_x["input_ids"]
        assert isinstance(input_ids, torch.Tensor), "Input ids should be a tensor."
        attention_mask = lang_x["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor), "Attention mask should be a tensor."

        # Step 4: Generate text
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.device),
            lang_x=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=150,
            num_beams=3,
        )
        output_str = self.tokenizer.decode(generated_text[0])
        response_str = output_str.split("<|endofchunk|>")[-1]
        end = time()

        return response_str, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, List[str]]:
        return flamingo_formatter(prompt, images)
