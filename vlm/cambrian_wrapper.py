from typing import List, Tuple
from time import time
from external_code.cambrian.cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from external_code.cambrian.cambrian.conversation import conv_templates
from external_code.cambrian.cambrian.model.builder import load_pretrained_model
from external_code.cambrian.cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import os
import torch
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import cambrian_formatter


class CambrianWrapper(VLMInterface):
    def __init__(self, model_type: str, **kwargs) -> None:
        assert model_type in ["llama_3", "phi3"], "Invalid conversation mode."

        self.conv_mode = model_type
        if model_type == "llama_3":
            model_path = os.path.expanduser("nyu-visionx/cambrian-8b")
        elif model_type == "phi3":
            model_path = os.path.expanduser("nyu-visionx/cambrian-phi3-3b")

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name)

    def process(self, image, question, tokenizer, image_processor, model_config):
        qs = question

        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        return input_ids, image_tensor, image_size, prompt

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        pil_images = []

        message = self.parse_prompt(prompt, images, **kwargs)

        for img in images:
            pil_images.append(Image.open(img).convert('RGB'))

        start = time()
        
        input_ids, image_tensor, image_sizes, prompt = self.process(pil_images[0], message, self.tokenizer, self.image_processor, self.model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        end = time()

        return outputs, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, List[str]]:
        rel_questions = kwargs.get("rel_questions", [])
        assert isinstance(rel_questions, list), "Relationship questions need to be a list."
        return cambrian_formatter(prompt, images, rel_questions=rel_questions)
