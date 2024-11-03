from typing import List, Tuple
from time import time
from transformers import BitsAndBytesConfig
from external_code.LLaVA.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from external_code.LLaVA.llava.model.builder import load_pretrained_model
from PIL import Image
import torch
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import hf_llava_formatter


class LlavaWrapper(VLMInterface):
    def __init__(self, model_path: str, **kwargs) -> None:

        kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )

        self.model_name = "llava-1.5-7b-hf"
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, **kwargs)

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        pil_images = []

        message, images_to_load = self.parse_prompt(prompt, images, **kwargs)

        for img in images_to_load:
            pil_images.append(Image.open(img).convert('RGB'))

        for img in images:
            pil_images.append(Image.open(img).convert('RGB'))

        start = time()

        input_ids = tokenizer_image_token(message, self.tokenizer, -200, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images(pil_images, self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[pil_images[0].size],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=150,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        end = time()

        return outputs, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, List[str]]:
        rel_questions = kwargs.get("rel_questions", [])
        assert isinstance(rel_questions, list), "Relationship questions need to be a list."
        return hf_llava_formatter(prompt, images, self.model_name, rel_questions=rel_questions)
