from typing import List, Tuple
from time import time
import sys
import torch
from PIL import Image
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import mobile_vlm_formatter
sys.path.append('external_code/MobileVLM')
from external_code.MobileVLM.mobilevlm.model.mobilevlm import load_pretrained_model
from external_code.MobileVLM.mobilevlm.conversation import conv_templates, SeparatorStyle
from external_code.MobileVLM.mobilevlm.utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from external_code.MobileVLM.mobilevlm.constants import IMAGE_TOKEN_INDEX


class MobileVLM2(VLMInterface):
    def __init__(self, model_path: str) -> None:
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path)

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        conv = conv_templates['v1'].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        messages, images_to_load = self.parse_prompt(prompt, images, **kwargs)
        messages = conv.system + " " + messages

        pil_images = []
        for img in images_to_load:
            pil_images.append(Image.open(img))

        for img in images:
            pil_images.append(Image.open(img))

        start = time()
        # Process images
        images_tensor = process_images(pil_images, self.image_processor, self.model.config)
        if isinstance(images_tensor, torch.Tensor):
            images_tensor = images_tensor.to(self.model.device, dtype=torch.float16)
        # Input
        input_ids = tokenizer_image_token(messages, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.unsqueeze(0).cuda()
        else:
            raise ValueError("Input_ids is not a tensor")
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        # Inference
        temperature = 0.0
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria] # type: ignore
            )
        # Result-Decode
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
        results_str = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        results_str = results_str.strip()
        if results_str.endswith(stop_str):
            results_str = results_str[: -len(stop_str)]
        if results_str is None:
            results_str = ""

        end = time()

        return results_str, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, List[str]]:
        rel_questions = kwargs.get("rel_questions", [])
        return mobile_vlm_formatter(prompt, images, rel_questions=rel_questions)
