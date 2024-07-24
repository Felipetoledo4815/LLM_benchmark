from typing import List, Tuple
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from time import time
import torch
from vlm.vlm_interface import VLMInterface


class HFPaliGemma(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.processor = AutoProcessor.from_pretrained(model_name, use_auth_token=True)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_auth_token=True
        )

    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:
        image = Image.open(images[0])

        message = prompts[0]

        start = time()
        inputs = self.processor(message, image, return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        output_str = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        end = time()

        return output_str, end - start
