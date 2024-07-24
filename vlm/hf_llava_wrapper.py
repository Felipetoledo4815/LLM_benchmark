from typing import List, Tuple
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from time import time
import torch
from vlm.vlm_interface import VLMInterface


class HFLlavaWrapper(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:
        image = Image.open(images[0])

        if self.model_name == "llava-1.5-7b-hf":
            message = "USER:" + prompts[0] + "\n<image>\nASSISTANT:"
        else:
            raise NotImplementedError(f"Prompt for model {self.model_name} has not been implemented.")

        start = time()
        inputs = self.processor(message, image, return_tensors="pt")
        output_ids = self.model.generate(**inputs, max_new_tokens=150)
        output_str = self.processor.decode(output_ids[0][2:], skip_special_tokens=True)
        response_str = output_str.split("ASSISTANT:")[-1]
        end = time()

        return response_str, end - start
