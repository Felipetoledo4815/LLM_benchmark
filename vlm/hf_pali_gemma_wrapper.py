from typing import List, Tuple
from time import time
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import pali_gemma_formatter


class HFPaliGemma(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, use_auth_token=True)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=self.quantization_config,
            device_map=self.device,
            low_cpu_mem_usage=True,
            use_auth_token=True
        )
        if isinstance(model, PaliGemmaForConditionalGeneration):
            self.model = model
        else:
            raise ValueError("Model is not an instance of PaliGemmaForConditionalGeneration.")

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        assert len(images) == 1, "PaliGemma model only supports one image at a time."
        image = Image.open(images[0])

        message = self.parse_prompt(prompt, images, **kwargs)

        start = time()
        inputs = self.processor(message, image, return_tensors="pt")
        # Move the inputs dictionary to the specified device
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        output_str = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        end = time()

        return output_str, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> str:
        rel_questions = kwargs.get("rel_questions", [])
        assert isinstance(rel_questions, list), "Relationship questions need to be a list."
        return pali_gemma_formatter(prompt, images, rel_questions)
