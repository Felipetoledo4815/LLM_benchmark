from typing import List, Tuple
from time import time
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
from vlm.vlm_interface import VLMInterface
from utils.prompt_formatter import hf_llava_formatter


class HFLlavaNextWrapper(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        if isinstance(processor, LlavaNextProcessor):
            self.processor = processor
        else:
            raise NotImplementedError(f"Processor for model {self.model_name} is returning an unexpected type.")
        # specify how to quantize the model
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        if isinstance(model, LlavaNextForConditionalGeneration):
            self.model = model
        else:
            raise ValueError("Model is not an instance of LlavaNextForConditionalGeneration.")

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        pil_images = []

        if not (self.model_name in ['llava-v1.6-mistral-7b-hf'] or self.model_name.startswith("llava-v1.6-vicuna-")):
            raise NotImplementedError(f"Prompt for model {self.model_name} has not been implemented.")
        message, images_to_load = self.parse_prompt(prompt, images, **kwargs)

        for img in images_to_load:
            pil_images.append(Image.open(img))

        for img in images:
            pil_images.append(Image.open(img))

        start = time()
        inputs = self.processor(message, pil_images, padding=True, return_tensors="pt")
        # Move the inputs dictionary to the specified device
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        output_ids = self.model.generate(**inputs, max_new_tokens=150)
        output_str = self.processor.decode(output_ids[0], skip_special_tokens=True)

        # Post-process
        if self.model_name == "llava-v1.6-mistral-7b-hf":
            response_str = output_str.split("[/INST]")[-1]
        elif self.model_name.startswith("llava-v1.6-vicuna-"):
            response_str = output_str.split("ASSISTANT:")[-1]
        end = time()

        return response_str, end - start

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, List[str]]:
        rel_questions = kwargs.get("rel_questions", [])
        assert isinstance(rel_questions, list), "Relationship questions need to be a list."
        return hf_llava_formatter(prompt, images, self.model_name, rel_questions=rel_questions)
