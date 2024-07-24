from typing import List, Tuple
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from time import time
import torch
from vlm.vlm_interface import VLMInterface

class HFLlavaNextWrapper(VLMInterface):
    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name.split('/')[-1]
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
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:
        image = Image.open(images[0])

        if self.model_name == "llava-v1.6-mistral-7b-hf":
            message = "[INST] " + prompts[0] + "\n<image> [/INST]"
        elif self.model_name == "llava-v1.6-34b-hf":
            message = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n" \
                f"{prompts[0]}\n<image>\n" \
                "<|im_end|><|im_start|>assistant\n"
        elif self.model_name.startswith("llava-v1.6-vicuna-"):
            message = "A chat between a curious human and an artificial intelligence assistant." \
                "The assistant gives helpful, detailed, and polite answers to the human's questions." \
                "USER: " + prompts[0] + " <image> ASSISTANT:"
        else:
            raise NotImplementedError(f"Prompt for model {self.model_name} has not been implemented.")

        start = time()
        inputs = self.processor(message, image, return_tensors="pt")
        # # Workaround for bug in llava 1.6 34b https://huggingface.co/llava-hf/llava-v1.6-34b-hf/discussions/8
        # if self.model_name == "llava-v1.6-34b-hf":
        #     inputs['input_ids'][inputs['input_ids'] == 64003] = 64000
        output_ids = self.model.generate(**inputs, max_new_tokens=150)
        output_str = self.processor.decode(output_ids[0], skip_special_tokens=True)

        # Post-process
        if self.model_name == "llava-v1.6-mistral-7b-hf":
            response_str = output_str.split("[/INST]")[-1]
        elif self.model_name.startswith("llava-v1.6-vicuna-"):
            response_str = output_str.split("ASSISTANT:")[-1]
        end = time()

        return response_str, end - start
