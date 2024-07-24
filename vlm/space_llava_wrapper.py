from typing import List, Tuple
from collections.abc import Iterator
import io
import base64
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from time import time
from vlm.vlm_interface import VLMInterface


class SpaceLlavaWrapper(VLMInterface):
    def __init__(self, clip_path: str,
                 model_path: str,
                 n_ctx: int = 2048,
                 logits_all: bool = True,
                 n_gpu_layers: int = -1,
                 verbose: bool = False) -> None:
        self.chat_handler = Llava15ChatHandler(clip_model_path=clip_path, verbose=True)
        self.spacellava = Llama(model_path=model_path, chat_handler=self.chat_handler,
                                n_ctx=n_ctx, logits_all=logits_all, n_gpu_layers=n_gpu_layers, verbose=verbose)

    def inference(self, prompts: List[str], images: List[str]) -> Tuple[str, float]:
        start = time()
        data_uri = self.image_to_base64_data_uri(images[0])
        messages = [
            {"role": "system", "content": "You are an assistant that describe images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]
        results = self.spacellava.create_chat_completion(messages=messages, temperature=0.0)
        if isinstance(results, Iterator):
            raise NotImplementedError("How to handle CreateChatCompletionStreamResponse has not been implemented.")
        else:
            result_str = results["choices"][0]["message"]["content"]
            if result_str is None:
                result_str = ""
        end = time()
        return result_str, end - start

    def image_to_base64_data_uri(self, image_input: str | Image.Image) -> str:
        # Check if the input is a file path (string)
        if isinstance(image_input, str):
            with open(image_input, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Check if the input is a PIL Image
        elif isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(buffer, format="PNG")  # You can change the format if needed
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        else:
            raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")

        return f"data:image/png;base64,{base64_data}"
