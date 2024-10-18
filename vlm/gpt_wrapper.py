from vlm.vlm_interface import VLMInterface
from typing import List, Tuple
from collections.abc import Iterator
import io
import base64
import time
from PIL import Image

import models 
import json
import os
import time
import openai
from openai import OpenAI
import random
import numpy as np

from utils.prompt_formatter import gpt_formatter

class GPTWrapper(VLMInterface):
    def __init__(self, gpt_model_name):
        with open(os.path.join(os.path.dirname(__file__), '..', 'keys.json'), 'r') as f:
            openai.api_key = json.load(f)['openai']['apiKey']
        
        self.model_name = gpt_model_name
    
    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        start = time.time()
        images_uri = []
        for img in images:
            images_uri.append(self.image_to_base64_data_uri(img))
        prompt = self.parse_prompt(prompt, images, **kwargs)
        result_str_obj = self.openai_query(images_uri, prompt) # commenting this for now - will send the prompt as it is
        # result_str = models.openai_query(prompt, model=self.model_name, attemptd_id=_, max_tries=50)
        result_str=result_str_obj.message.content
        end = time.time()
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

    # def parse_prompt(self, prompt: str, images: List[str], **kwargs):
    #     return super().parse_prompt(prompt, images, **kwargs)
    
    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> str:
        rel_questions = kwargs.get("rel_questions", [])
        assert isinstance(rel_questions, list), "Relationship questions need to be a list."

        return gpt_formatter(prompt, images, rel_questions)
    
    def openai_query(self, img_uri, prompt):
        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_uri[0], # just testing with 1 img
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return(response.choices[0])
