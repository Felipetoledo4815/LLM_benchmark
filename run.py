import argparse
import io
import base64
import sys
import random
from llama_cpp.llama_chat_format import Llava15ChatHandler
from llama_cpp import Llama
from PIL import Image
from prompts import zero_shot_prompts
from logger import Logger
from metrics import Metrics
#TODO: how to turn this repo into a python module?
sys.path.append('../LLM_SRP/')
from dataset.llm_srp_dataset import LLMSRPDataset


def image_to_base64_data_uri(image_input):
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


def main():
    parser = argparse.ArgumentParser("Benchmark for spatial relationships.")
    parser.add_argument('--nr_images', type=int, default=10, help='Number of images to use for the benchmark.')
    args = parser.parse_args()

    print("Number of images: ", args.nr_images)

    llm_srp_dataset = LLMSRPDataset(['nuscenes'])
    logger = Logger("log.txt", overwrite=False)
    metrics = Metrics()

    # TODO: add wrapper class for other types of models: GPT api, Hugging face transformers, and llama cpp
    mmproj = "./models/mmproj-model-f16.gguf"
    model_path = "./models/ggml-model-q4_0.gguf"
    chat_handler = Llava15ChatHandler(clip_model_path=mmproj, verbose=True)
    spacellava = Llama(model_path=model_path, chat_handler=chat_handler,
                       n_ctx=3072, logits_all=True, n_gpu_layers=-1, verbose=False)

    for i in range(args.nr_images):
        r_n = random.randint(0, len(llm_srp_dataset) - 1)
        img_path, triplets = llm_srp_dataset[r_n]

        data_uri = image_to_base64_data_uri(img_path)
        prompt = zero_shot_prompts['list_of_triplets'][0]['prompt']
        messages = [
            {"role": "system", "content": "You are an assistant that describe images."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        results = spacellava.create_chat_completion(messages=messages, temperature=0.0)
        result_str = results["choices"][0]["message"]["content"]
        sg_iou = metrics.sg_iou(result_str, triplets)
        logger.log(prediction=result_str, ground_truth=triplets, image_id=r_n, sg_iou=sg_iou)

    print("Average IoU: ", metrics.get_avg_sg_iou())


if __name__ == '__main__':
    main()
