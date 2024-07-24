import argparse
import sys
import random
from tqdm import tqdm
from prompts import zero_shot_prompts
from utils.logger import Logger
from utils.metrics import Metrics
from vlm.space_llava_wrapper import SpaceLlavaWrapper
from vlm.mobile_vlm_wrapper import MobileVLM
from vlm.hf_llava_next_wrapper import HFLlavaNextWrapper
from vlm.hf_llava_wrapper import HFLlavaWrapper
from vlm.hf_pali_gemma_wrapper import HFPaliGemma
from vlm.open_flamingo_wrapper import OpenFlamingo
# TODO: how to turn this repo into a python module?
sys.path.append('../LLM_SRP/')
from dataset.llm_srp_dataset import LLMSRPDataset


def main():
    parser = argparse.ArgumentParser("Benchmark for spatial relationships.")
    parser.add_argument('--nr_images', type=int, default=10, help='Number of images to use for the benchmark.')
    parser.add_argument('--log_folder', type=str, default='./logs/test/', help='Log file to store the results.')
    parser.add_argument("--model", type=str,
                        choices=['spacellava', 'llava_1.5', 'llava_1.6_mistral',
                                 'llava_1.6_vicuna', "paligemma", "mobilevlm", "openflamingo"],
                        default='llava_1.5', help='Model to use for inference.')
    args = parser.parse_args()

    print("Number of images: ", args.nr_images)

    llm_srp_dataset = LLMSRPDataset(['nuscenes'], configs={'nuscenes': 'nuscenes_mini'})
    logger = Logger(args.log_folder, datasets_used=llm_srp_dataset.get_dataset_names())
    metrics = Metrics(llm_srp_dataset.get_entity_names(), llm_srp_dataset.get_relationship_names())
    prompt = zero_shot_prompts['list_of_triplets'][0]['prompt']

    if args.model == 'llava_1.5':
        vlm = HFLlavaWrapper("llava-hf/llava-1.5-7b-hf", cache_dir="./models/llava-1.5-7b-hf")
    elif args.model == 'llava_1.6_mistral':
        vlm = HFLlavaNextWrapper("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir="./models/llava-v1.6-mistral-7b-hf")
    elif args.model == 'llava_1.6_vicuna':
        vlm = HFLlavaNextWrapper("llava-hf/llava-v1.6-vicuna-13b-hf", cache_dir="./models/llava-v1.6-vicuna-13b-hf")
    elif args.model == 'spacellava':
        vlm = SpaceLlavaWrapper(clip_path="./models/spacellava/mmproj-model-f16.gguf",
                                model_path="./models/spacellava/ggml-model-q4_0.gguf")
    elif args.model == 'mobilevlm':
        vlm = MobileVLM(clip_path="./models/mobile-vlm/mmproj-model-f16.gguf",
                        model_path="./models/mobile-vlm/ggml-model-q4_k.gguf")
    elif args.model == 'paligemma':
        vlm = HFPaliGemma("google/paligemma-3b-mix-448", cache_dir="./models/paligemma-3b-mix-448")
    elif args.model == 'openflamingo':
        vlm = OpenFlamingo("openflamingo/OpenFlamingo-3B-vitl-mpt1b", cache_dir="./models/OpenFlamingo-3B-vitl-mpt1b")
    else:
        raise ValueError(f"Unknown model {args.model}.")

    total_time = 0
    random.seed(0)
    for _ in tqdm(range(args.nr_images)):
        r_n = random.randint(0, len(llm_srp_dataset) - 1)
        img_path, triplets, _ = llm_srp_dataset[r_n]

        llm_output, time_spent = vlm.inference([prompt], [img_path])
        total_time += time_spent

        recall, precision, f1 = metrics.calculate_metrics(llm_output, triplets)
        logger.log_sample(prediction=llm_output,
                          ground_truth=triplets,
                          image_id=r_n,
                          metrics=(recall, precision, f1),
                          time=time_spent)

    metrics.plot_heatmaps(logger.get_log_folder() / "heatmaps.png")
    avg_time_spent = total_time / args.nr_images

    print(f"Average time per prediction: {avg_time_spent:.2f}")
    print(f"Average recall: {metrics.get_avg_recall():.3f}")
    print(f"Average precision: {metrics.get_avg_precision():.3f}")
    print(f"Macro Average F1: {metrics.get_macro_avg_f1():.3f}")
    print(f"Weighted Average F1: {metrics.get_weighted_avg_f1():.3f}")

    logger.log_summary(metrics, avg_time_spent)


if __name__ == '__main__':
    main()
