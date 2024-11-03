import argparse
import random
import math
from pathlib import Path
from tqdm import tqdm
from utils.logger import Logger
from utils.metrics import Metrics
from utils.query_mode import QueryMode
from utils.utils import get_prompt, get_model
from LLM_SRP.dataset.llm_srp_dataset import LLMSRPDataset


def define_experiment(args: argparse.Namespace, max_num_images) -> str:
    if args.nr_images > max_num_images:
        exp_id = f"{args.model}__mode_{args.mode}__{args.shot}_shot__all"
    else:
        exp_id = f"{args.model}__mode_{args.mode}__{args.shot}_shot__{args.nr_images}imgs"
    print(f"Running experiment: {exp_id}")
    print("Model: ", args.model)
    print("Mode: ", args.mode)
    print("Shots: ", args.shot)
    print("Number of images: ", f"All ({max_num_images})" if args.nr_images > max_num_images else args.nr_images)
    return exp_id


def main():
    parser = argparse.ArgumentParser("Benchmark for spatial relationships.")
    parser.add_argument('--nr_images', type=int, default=10, help='Number of images to use for the benchmark.')
    parser.add_argument('--log_folder', type=Path, default='./logs/test/', help='Log file to store the results.')
    parser.add_argument("--model", type=str,
                        choices=['spacellava', 'llava_1.5',
                                 'llava_1.5_ft_m1', 'llava_1.5_ft_m2', 'llava_1.5_ft_m3', 'llava_1.5_ft_m4',
                                 'llava_1.5_lora_m1', 'llava_1.5_lora_m2', 'llava_1.5_lora_m3', 'llava_1.5_lora_m4',
                                 'llava_1.6_mistral', 'llava_1.6_mistral_ft', 'llava_1.6_vicuna',
                                 "paligemma", "openflamingo", "roadscene2vec", "cambrian-llama3", "cambrian-phi3"],
                        default='llava_1.5', help='Model to use for inference.')
    parser.add_argument("--lora", type=Path, default=None, help='Path to the LoRA model.')
    parser.add_argument("--mode", type=str, choices=['1', '2', '3', '4'],
                        default='1', help='Mode to use for inference.')
    parser.add_argument("--shot", type=str, choices=['zero', 'one', 'two'], default='zero',
                        help='Number of shots to use for inference.')
    parser.add_argument("--even_sample", action='store_true', help='Sample even number of images from each dataset.')
    parser.add_argument("--gpt_exp", action='store_true', help='Benchmark for GPT-4.')
    args = parser.parse_args()

    if args.gpt_exp:
        llm_srp_dataset = LLMSRPDataset(["nuscenes", "kitti", "waymo_training"], configs={
            "nuscenes": "nuscenes",
            "kitti": "kitti_training",
            "waymo_training": "waymo_training"
            })
    else:
        # llm_srp_dataset = LLMSRPDataset(['nuscenes'], configs={'nuscenes': 'nuscenes_mini'})
        llm_srp_dataset = LLMSRPDataset(["nuscenes", "kitti", "waymo_training", "waymo_validation"], configs={
                "nuscenes": "nuscenes",
                "kitti": "kitti_training",
                "waymo_training": "waymo_training",
                "waymo_validation": "waymo_validation",
                })
        llm_srp_dataset.set_split('test')

    max_num_images = min(args.nr_images, len(llm_srp_dataset))
    exp_id = define_experiment(args, max_num_images)

    logger = Logger(args.log_folder/exp_id, datasets_used=llm_srp_dataset.get_dataset_names())
    metrics = Metrics(llm_srp_dataset.get_entity_names(), llm_srp_dataset.get_relationship_names())

    vlm = get_model(args.model, args.lora)
    mode = QueryMode(args.mode, vlm, llm_srp_dataset.get_entity_names(),
                     llm_srp_dataset.get_relationship_names())

    total_time = 0
    random.seed(0)
    if args.even_sample:
        samples = []
        n_samples = math.floor(max_num_images / 3)
        for d in ['nuscenes', 'kitti', 'waymo_training']:
            d_test_samples = []
            for idx, i in enumerate(llm_srp_dataset.indices):
                if i >= llm_srp_dataset.dataset_limits[d][0] and i < llm_srp_dataset.dataset_limits[d][1]:
                    d_test_samples.append(idx)
            samples.extend(random.sample(d_test_samples, n_samples))
    else:
        # Sample random images from the dataset
        samples = random.sample(range(len(llm_srp_dataset)), max_num_images)

    for i in tqdm(samples):
        img_path, triplets, bboxes = llm_srp_dataset[i]

        assert isinstance(img_path, str), "Image path needs to be a string."

        prompt = get_prompt(args.mode, args.model, args.shot)
        llm_output, predicted_triplets, time_per_image = mode.query(prompt, [img_path], bboxes)

        total_time += time_per_image
        recall, precision, f1, tp, fp, fn, metrics_dict = metrics.calculate_metrics(predicted_triplets, triplets)
        logger.log_sample(prediction=llm_output,
                          parsed_prediction=predicted_triplets,
                          ground_truth=triplets,
                          image_id=i,
                          image_path=img_path,
                          metrics=(recall, precision, f1),
                          tp=tp,
                          fp=fp,
                          fn=fn,
                          metrics_dict=metrics_dict,
                          time=time_per_image)

    logger.log_metric(metrics)
    _,_ = metrics.plot_heatmaps(logger.get_log_folder() / "heatmaps.png")
    avg_time_spent = total_time / args.nr_images

    print(f"Average time per prediction: {avg_time_spent:.2f}")
    print(f"Average recall: {metrics.get_avg_recall():.3f}")
    print(f"Average precision: {metrics.get_avg_precision():.3f}")
    print(f"Macro Average F1: {metrics.get_macro_avg_f1():.3f}")
    print(f"Micro Average F1: {metrics.get_micro_avg_f1():.3f}")
    print(f"Total TP: {metrics.total_tp}")
    print(f"Total FP: {metrics.total_fp}")
    print(f"Total FN: {metrics.total_fn}")

    logger.log_summary(metrics, avg_time_spent)


if __name__ == '__main__':
    main()
