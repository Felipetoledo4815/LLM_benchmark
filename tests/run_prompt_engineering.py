import argparse
import sys
from pathlib import Path
from utils.utils import get_model
from utils.metrics import Metrics
from utils.query_mode import QueryMode
from LLM_SRP.dataset.llm_srp_dataset import LLMSRPDataset


def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text()


def main():
    parser = argparse.ArgumentParser("Prompt engineering for different models and modes.")
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
    parser.add_argument("--img_idx", type=int, default=0, help='Image index to use for inference.')
    parser.add_argument("--prompt_path", type=Path, default=Path("./tests/prompt.txt"),
                        help='Path to prompts .txt file.')
    args = parser.parse_args()

    llm_srp_dataset = LLMSRPDataset(['nuscenes'], configs={'nuscenes': 'nuscenes_mini'})

    vlm = get_model(args.model, args.lora)
    mode = QueryMode(args.mode, vlm, llm_srp_dataset.get_entity_names(),
                        llm_srp_dataset.get_relationship_names())
    metrics = Metrics(llm_srp_dataset.get_entity_names(), llm_srp_dataset.get_relationship_names())

    while True:
        img_path, triplets, bboxes = llm_srp_dataset[args.img_idx]
        print("Image path:", img_path)
        assert isinstance(img_path, str), "Image path needs to be a string."

        prompt = load_prompt(args.prompt_path)
        llm_output, predicted_triplets, time_per_image = mode.query(prompt, [img_path], bboxes)
        recall, precision, f1, tp, fp, fn, metrics_dict = metrics.calculate_metrics(predicted_triplets, triplets)

        print("Time per image:", time_per_image)
        print("LLM output:", llm_output)
        print("Predicted triplets:", predicted_triplets)
        print("Ground truth triplets:", triplets)

        print("---")

        print(f"Average time per prediction: {time_per_image:.2f}")
        print(f"Recall: {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"F1: {f1:.3f}")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")

        user_input = input("Click 'Enter' to reload prompt or type 'q' to exit")
        if user_input == 'q':
            print("Exiting")
            sys.exit(0)


if __name__ == '__main__':
    main()
