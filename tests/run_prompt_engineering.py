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
                        choices=['spacellava', 'llava_1.5', 'llava_1.5_ft', 'llava_1.6_mistral',
                                 'llava_1.6_vicuna', "paligemma", "mobilevlm", "mobilevlm2", "openflamingo", "gpt"],
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
        assert isinstance(img_path, str), "Image path needs to be a string."

        prompt = load_prompt(args.prompt_path)
        llm_output, predicted_triplets, time_per_image = mode.query(prompt, [img_path], bboxes)
        recall, precision, f1, tp, fp, fn = metrics.calculate_metrics(predicted_triplets, triplets)

        print("Time per image:", time_per_image)
        print("LLM output:", llm_output)
        print("Predicted triplets:", predicted_triplets)
        print("Ground truth triplets:", triplets)

        print("---")

        print(f"Average time per prediction: {time_per_image:.2f}")
        print(f"Average recall: {metrics.get_avg_recall():.3f}")
        print(f"Average precision: {metrics.get_avg_precision():.3f}")
        print(f"Macro Average F1: {metrics.get_macro_avg_f1():.3f}")
        print(f"Micro Average F1: {metrics.get_micro_avg_f1():.3f}")
        print(f"Total TP: {metrics.total_tp}")
        print(f"Total FP: {metrics.total_fp}")
        print(f"Total FN: {metrics.total_fn}")

        user_input = input("Click 'Enter' to reload prompt or type 'q' to exit")
        if user_input == 'q':
            print("Exiting")
            sys.exit(0)


if __name__ == '__main__':
    main()
