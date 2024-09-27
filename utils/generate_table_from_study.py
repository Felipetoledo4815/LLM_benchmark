import argparse
import os
import pandas as pd
from pathlib import Path


NUMBER_WORDS = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10
}
MODEL_NAMES = {
    "llava_1.6_mistral": "Llava 1.6 Mistral",
    "spacellava": "SpaceLlava",
    "llava_1.5": "Llava 1.5",
    "llava_1.5_ft_v1": "Llava 1.5 FT v1",
    "llava_1.5_ft": "Llava 1.5 FT"
}


def retrieve_summary_from_log(log_path: str) -> dict:
    with open(log_path, 'r') as f:
        lines = f.readlines()
    model_dir = log_path.split('/')[-2]
    model_name, mode, shot, nr_imgs = model_dir.split("__")
    nr_imgs = int(nr_imgs.split("imgs")[0])
    summary = {
        "model_name": MODEL_NAMES[model_name],
        "mode": int(mode.split('_')[1]),
        "shot": NUMBER_WORDS[shot.split('_')[0].lower()],
        "nr_imgs": nr_imgs
    }
    flag = False
    for line in lines:
        if "### Log Summary ###" in line:
            flag = True
        if flag and ":" in line:
            metric = line.split(": ")[0]
            value = float(line.split(": ")[1].strip())
            summary[metric] = value
    return summary

def main():
    parser = argparse.ArgumentParser("Generate table from study.")
    parser.add_argument('--study_dir', type=Path, help='Path to study directory.')
    args = parser.parse_args()

    data = []

    for directory in os.listdir(args.study_dir):
        if os.path.isdir(os.path.join(args.study_dir, directory)):
            log_path = os.path.join(args.study_dir, directory, 'log.txt')
            summary = retrieve_summary_from_log(log_path)
            data.append(summary)

    df = pd.DataFrame(data)
    df["Macro Average F1"] = 2 * ( (df["Average recall"] * df["Average precision"]) / (df["Average recall"] + df["Average precision"]) )

    print("\nTable\n")
    df = df.sort_values(by=['mode', 'model_name', 'shot'])
    print(df)

    print("Latex Table\n")
    df = df.rename(columns={"model_name": "Model Name", "mode": "Mode", "shot": "Shot", "nr_imgs": "\\# Imgs",
                            "Average time per prediction": "\\makecell{Avg time \\\\ prediction}",
                            "Average recall": "\\makecell{Avg \\\\ recall}",
                            "Average precision": "\\makecell{Avg \\\\ precision}",
                            "Macro Average F1": "\\makecell{Avg F1}",
                            "Micro Average F1": "\\makecell{Micro \\\\ Avg F1}",
                            "Total TP": "TP",
                            "Total FP": "FP",
                            "Total FN": "FN"})
    print(df.to_latex(index=False, float_format="%.2f"))

if __name__ == '__main__':
    main()
