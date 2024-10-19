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
    "llava_1.6_mistral_ft": "Llava 1.6 Mistral FT",
    "spacellava": "SpaceLlava",
    "roadscene2vec": "RoadScene2Vec",
    "llava_1.5": "Llava 1.5",
    # "llava_1.5_ft_v1": "Llava 1.5 FT v1",
    "llava_1.5_ft": "Llava 1.5 FT",
    "gpt": "GPT-4.0"
}


def retrieve_summary_from_log(log_path: str) -> dict:
    with open(log_path, 'r') as f:
        lines = f.readlines()
    model_dir = log_path.split('/')[-2]
    model_name, mode, shot, nr_imgs = model_dir.split("__")
    nr_imgs = int(nr_imgs.split("imgs")[0])
    if model_name not in MODEL_NAMES:
        return None
    summary = {
        "model_name": MODEL_NAMES[model_name],
        "mode": int(mode.split('_')[1]),
        # "shot": NUMBER_WORDS[shot.split('_')[0].lower()],
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
        if "zero_shot" in directory:
            if os.path.isdir(os.path.join(args.study_dir, directory)):
                log_path = os.path.join(args.study_dir, directory, 'log.txt')
                summary = retrieve_summary_from_log(log_path)
                if summary is None:
                    continue
                else:
                    data.append(summary)

    df = pd.DataFrame(data)
    df["Macro Average F1"] = 2 * ( (df["Average recall"] * df["Average precision"]) / (df["Average recall"] + df["Average precision"]) )

    print("\nTable\n")
    df = df.sort_values(by=['mode', 'model_name'])
    print(df)

    print("Latex Table\n")
    df = df.rename(columns={"model_name": "Model Name", "mode": "Mode", "nr_imgs": "\\# Imgs",
                            "Average time per prediction": "\\makecell{Avg time \\\\ prediction}",
                            "Average recall": "\\makecell{Avg \\\\ recall}",
                            "Average precision": "\\makecell{Avg \\\\ precision}",
                            "Macro Average F1": "\\makecell{Avg F1}",
                            "Micro Average F1": "\\makecell{Micro \\\\ Avg F1}",
                            "Total TP": "TP",
                            "Total FP": "FP",
                            "Total FN": "FN"})

    # Round values from colum "\\makecell{Avg F1}" to 3 decimal places
    df["\\makecell{Avg F1}"] = df["\\makecell{Avg F1}"].round(3)

    # Convert TP, FP, FN to int
    df["TP"] = df["TP"].astype(int)
    df["FP"] = df["FP"].astype(int)
    df["FN"] = df["FN"].astype(int)

    # Bold max values for Avg F1
    for m in range(1,5):
        index = df[df["Mode"] == m].sort_values("\\makecell{Avg F1}", ascending=False).index[0]
        value = df.loc[index, "\\makecell{Avg F1}"]
        df.loc[index, "\\makecell{Avg F1}"] = f"\\textbf{{{value}}}"

        last_index = df[df["Mode"] == m].index[-1]
        value_fn = df.loc[last_index, "FN"]
        df.loc[last_index, "FN"] = f"{value_fn} \\\\ \\hline"

    latex_text = df.to_latex(index=False, escape=False)
    latex_text = latex_text.replace('\\\\ \\hline \\\\', "\\\\ \\hline")
    print(latex_text)

if __name__ == '__main__':
    main()
