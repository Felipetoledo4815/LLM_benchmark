import argparse
import os
import pandas as pd
import numpy as np
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
    "llava_1.6_mistral": "L1.6-Mis",
    "llava_1.6_vicuna": "L1.6-Vic",
    "cambrian-llama3": "C-Llama3",
    "cambrian-phi3": "C-Phi3",
    "spacellava": "SpaceLlaVA",
    "roadscene2vec": "RS2V",
    "llava_1.5": "L1.5",
    "llava_1.5_ft_m1": "L1.5-FT",
    "llava_1.5_lora_m1": "L1.5-L",
    "llava_1.5_ft_m2": "L1.5-FT",
    "llava_1.5_lora_m2": "L1.5-L",
    "llava_1.5_ft_m3": "L1.5-FT",
    "llava_1.5_lora_m3": "L1.5-L",
    "llava_1.5_ft_m4": "L1.5-FT",
    "llava_1.5_lora_m4": "L1.5-L",
    "gpt": "GPT-4-T",
    "paligemma": "PaliGemma",
}


def retrieve_summary_from_log(log_path: str) -> dict:
    with open(log_path, 'r') as f:
        lines = f.readlines()
    model_dir = log_path.split('/')[-2]
    model_name, mode, shot, nr_imgs = model_dir.split("__")
    mode = int(mode.split('_')[1])
    nr_imgs = int(nr_imgs.split("imgs")[0])
    if model_name not in MODEL_NAMES:
        return None
    all_samples = []
    for line in lines:
        if "image_id=" in line:
            aux_dict = {
                "img_id": int(line.split("=")[1].strip())
            }
        elif "image_path=" in line:
            aux_dict["img_path"] = line.split("=")[1].strip()
            if "nuscenes" in aux_dict["img_path"]:
                aux_dict["dataset"] = "nuScenes"
            elif "kitti" in aux_dict["img_path"]:
                aux_dict["dataset"] = "KITTI"
            elif "waymo" in aux_dict["img_path"]:
                aux_dict["dataset"] = "Waymo"
        elif "time=" in line:
            aux_dict["time"] = float(line.split("=")[1].strip())
        elif "recall=" in line:
            aux_dict["recall"] = float(line.split("=")[1].strip())
        elif "precision=" in line:
            aux_dict["precision"] = float(line.split("=")[1].strip())
        elif "f1=" in line:
            aux_dict["f1"] = float(line.split("=")[1].strip())
        elif "tp=" in line:
            aux_dict["tp"] = int(line.split("=")[1].strip())
        elif "fp=" in line:
            aux_dict["fp"] = int(line.split("=")[1].strip())
        elif "fn=" in line:
            aux_dict["fn"] = int(line.split("=")[1].strip())
        elif "---" in line:
            all_samples.append(aux_dict)

    df = pd.DataFrame(all_samples)
    total = df.agg({
        'time': 'mean',
        'recall':'mean',
        'precision':'mean',
        'f1':'mean'
    })
    d_metrics = df.groupby('dataset').agg({
        'recall':'mean',
        'precision':'mean',
        'f1':'mean'
    })
    indexes = [np.array(["QM","Time","Total","Total","Total","Kitti","Kitti","Kitti","Waymo","Waymo","Waymo","nuScenes","nuScenes","nuScenes"]), 
               np.array(["","","R","P","F1","R","P","F1","R","P","F1","R","P","F1"])]
    total_values = total.values.flatten()
    d_metrics_values = d_metrics.values.flatten()
    # if len(d_metrics_values) < 9:
    #     return None
    all_values = np.concatenate(([mode], total_values, d_metrics_values)).reshape(1,14)
    result = pd.DataFrame(all_values, index=[MODEL_NAMES[model_name]], columns=indexes)
    result["Mode"] = result["Mode"].astype(int)
    return result

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

    df = pd.concat(data)
    df = df.reset_index().sort_values(by=['Mode', 'index']).set_index('index')
    df = df.reset_index().rename(columns={'index': 'Model'})
    
    # Set display options
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)        # Set the display width to a large number
    pd.options.display.float_format = '{:.2f}'.format

    #TODO: Format the new table

    print("\nTable\n")
    print(df)

    print("Latex Table\n")
    # df = df.rename(columns={"model_name": "Model Name", "mode": "Mode", "nr_imgs": "\\# Imgs",
    #                         "Average time per prediction": "\\makecell{Avg time \\\\ prediction}",
    #                         "Average recall": "\\makecell{Avg \\\\ recall}",
    #                         "Average precision": "\\makecell{Avg \\\\ precision}",
    #                         "Macro Average F1": "\\makecell{Avg F1}",
    #                         "Micro Average F1": "\\makecell{Micro \\\\ Avg F1}",
    #                         "Total TP": "TP",
    #                         "Total FP": "FP",
    #                         "Total FN": "FN"})

    # # Round values from colum "\\makecell{Avg F1}" to 3 decimal places
    # df["\\makecell{Avg F1}"] = df["\\makecell{Avg F1}"].round(3)

    # # Convert TP, FP, FN to int
    # df["TP"] = df["TP"].astype(int)
    # df["FP"] = df["FP"].astype(int)
    # df["FN"] = df["FN"].astype(int)

    # Bold max values for Avg F1
    for m in range(1,5):
        for f1 in [("Total", "F1"), ("Kitti", "F1"), ("Waymo", "F1"), ("nuScenes", "F1")]:
            index = df[df["Mode"] == m].sort_values(f1, ascending=False).index[0]
            value = df.loc[index, f1]
            df.loc[index, f1] = f"\\textbf{{{value:.2f}}}"

        last_index = df[df["Mode"] == m].index[-1]
        value_fn = df.loc[last_index, ("nuScenes", "F1")]
        df.loc[last_index, ("nuScenes", "F1")] = f"{value_fn:.2f} \\\\ \\hline"

    latex_text = df.to_latex(index=False, escape=False)
    latex_text = latex_text.replace('\\\\ \\hline \\\\', "\\\\ \\hline")
    latex_text = latex_text.replace('lrrrrlrrlrrlrrl', 'l|c|c|ccc|ccc|ccc|ccc')
    latex_text = latex_text.replace('\multicolumn{3}{l}{Total}', '\multicolumn{3}{c|}{Total}')
    latex_text = latex_text.replace('\multicolumn{3}{l}{Kitti}', '\multicolumn{3}{c|}{Kitti}')
    latex_text = latex_text.replace('\multicolumn{3}{l}{Waymo}', '\multicolumn{3}{c|}{Waymo}')
    latex_text = latex_text.replace('\multicolumn{3}{l}{nuScenes}', '\multicolumn{3}{c}{nuScenes}')
    latex_text = latex_text.replace('\multicolumn{3}{c}{nuScenes} \\\\','\multicolumn{3}{c}{nuScenes} \\\\ & &')
    print(latex_text)

if __name__ == '__main__':
    main()
