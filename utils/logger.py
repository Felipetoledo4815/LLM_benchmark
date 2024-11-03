from typing import Tuple, List
from pathlib import Path
import datetime
import json
import pickle
from utils.metrics import Metrics

class Logger:
    def __init__(self, log_folder: str, datasets_used: List[str]) -> None:
        self.log_folder = Path(log_folder)
        self.datasets_used = datasets_used
        if not self.log_folder.exists():
            self.log_folder.mkdir(parents=True, exist_ok=True)
        with open(self.log_folder / "log.txt", 'w', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'### Log started at {timestamp} ###\n')
            f.write(f"Datasets used: {', '.join(self.datasets_used)}\n\n")
        # Delete the metrics file if it exists
        if (self.log_folder / "metrics.json").exists():
            (self.log_folder / "metrics.json").unlink()

    def log_sample(self, prediction: str,
                   parsed_prediction: List[Tuple[str,str,str]],
                   ground_truth: List[Tuple[str,str,str]],
                   image_id: int,
                   image_path: str,
                   metrics: Tuple[float, float, float],
                   tp: int,
                   fp: int,
                   fn: int,
                   metrics_dict: dict,
                   time: float | None = None) -> None:
        ground_truth_string = " [\n"+',\n'.join(['('+', '.join(item)+')' for item in ground_truth])+"\n]"
        log_entry = (f'image_id={image_id}\n' +
                    f'image_path={image_path}\n' +
                    (f'time={time:.2f}\n' if time is not None else '') +
                    f'recall={metrics[0]:.3f}\n' +
                    f'precision={metrics[1]:.3f}\n' +
                    f'f1={metrics[2]:.3f}\n' +
                    f'tp={tp}\n' +
                    f'fp={fp}\n' +
                    f'fn={fn}\n' +
                    f'prediction={prediction}\n' +
                    f'parsed_prediction={parsed_prediction}\n' +
                    f'ground_truth={ground_truth_string}\n\n' +
                    '---\n\n')
        with open(self.log_folder / "log.txt", 'a', encoding='utf-8') as f:
            f.write(log_entry)
        # Add more information to the metrics_dict
        metrics_dict['image_id'] = image_id
        metrics_dict['img_path'] = image_path
        metrics_dict['time'] = time
        # Save the metrics_dict to a json file
        metrics_json_path = self.log_folder / "metrics.json"
        if metrics_json_path.exists():
            with open(metrics_json_path, 'r', encoding='utf-8') as f:
                metrics_json = json.load(f)
            metrics_json.append(metrics_dict)
            with open(metrics_json_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(metrics_json))
        else:
            metrics_json = [metrics_dict]
            with open(self.log_folder / "metrics.json", 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_json))

    def log_summary(self, metrics: Metrics, avg_time_spent: float | None = None) -> None:
        with open(self.log_folder / "log.txt", 'a', encoding='utf-8') as f:
            f.write('### Log Summary ###\n')
            if avg_time_spent is not None:
                f.write(f'Average time per prediction: {avg_time_spent:.2f}\n')
            f.write(f'Average recall: {metrics.get_avg_recall():.3f}\n')
            f.write(f'Average precision: {metrics.get_avg_precision():.3f}\n')
            f.write(f'Macro Average F1: {metrics.get_macro_avg_f1():.3f}\n')
            f.write(f'Micro Average F1: {metrics.get_micro_avg_f1():.3f}\n')
            f.write(f"Total TP: {metrics.total_tp}\n")
            f.write(f"Total FP: {metrics.total_fp}\n")
            f.write(f"Total FN: {metrics.total_fn}\n")

    def log_metric(self, metric: Metrics):
        with open(self.log_folder / "metrics.pkl", 'wb') as f:
            pickle.dump(metric, f) 

    def get_log_folder(self) -> Path:
        return self.log_folder
