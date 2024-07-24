from typing import Tuple, List
from pathlib import Path
import datetime
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

    def log_sample(self, prediction: str,
                   ground_truth: str,
                   image_id: int,
                   metrics: Tuple[float, float, float],
                   time: float | None = None) -> None:
        ground_truth_string = " [\n"+',\n'.join(['('+', '.join(item)+')' for item in ground_truth])+"\n]"
        log_entry = (f'image_id={image_id}\n' +
                    (f'time={time:.2f}\n' if time is not None else '') +
                    f'recall={metrics[0]:.3f}\n' +
                    f'precision={metrics[1]:.3f}\n' +
                    f'f1={metrics[2]:.3f}\n' +
                    f'prediction={prediction}\n' +
                    f'ground_truth={ground_truth_string}\n\n' +
                    '---\n\n')
        with open(self.log_folder / "log.txt", 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def log_summary(self, metrics: Metrics, avg_time_spent: float | None = None) -> None:
        with open(self.log_folder / "log.txt", 'a', encoding='utf-8') as f:
            f.write('### Log Summary ###\n')
            if avg_time_spent is not None:
                f.write(f'Average time per prediction: {avg_time_spent:.2f}\n')
            f.write(f'Average recall: {metrics.get_avg_recall():.3f}\n')
            f.write(f'Average precision: {metrics.get_avg_precision():.3f}\n')
            f.write(f'Macro Average F1: {metrics.get_macro_avg_f1():.3f}\n')
            f.write(f'Weighted Average F1: {metrics.get_weighted_avg_f1():.3f}\n')

    def get_log_folder(self) -> Path:
        return self.log_folder
