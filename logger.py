import datetime


class Logger:
    def __init__(self, log_file: str, overwrite=False) -> None:
        self.log_file = log_file
        self.overwrite = overwrite
        with open(self.log_file, 'w' if self.overwrite else 'a', encoding='utf-8') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'### Log started at {timestamp} ###\n\n')

    def log(self, prediction: str, ground_truth: str, image_id: int, sg_iou: float) -> None:
        #TODO: how to handle image_id if there are more than 1 datasets?
        ground_truth_string = " [\n"+',\n'.join(['('+', '.join(item)+')' for item in ground_truth])+"\n]"
        log_entry = (f'image_id={image_id}\n'
                     f'sg_iou={sg_iou:.3f}\n'
                     f'prediction={prediction}\n'
                     f'ground_truth={ground_truth_string}\n\n'
                     '---\n\n')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
