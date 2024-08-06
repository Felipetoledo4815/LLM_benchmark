from typing import List, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Metrics:
    def __init__(self, entity_names: List | None = None, relationship_names: List | None = None) -> None:
        self.samples_count = 0
        self.total_recall = 0
        self.total_precision = 0
        self.total_f1 = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.precision_heatmap = None
        self.recall_heatmap = None
        if entity_names is not None and relationship_names is not None:
            df = pd.DataFrame(index=[rn.lower() for rn in relationship_names])
            for entity in entity_names[:-1]:
                df[entity.lower()] = 0.0
            self.precision_heatmap = df.copy()
            self.recall_heatmap = df.copy()
            self.count_matrix = df.copy()

    def __sg_list_to_lower_key__(self, sg_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        formatted_list = []
        for tup in sg_list:
            # Convert each string in the tuple to lowercase and create a new tuple
            formatted_tup = tuple(elem.lower() for elem in tup)
            formatted_list.append(formatted_tup)
        return formatted_list

    def calculate_recall(self, pred: List, target_list: List) -> float:
        """
        Calculate recall for a single prediction.
        :param pred: Prediction in string or list format.
        :param target_list: Target in list format.
        :return: Recall
        Recall = (Intersection of prediction and target_list) / len(target_list)
        """
        assert isinstance(pred, list), "Prediction must be a list of triplets"
        pred_list = self.__sg_list_to_lower_key__(pred)
        target_list = self.__sg_list_to_lower_key__(target_list)
        # Calculate intersection
        intersection_count = 0
        for item in set(pred_list):
            intersection_count += min(pred_list.count(item), target_list.count(item))
        if len(target_list) == 0:
            recall = 1.0 if len(pred_list) == 0 else 0.0
        else:
            recall = intersection_count / len(target_list)
        return recall

    def calculate_precision(self, pred: List, target_list: List) -> float:
        """
        Calculate precision for a single prediction.
        :param pred: Prediction in string or list format.
        :param target_list: Target in list format.
        :return: Precision
        Precision = (Intersection of prediction and target_list) / len(prediction)
        """
        assert isinstance(pred, list), "Prediction must be a list of triplets"
        pred_list = self.__sg_list_to_lower_key__(pred)
        target_list = self.__sg_list_to_lower_key__(target_list)
        # Calculate intersection
        intersection_count = 0
        for item in set(pred_list):
            intersection_count += min(pred_list.count(item), target_list.count(item))
        if len(pred_list) == 0:
            precision = 1.0 if len(target_list) == 0 else 0.0
        else:
            precision = intersection_count / len(pred_list)
        return precision

    def get_tp_fp_fn(self, pred: List, target_list: List) -> Tuple[int, int, int]:
        """
        Get True Positives (TP), False Positives (FP), and False Negatives (FN) for a single prediction.
        :param pred: Prediction in string or list format.
        :param target_list: Target in list format.
        :return: Precision
        """
        assert isinstance(pred, list), "Prediction must be a list of triplets"
        pred_list = self.__sg_list_to_lower_key__(pred)
        target_list = self.__sg_list_to_lower_key__(target_list)
        # Calculate intersection
        intersection_count = 0
        for item in set(pred_list):
            intersection_count += min(pred_list.count(item), target_list.count(item))
        tp = intersection_count
        fp = len(pred_list) - tp
        fn = len(target_list) - tp
        return tp, fp, fn

    def __calculate_f1_with_precision_recall(self, precision: float, recall: float) -> float:
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def __calculate_f1_with_pred_target(self, pred: List, target_list: List) -> float:
        precision = self.calculate_precision(pred, target_list)
        recall = self.calculate_recall(pred, target_list)
        return self.__calculate_f1_with_precision_recall(precision, recall)

    def calculate_f1(self, *args, **kwargs) -> float:
        if (len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list)) or \
                (len(kwargs) == 2 and 'pred' in kwargs and 'target' in kwargs):
            if len(args) == 2:
                pred, target = args
            else:
                pred = kwargs['pred']
                target = kwargs['target']
                assert isinstance(pred, list), "Prediction must be a list of triplets"
                assert isinstance(target, list), "Target must be a list of triplets"
            return self.__calculate_f1_with_pred_target(pred, target)
        elif (len(args) == 2 and isinstance(args[0], float) and isinstance(args[1], float)) or \
                (len(kwargs) == 2 and 'precision' in kwargs and 'recall' in kwargs):
            if len(args) == 2:
                precision, recall = args
            else:
                precision = kwargs['precision']
                recall = kwargs['recall']
                assert isinstance(precision, float), "Precision must be a float"
                assert isinstance(recall, float), "Recall must be a float"
            return self.__calculate_f1_with_precision_recall(precision, recall)
        else:
            raise ValueError("Invalid arguments provided: " + str(args) + str(kwargs))

    def update_heatmaps(self, pred: List, target_list: List) -> None:
        """
        Update precision and recall heatmaps.
        :param pred: Prediction in string or list format.
        :param target_list: Target in list format.
        """
        if self.precision_heatmap is None or self.recall_heatmap is None:
            return
        assert isinstance(pred, list), "Prediction must be a list of triplets"
        pred_list = self.__sg_list_to_lower_key__(pred)
        target_list = self.__sg_list_to_lower_key__(target_list)
        for target in set(target_list):
            triplet_tp = min(pred_list.count(target), target_list.count(target))
            count_triplet_in_gt = target_list.count(target)
            count_triplet_in_pred = pred_list.count(target)
            self.precision_heatmap.loc[target[1], target[0]] += triplet_tp / \
                count_triplet_in_pred if count_triplet_in_pred > 0 else 0
            self.recall_heatmap.loc[target[1], target[0]] += triplet_tp / \
                count_triplet_in_gt if count_triplet_in_gt > 0 else 0
            self.count_matrix.loc[target[1], target[0]] += 1

    def calculate_metrics(self, pred_list: List, target_list: List) -> Tuple[float, float, float, int, int, int]:
        """
        Calculate recall and precision for a single prediction.
        :param pred: Prediction in string format.
        :param target_list: Target in list format.
        """
        self.samples_count += 1
        recall = self.calculate_recall(pred_list, target_list)
        self.total_recall += recall
        precision = self.calculate_precision(pred_list, target_list)
        self.total_precision += precision
        f1 = self.calculate_f1(precision, recall)
        self.total_f1 = f1
        tp, fp, fn = self.get_tp_fp_fn(pred_list, target_list)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.update_heatmaps(pred_list, target_list)
        return recall, precision, f1, tp, fp, fn

    def get_avg_recall(self) -> float:
        assert self.samples_count > 0, "At least one sample is required"
        return self.total_recall / self.samples_count

    def get_avg_precision(self) -> float:
        assert self.samples_count > 0, "At least one sample is required"
        return self.total_precision / self.samples_count

    def get_macro_avg_f1(self) -> float:
        assert self.samples_count > 0, "At least one sample is required"
        return self.total_f1 / self.samples_count

    def get_micro_avg_f1(self) -> float:
        assert (self.total_tp + self.total_fp + self.total_fn) > 0, "One of TP, FP, and FN must be greater than 0"
        return self.total_tp / (self.total_tp + 0.5 * (self.total_fp + self.total_fn))

    def plot_heatmaps(self, out_path: str | Path | None = None) -> None:
        if self.precision_heatmap is None or self.recall_heatmap is None:
            return
        _, axes = plt.subplots(1, 2, figsize=(20, 10))
        assert isinstance(axes, np.ndarray), "Axes must be a numpy array"
        aux_count_matrix = self.count_matrix.replace(0, 1, inplace=False)
        mask_matrix = self.count_matrix == 0
        avg_p_heatmap = self.precision_heatmap / aux_count_matrix
        avg_p_heatmap[mask_matrix] = -1
        sns.heatmap(avg_p_heatmap, annot=True, fmt=".2f", ax=axes[0])
        axes[0].set_title("Precision Heatmap")
        avg_r_heatmap = self.recall_heatmap / aux_count_matrix
        avg_r_heatmap[mask_matrix] = -1
        sns.heatmap(avg_r_heatmap, annot=True, fmt=".2f", ax=axes[1])
        axes[1].set_title("Recall Heatmap")

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(str(out_path))
        else:
            plt.show()
