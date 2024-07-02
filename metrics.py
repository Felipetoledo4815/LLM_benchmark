from typing import List, Tuple
import re


class Metrics:
    def __init__(self) -> None:
        self.samples_count = 0
        self.total_sg_iou = 0

    def __parse_string_to_sg__(self, string: str) -> List[Tuple[str, str, str]] | List:
        # Pattern to match tuples like (element, relation, element)
        pattern = r"\(([^,]+), ([^,]+), ([^,]+)\)"
        # Find all matches of the pattern
        sg_list = re.findall(pattern, string)
        #TODO: what to return if the output is not formatted correctly? Now [] is returned.
        # Check if the output is formatted correctly
        if not isinstance(sg_list, List):
            return []
        for item in sg_list:
            if not (isinstance(item, Tuple) and len(item) == 3 and all(isinstance(elem, str) for elem in item)):
                return []
        return sg_list

    def __sg_list_to_lower_key__(self, sg_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        formatted_list = []
        for tup in sg_list:
            # Convert each string in the tuple to lowercase and create a new tuple
            formatted_tup = tuple(elem.lower() for elem in tup)
            formatted_list.append(formatted_tup)
        return formatted_list

    def sg_iou(self, pred: str, target: List) -> float:
        pred_list = self.__sg_list_to_lower_key__(self.__parse_string_to_sg__(pred))
        target_list = self.__sg_list_to_lower_key__(target)
        # Calculate intersection
        intersection_count = 0
        for item in set(pred_list):
            intersection_count += min(pred_list.count(item), target_list.count(item))
        # Use the size of the target list for IoU calculation
        iou = intersection_count / len(target_list) if len(target_list) > 0 else 0
        self.total_sg_iou += iou
        self.samples_count += 1
        return iou

    def get_avg_sg_iou(self) -> float:
        return self.total_sg_iou / self.samples_count if self.samples_count > 0 else 0
