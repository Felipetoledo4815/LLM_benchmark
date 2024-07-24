from typing import List, Tuple
import re

def parse_string_to_sg(string: str) -> List[Tuple[str, str, str]] | List:
    # Pattern to match tuples like (element, relation, element)
    pattern = r"\(([^,]+), ([^,]+), ([^,]+)\)"
    # Find all matches of the pattern
    sg_list = re.findall(pattern, string)
    # Check if the output is formatted correctly
    if not isinstance(sg_list, List):
        return []
    for item in sg_list:
        if not (isinstance(item, Tuple) and len(item) == 3 and all(isinstance(elem, str) for elem in item)):
            return []
    return sg_list
