from typing import List, Tuple
import re
from vlm.vlm_interface import VLMInterface


def parse_string_to_sg(string: str) -> List[Tuple[str, str, str]] | List:
    # Pattern to match tuples like (element, relation, element)
    pattern = r"\(([^,]+), ([^,]+), ([^,]+)\)"
    # Find all matches of the pattern
    sg_list = re.findall(pattern, string)
    # Check if the output is formatted correctly
    if not isinstance(sg_list, List):
        return []
    sanitized_sg_list = []
    for item in sg_list:
        if not (isinstance(item, Tuple) and len(item) == 3 and all(isinstance(elem, str) for elem in item)):
            print(f"Weird string: {string}. Skipping.")
            return []
        sanitized_item = tuple(elem.strip('\'"') for elem in item)
        sanitized_sg_list.append(sanitized_item)
    return sanitized_sg_list


def parse_string_to_count(string: str) -> int:
    try:
        # Search for the first occurrence of a number in the string after 'yes'
        match = re.search(r'\d+', string)
        if match:
            # Convert the matched number to an integer
            n = int(match.group())
            if 1 <= n <= 10:
                return n
            else:
                return 0
        else:
            if 'yes' in string.lower():
                return 1
            elif 'no' in string.lower():
                return 0
            else:
                return 0
    except ValueError:
        return 0


def get_relationship_questions(entities, relationships) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    questions = []
    triplets = []
    for entity in entities:
        if entity == "ego":
            continue
        for relationship in relationships:
            questions.append(f"{entity} {relationship} of ego")
            triplets.append((entity, relationship, "ego"))
    return triplets, questions


def get_rel_text(rel: str):
    # Replace underscores with spaces
    text_rel = rel.replace("_", " ")

    # Replace 'm' following a number with 'meters'
    text_rel = re.sub(r'(\d+)m', r'\1 meters', text_rel)

    return text_rel


class QueryMode:
    #TODO: Add tests for this class
    def __init__(self, mode: str, vlm: VLMInterface, entity_list: List[str],
                 relationship_list: List[str]) -> None:
        self.mode = getattr(self, f"mode{mode}")
        self.vlm = vlm
        self.entity_list = entity_list
        self.relationship_list = relationship_list

    def query(self, prompt: str, images: List[str]) -> Tuple[str, List[Tuple[str, str, str]], float]:
        """
        Query the model with the given prompt and images based on the mode.
        :param prompt: Prompt to query the model.
        :param images: List of image paths.
        :return:
            llm_output: Output from the model.
            predicted_triplets: List of predicted triplets.
            time_spent: Time spent for the inference.
        """
        return self.mode(prompt, images)

    def mode1(self, prompt: str, images: List[str]) -> Tuple[str, List[Tuple[str, str, str]], float]:
        llm_output, time_spent = self.vlm.inference(prompt, images)
        predicted_triplets = parse_string_to_sg(llm_output)
        return llm_output, predicted_triplets, time_spent

    def mode2(self, prompt: str, images: List[str]) -> Tuple[str, List[Tuple[str, str, str]], float]:
        all_triplets, all_rel_questions = get_relationship_questions(self.entity_list, self.relationship_list)
        predicted_triplets = []
        all_llm_outputs = []
        time_per_image = 0
        for triplet, rel_question in zip(all_triplets, all_rel_questions):
            llm_output, time_spent = self.vlm.inference(prompt, images, rel_questions=[get_rel_text(rel_question)])
            all_llm_outputs.append(llm_output)
            count = parse_string_to_count(llm_output)
            time_per_image += time_spent
            if count > 0:
                for _ in range(count):
                    predicted_triplets.append(triplet)
        llm_output = str(all_llm_outputs)
        return llm_output, predicted_triplets, time_per_image
