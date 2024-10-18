from typing import List, Tuple
import re
import tempfile
import os
import cv2
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


def parse_string_to_bool(string: str) -> int:
    # Check if the string contains "yes"
    if "yes" in string.lower():
        return True
    else:
        return False


def get_entities_relationship_questions(entities, relationships) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    questions = []
    triplets = []
    for entity in entities:
        if entity == "ego":
            continue
        for relationship in relationships:
            questions.append(f"{entity} {relationship} of ego")
            triplets.append((entity, relationship, "ego"))
    return triplets, questions


def get_relationship_questions(relationships) -> Tuple[List[str], List[str]]:
    questions = []
    relations = []
    for relationship in relationships:
        questions.append(f"{relationship} of ego")
        relations.append(relationship)
    return questions, relations


def get_rel_text(rel: str):
    # Replace underscores with spaces
    text_rel = rel.replace("_", " ")

    # Replace 'm' following a number with 'meters'
    text_rel = re.sub(r'(\d+)m', r'\1 meters', text_rel)

    return text_rel


def print_bb_on_image(image_path: str, bb: str) -> str:
    # Load the image
    image = cv2.imread(image_path)  # pylint: disable=no-member

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Unpack the bounding box coordinates
    numbers = bb.strip('()').split(',')
    numbers = [int(num.strip()) for num in numbers]
    bottom_left = (numbers[0], numbers[1])
    top_right = (numbers[2], numbers[3])

    # Draw the bounding box on the image
    cv2.rectangle(image, bottom_left, top_right, (0, 0, 255), 8)  # Red color in BGR; pylint: disable=no-member

    # Create a temporary file to save the modified image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, image)  # pylint: disable=no-member

    # Return the path of the temporary file
    return temp_file_path


class QueryMode:
    # TODO: Add tests for this class
    def __init__(self, mode: str, vlm: VLMInterface, entity_list: List[str],
                 relationship_list: List[str]) -> None:
        self.mode = getattr(self, f"mode{mode}")
        self.vlm = vlm
        self.entity_list = entity_list
        self.relationship_list = relationship_list

    def query(self,
              prompt: str,
              images: List[str],
              bboxes: List[Tuple[str, List[Tuple[str, str, str]]]] | None = None
              ) -> Tuple[str, List[Tuple[str, str, str]], float]:
        """
        Query the model with the given prompt and images based on the mode.
        :param prompt: Prompt to query the model.
        :param images: List of image paths.
        :param bboxes: List of bounding boxes.
        :return:
            llm_output: Output from the model.
            predicted_triplets: List of predicted triplets.
            time_spent: Time spent for the inference.
        """
        return self.mode(prompt, images, bboxes)

    def mode1(self,
              prompt: str,
              images: List[str],
              bboxes: None = None) -> Tuple[str, List[Tuple[str, str, str]], float]:
        llm_output, time_spent = self.vlm.inference(prompt, images)
        predicted_triplets = parse_string_to_sg(llm_output)
        return llm_output, predicted_triplets, time_spent

    def mode2(self,
              prompt: str,
              images: List[str],
              bboxes: None = None) -> Tuple[str, List[Tuple[str, str, str]], float]:
        all_triplets, all_rel_questions = get_entities_relationship_questions(self.entity_list, self.relationship_list)
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

    def mode3(self, prompt: str, images: List[str],
              bboxes: List[Tuple[str, List[Tuple[str, str, str]]]]) -> Tuple[str, List[Tuple[str, str, str]], float]:
        predicted_triplets = []
        all_llm_outputs = []
        time_per_image = 0
        for bb, _ in bboxes:
            temp_img_path = print_bb_on_image(images[0], bb)
            llm_output, time_spent = self.vlm.inference(prompt, [temp_img_path])
            all_llm_outputs.append(llm_output)
            parsed_prediction = parse_string_to_sg(llm_output)
            predicted_triplets.extend(parsed_prediction)
            time_per_image += time_spent
            os.remove(temp_img_path)
        llm_output = str(all_llm_outputs)
        return llm_output, predicted_triplets, time_per_image

    def mode4(self,
              prompt: str,
              images: List[str],
              bboxes: List[Tuple[str, List[Tuple[str, str, str]]]]) -> Tuple[str, List[Tuple[str, str, str]], float]:
        all_rel_questions, all_relationships = get_relationship_questions(self.relationship_list)
        predicted_triplets = []
        all_llm_outputs = []
        time_per_image = 0
        for bb, triplets in bboxes:
            temp_img_path = print_bb_on_image(images[0], bb)
            for rel_question, relationship in zip(all_rel_questions, all_relationships):
                llm_output, time_spent = self.vlm.inference(prompt, [temp_img_path],
                                                            rel_questions=[get_rel_text(rel_question)])
                all_llm_outputs.append(llm_output)
                answer = parse_string_to_bool(llm_output)
                if answer:
                    entity = triplets[0][0]
                    predicted_triplets.append((entity, relationship, "ego"))
                time_per_image += time_spent
            os.remove(temp_img_path)
        llm_output = str(all_llm_outputs)
        return llm_output, predicted_triplets, time_per_image
