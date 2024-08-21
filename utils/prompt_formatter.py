from typing import List, Tuple, Callable


def check_prompt_consistency(prompt: str, images: List[str], rel_questions: List[str]) -> None:
    # Check start of prompt
    prompt_starts_with_user = prompt.strip().startswith("||$*USER*$||")
    assert prompt_starts_with_user, "Prompt should start with user placeholder. We have not implemented system role definition."

    # Check that start and end user placeholders are equal
    count_user_start = prompt.count("||$*USER*$||")
    count_user_end = prompt.count("||$*USER-END*$||")
    assert count_user_start == count_user_end, "Number of user placeholders should be equal to the number of user end placeholders."

    # Check that number of image placeholders is equal to the number of images
    count_image_placeholders = prompt.count("||$*IMAGE*$||")
    assert count_image_placeholders == len(images), "Number of images does not match the number of placeholders."

    # Check that number of rel_questions placeholders is equal to the number of questions
    count_rel_question_placeholders = prompt.count("||$*REL_QUESTION*$||")
    assert count_rel_question_placeholders == len(rel_questions), "Number of relationship questions does not match the number of placeholders."

    # Check that each user message contains one image placeholder
    for user_message_text in prompt.split("||$*USER*$||")[1:]:
        user_message_text = user_message_text.split("||$*USER-END*$||")[0]

        count_img = user_message_text.count("||$*IMAGE*$||") + user_message_text.count("||$*IMAGE:")
        assert count_img == 1, "Each user message should contain one image placeholder."


def check_prompt_consistency_few_shots(prompt: str) -> None:
    # Check that start and end system placeholders are equal
    count_system_start = prompt.count("||$*SYSTEM*$||")
    count_system_end = prompt.count("||$*SYSTEM-END*$||")
    assert count_system_start == count_system_end, "Number of system placeholders should be equal to the number of system end placeholders."

    # Check that number of user placeholders is greater than number of system placeholders
    count_user_placeholders = prompt.count("||$*USER*$||")
    count_system_placeholders = prompt.count("||$*SYSTEM*$||")
    assert count_user_placeholders > count_system_placeholders, "Number of user placeholders should be greater than system placeholders. No need to add system placeholder at the end."


def llama_cpp_formatter(prompt: str, images: List[str], img_parser: Callable, rel_questions: List[str]) -> List:
    check_prompt_consistency(prompt, images, rel_questions)
    check_prompt_consistency_few_shots(prompt)
    message = [
        {"role": "system", "content": "You are an assistant that describe images."}
    ]
    for i, message_text in enumerate(prompt.split("||$*USER*$||")[1:]):
        user_message_text = message_text.split("||$*USER-END*$||")[0]
        ### Relationship Questions
        if user_message_text.count("||$*REL_QUESTION*$||") == 1:
            user_message_text = user_message_text.replace("||$*REL_QUESTION*$||", rel_questions.pop(0))
        ### Images
        if user_message_text.count("||$*IMAGE:") == 1:
            img_path = user_message_text.split("||$*IMAGE:")[1].split("*$||")[0]
            # In case of few-shot learning, we use the path inside the prompt
            image = img_parser(img_path.strip())
            first_part = user_message_text.split("||$*IMAGE:")[0]
            second_part = user_message_text.split("||$*IMAGE:")[1].split("*$||")[1]
        else:
            # Image to be passed to the model
            image = images.pop(0)
            first_part = user_message_text.split("||$*IMAGE*$||")[0]
            second_part = user_message_text.split("||$*IMAGE*$||")[1]

        user_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image}}
            ]
        }
        if first_part:
            user_message["content"].insert(0, {"type": "text", "text": first_part})
        if second_part:
            user_message["content"].append({"type": "text", "text": second_part})

        message.append(user_message)

        count_system_message = message_text.count("||$*SYSTEM*$||")
        if count_system_message > 0 and i <= count_system_message:
            system_message_text = prompt.split("||$*SYSTEM*$||")[i + 1].split("||$*SYSTEM-END*$||")[0]
            system_message = {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message_text}
                ]
            }
            message.append(system_message)

    return message


def hf_llava_formatter(prompt: str, images: List[str], model_v: str, rel_questions: List[str]) -> Tuple[str, List[str]]:
    check_prompt_consistency(prompt, images, rel_questions)
    check_prompt_consistency_few_shots(prompt)
    message = ""
    if model_v.startswith("llava-v1.6-vicuna-"):
        message = "A chat between a curious human and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the human's questions."
    images_to_load = []

    for i, message_text in enumerate(prompt.split("||$*USER*$||")[1:]):
        user_message_text = message_text.split("||$*USER-END*$||")[0]
        ### Relationship Questions
        if user_message_text.count("||$*REL_QUESTION*$||") == 1:
            user_message_text = user_message_text.replace("||$*REL_QUESTION*$||", rel_questions.pop(0))
        ### Images
        if user_message_text.count("||$*IMAGE:") == 1:
            img_path = user_message_text.split("||$*IMAGE:")[1].split("*$||")[0]
            images_to_load.append(img_path.strip())
            first_part = user_message_text.split("||$*IMAGE:")[0]
            second_part = user_message_text.split("||$*IMAGE:")[1].split("*$||")[1]
        else:
            first_part = user_message_text.split("||$*IMAGE*$||")[0]
            second_part = user_message_text.split("||$*IMAGE*$||")[1]

        if model_v == "llava-1.5-7b-hf":
            message += f"USER: {first_part} <image> {second_part}\n"
        elif model_v == "llava-v1.6-mistral-7b-hf":
            message += f"[INST] {first_part} <image> {second_part}[/INST]"
        elif model_v.startswith("llava-v1.6-vicuna-"):
            message += f"USER: {first_part} <image> {second_part}\n"

        count_system_message = message_text.count("||$*SYSTEM*$||")
        if count_system_message > 0 and i <= count_system_message:
            system_message_text = prompt.split("||$*SYSTEM*$||")[i + 1].split("||$*SYSTEM-END*$||")[0]

            if model_v == "llava-1.5-7b-hf":
                message += f"ASSISTANT: {system_message_text}"
            elif model_v == "llava-v1.6-mistral-7b-hf":
                message += f"{system_message_text}"
            elif model_v.startswith("llava-v1.6-vicuna-"):
                message += f"ASSISTANT: {system_message_text}"

    if model_v == "llava-1.5-7b-hf":
        message += "ASSISTANT:"
    elif model_v.startswith("llava-v1.6-vicuna-"):
        message += "ASSISTANT:"

    return message, images_to_load


def flamingo_formatter(prompt: str, images: List[str], rel_questions: List[str]) -> Tuple[str, List[str]]:
    check_prompt_consistency(prompt, images, rel_questions)
    count_system_message = prompt.count("||$*SYSTEM*$||")
    assert count_system_message == 0, "Flamingo does not support system messages for few shot learning. \
        However, it supports in-context learning."
    message = ""

    images_to_load = []
    for _, user_message_text in enumerate(prompt.split("||$*USER*$||")[1:]):
        user_message_text = user_message_text.split("||$*USER-END*$||")[0]
        ### Relationship Questions
        if user_message_text.count("||$*REL_QUESTION*$||") == 1:
            user_message_text = user_message_text.replace("||$*REL_QUESTION*$||", rel_questions.pop(0))
        # Images
        if user_message_text.count("||$*IMAGE:") == 1:
            img_path = user_message_text.split("||$*IMAGE:")[1].split("*$||")[0]
            images_to_load.append(img_path.strip())
            first_part = user_message_text.split("||$*IMAGE:")[0]
            second_part = user_message_text.split("||$*IMAGE:")[1].split("*$||")[1]
            message += f"{first_part} <image> {second_part}<|endofchunk|>"
        else:
            first_part = user_message_text.split("||$*IMAGE*$||")[0]
            second_part = user_message_text.split("||$*IMAGE*$||")[1]
            message += f"{first_part} <image> {second_part}"

    return message, images_to_load
    # raise NotImplementedError("Flamingo formatter is not implemented.")


def pali_gemma_formatter(prompt: str, images: List[str], rel_questions: List[str]) -> str:
    check_prompt_consistency(prompt, images, rel_questions)
    # Pali Gemma checks
    count_system_message = prompt.count("||$*SYSTEM*$||")
    assert count_system_message == 0, "PaliGemma does not support system messages for few shot learning."
    count_images = prompt.count("||$*IMAGE:")
    assert count_images == 0, "PaliGemma does not support multiple images for few shot learning."
    count_user_message = prompt.count("||$*USER*$||")
    assert count_user_message == 1, "PaliGemma expects only one user message."

    message = ""
    # Retrieve user message
    user_message_text = prompt.split("||$*USER*$||")[1]
    user_message_text = user_message_text.split("||$*USER-END*$||")[0]
    # Relationship Questions
    if user_message_text.count("||$*REL_QUESTION*$||") == 1:
        user_message_text = user_message_text.replace("||$*REL_QUESTION*$||", rel_questions.pop(0))
    # Remove image placeholder
    first_part = user_message_text.split("||$*IMAGE*$||")[0]
    second_part = user_message_text.split("||$*IMAGE*$||")[1]
    message += f"{first_part}{second_part}"

    return message


def mobile_vlm_formatter(prompt: str, images: List[str], rel_questions: List[str]) -> Tuple[str, List[str]]:
    check_prompt_consistency(prompt, images, rel_questions)
    check_prompt_consistency_few_shots(prompt)
    message = ""
    images_to_load = []

    for i, message_text in enumerate(prompt.split("||$*USER*$||")[1:]):
        user_message_text = message_text.split("||$*USER-END*$||")[0]
        ### Relationship Questions
        if user_message_text.count("||$*REL_QUESTION*$||") == 1:
            user_message_text = user_message_text.replace("||$*REL_QUESTION*$||", rel_questions.pop(0))
        ### Images
        if user_message_text.count("||$*IMAGE:") == 1:
            img_path = user_message_text.split("||$*IMAGE:")[1].split("*$||")[0]
            images_to_load.append(img_path.strip())
            first_part = user_message_text.split("||$*IMAGE:")[0]
            second_part = user_message_text.split("||$*IMAGE:")[1].split("*$||")[1]
        else:
            first_part = user_message_text.split("||$*IMAGE*$||")[0]
            second_part = user_message_text.split("||$*IMAGE*$||")[1]

        message += f"USER: {first_part} <image> {second_part}\n"

        count_system_message = message_text.count("||$*SYSTEM*$||")
        if count_system_message > 0 and i <= count_system_message:
            system_message_text = prompt.split("||$*SYSTEM*$||")[i + 1].split("||$*SYSTEM-END*$||")[0]

            message += f"ASSISTANT: {system_message_text}"

    message += "ASSISTANT:"

    return message, images_to_load
