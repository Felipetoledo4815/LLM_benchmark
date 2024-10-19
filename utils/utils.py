from pathlib import Path
from prompts.all_prompts import EVALUATION_PROMPTS
from vlm.vlm_interface import VLMInterface
from vlm.space_llava_wrapper import SpaceLlavaWrapper
# from vlm.mobile_vlm_wrapper import MobileVLM
# from vlm.mobile_vlm2_wrapper import MobileVLM2
from vlm.hf_llava_next_wrapper import HFLlavaNextWrapper
from vlm.hf_llava_wrapper import HFLlavaWrapper
from vlm.hf_pali_gemma_wrapper import HFPaliGemma
from vlm.open_flamingo_wrapper import OpenFlamingo
from vlm.roadscene2vec_wrapper import RoadScene2Vec

def get_prompt(mode: str, model: str, shot: str):
    if model == 'roadscene2vec':
        #TODO: This should not be a VLMInterface, it should be a SGG that does not need prompt
        return "No prompt needed for RoadScene2Vec."
    try:
        prompt = EVALUATION_PROMPTS[model][f"mode{mode}"][shot]
    except KeyError as exc:
        raise NotImplementedError(
            f"Mode {mode}, {shot}-shot for model {model} has not been implemented.") from exc
    return prompt


def get_model(model_name: str, lora=Path) -> VLMInterface:
    if model_name == 'llava_1.5':
        vlm = HFLlavaWrapper("llava-hf/llava-1.5-7b-hf", cache_dir="./models/llava-1.5-7b-hf")
    elif model_name == 'llava_1.5_ft':
        vlm = HFLlavaWrapper("llava-hf/llava-1.5-7b-hf", cache_dir="./models/llava-1.5-7b-hf", lora=lora)
    elif model_name == 'llava_1.6_mistral':
        vlm = HFLlavaNextWrapper("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir="./models/llava-v1.6-mistral-7b-hf")
    elif model_name == 'llava_1.6_mistral_ft':
        vlm = HFLlavaNextWrapper("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir="./models/llava-v1.6-mistral-7b-hf", lora=lora)
    elif model_name == 'llava_1.6_vicuna':
        vlm = HFLlavaNextWrapper("llava-hf/llava-v1.6-vicuna-13b-hf", cache_dir="./models/llava-v1.6-vicuna-13b-hf")
    elif model_name == 'spacellava':
        vlm = SpaceLlavaWrapper(clip_path="./models/spacellava/mmproj-model-f16.gguf",
                                model_path="./models/spacellava/ggml-model-q4_0.gguf")
    # elif model_name == 'mobilevlm':
    #     vlm = MobileVLM(clip_path="./models/mobile-vlm/mmproj-model-f16.gguf",
    #                     model_path="./models/mobile-vlm/ggml-model-q4_k.gguf")
    # elif model_name == 'mobilevlm2':
    #     vlm = MobileVLM2(model_path='mtgv/MobileVLM-3B')
    elif model_name == 'paligemma':
        vlm = HFPaliGemma("google/paligemma-3b-mix-448", cache_dir="./models/paligemma-3b-mix-448")
    elif model_name == 'openflamingo':
        vlm = OpenFlamingo("openflamingo/OpenFlamingo-3B-vitl-mpt1b", cache_dir="./models/OpenFlamingo-3B-vitl-mpt1b")
    elif model_name == 'roadscene2vec':
        vlm = RoadScene2Vec("./models/roadscene2vec/config.yaml")
    else:
        raise ValueError(f"Unknown model {model_name}.")
    return vlm
