from prompts.mode1.zero_shot import LLAVA_M1_ZERO_SHOT_PROMPT, LLAVA_FT_M1_ZERO_SHOT_PROMPT, \
    PALIGEMMA_M1_ZERO_SHOT_PROMPT
from prompts.mode1.few_shots import LLAVA_M1_ONE_SHOT_PROMPT, LLAVA_M1_TWO_SHOTS_PROMPT, FLAMINGO_M1_TWO_SHOT_PROMPT
from prompts.mode2.zero_shot import LLAVA_M2_ZERO_SHOT_PROMPT, SPACELLAVA_M2_ZERO_SHOT_PROMPT, \
    MOBILEVLM_M2_ZERO_SHOT_PROMPT, PALIGEMMA_M2_ZERO_SHOT_PROMPT
from prompts.mode2.few_shot import LLAVA_M2_ONE_SHOT_PROMPT, SPACELLAVA_M2_ONE_SHOT_PROMPT, FLAMINGO_M2_TWO_SHOTS_PROMPT
from prompts.mode3.zero_shot import LLAVA_M3_ZERO_SHOT_PROMPT
from prompts.mode4.zero_shot import LLAVA_M4_ZERO_SHOT_PROMPT

EVALUATION_PROMPTS = {
    "llava_1.5": {
        # Why not few-shots? https://github.com/haotian-liu/LLaVA/issues/344
        "mode1": {
            "zero": LLAVA_M1_ZERO_SHOT_PROMPT
        },
        "mode2": {
            "zero": LLAVA_M2_ZERO_SHOT_PROMPT
        },
        "mode3": {
            "zero": LLAVA_M3_ZERO_SHOT_PROMPT
        },
        "mode4": {
            "zero": LLAVA_M4_ZERO_SHOT_PROMPT
        }
    },
    "llava_1.5_ft": {
        "mode1": {
            "zero": LLAVA_FT_M1_ZERO_SHOT_PROMPT
        },
        "mode2": {
            "zero": LLAVA_M2_ZERO_SHOT_PROMPT
        },
        "mode3": {
            "zero": LLAVA_M3_ZERO_SHOT_PROMPT
        },
        "mode4": {
            "zero": LLAVA_M4_ZERO_SHOT_PROMPT
        }
    },
    "llava_1.6_mistral": {
        "mode1": {
            "zero": LLAVA_M1_ZERO_SHOT_PROMPT,
            "one": LLAVA_M1_ONE_SHOT_PROMPT,
            "two": LLAVA_M1_TWO_SHOTS_PROMPT
        },
        "mode2": {
            "zero": LLAVA_M2_ZERO_SHOT_PROMPT,
            "one": LLAVA_M2_ONE_SHOT_PROMPT
        },
        "mode3": {
            "zero": LLAVA_M3_ZERO_SHOT_PROMPT
        },
        "mode4": {
            "zero": LLAVA_M4_ZERO_SHOT_PROMPT
        }
    },
    "llava_1.6_mistral_ft": {
        "mode1": {
            "zero": LLAVA_FT_M1_ZERO_SHOT_PROMPT
        },
        "mode2": {
            "zero": LLAVA_M2_ZERO_SHOT_PROMPT
        },
        "mode3": {
            "zero": LLAVA_M3_ZERO_SHOT_PROMPT
        },
        "mode4": {
            "zero": LLAVA_M4_ZERO_SHOT_PROMPT
        }
    },
    "spacellava": {
        "mode1": {
            "zero": LLAVA_M1_ZERO_SHOT_PROMPT,
            "one": LLAVA_M1_ONE_SHOT_PROMPT,
            "two": LLAVA_M1_TWO_SHOTS_PROMPT
        },
        "mode2": {
            "zero": SPACELLAVA_M2_ZERO_SHOT_PROMPT,
            "one": SPACELLAVA_M2_ONE_SHOT_PROMPT
        },
        "mode3": {
            "zero": LLAVA_M3_ZERO_SHOT_PROMPT
        },
        "mode4": {
            "zero": LLAVA_M4_ZERO_SHOT_PROMPT
        }
    },
    "mobilevlm": {
        "mode2": {
            "zero": MOBILEVLM_M2_ZERO_SHOT_PROMPT
        }
    },
    "mobilevlm2": {
        "mode1": {
            "zero": LLAVA_M1_ZERO_SHOT_PROMPT
        },
        "mode2": {
            "zero": MOBILEVLM_M2_ZERO_SHOT_PROMPT
        }
    },
    "openflamingo": {
        "mode1": {
            "two": FLAMINGO_M1_TWO_SHOT_PROMPT
        },
        "mode2": {
            "two": FLAMINGO_M2_TWO_SHOTS_PROMPT
        }
    },
    "paligemma": {
        "mode1": {
            "zero": PALIGEMMA_M1_ZERO_SHOT_PROMPT
        },
        "mode2": {
            "zero": PALIGEMMA_M2_ZERO_SHOT_PROMPT
        }
    }
}
