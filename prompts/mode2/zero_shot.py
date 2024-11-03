LLAVA_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||?
Answer yes/no.
If yes, how many times it is?
Answer from 1 to 10, for example: 'yes. 3' or 'yes. 1'
||$*USER-END*$||\
"""

SPACELLAVA_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||??
Answer yes/no.
If yes, return a number from 1 to 10 describing how many times?
If no, return 0.
||$*USER-END*$||\
"""

MOBILEVLM_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||?\
Answer yes/no.
||$*USER-END*$||\
"""

PALIGEMMA_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||?
Answer yes/no.
If yes, how many times it is?
Answer from 1 to 10, for example: 'yes. 3' or 'yes. 1'
||$*USER-END*$||\
"""

LLAVA_FT_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||?
||$*USER-END*$||\
"""

CAMBRIAN_PHI3_M2_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||? (yes/no). How many times? [1-10]
Do not provide any explanation and structure your answer in the following way: yes/no. [1-10]. For example: "Yes. 2."
||$*USER-END*$||\
"""
