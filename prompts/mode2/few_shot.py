LLAVA_M2_ONE_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img1.jpg*$||\
Is there a truck within 25 meters of ego?
Answer yes/no.
If yes, how many times it is?
Answer from 1 to 10, for example: 'yes. 3' or 'yes. 1'
||$*USER-END*$||\
||$*SYSTEM*$||
Yes. 1
||$*SYSTEM-END*$||
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||?
Answer yes/no.
If yes, how many times it is?
Answer from 1 to 10, for example: 'yes. 3' or 'yes. 1'
||$*USER-END*$||\
"""

SPACELLAVA_M2_ONE_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img1.jpg*$||\
Is there a truck within 25 meters of ego?
Answer yes/no.
If yes, return a number from 1 to 10 describing how many times?
If no, return 0.
||$*USER-END*$||\
||$*SYSTEM*$||
1
||$*SYSTEM-END*$||
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||??
Answer yes/no.
If yes, return a number from 1 to 10 describing how many times?
If no, return 0.
||$*USER-END*$||\
"""
