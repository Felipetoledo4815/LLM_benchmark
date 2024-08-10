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

#TODO: Implement the following prompt
FLAMINGO_M2_TWO_SHOTS_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img1.jpg*$||\
Is there a truck within 25 meters of ego? Answer yes/no. If yes, return a number from 1 to 10 describing how many times? If no, return 0.
Yes. 1
||$*USER-END*$||\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img2.jpg*$||\
Is there a car in front of ego? Answer yes/no. If yes, return a number from 1 to 10 describing how many times? If no, return 0.
Yes. 2
||$*USER-END*$||\
||$*USER*$||\
||$*IMAGE*$||\
Is there a ||$*REL_QUESTION*$||? Answer yes/no. If yes, return a number from 1 to 10 describing how many times? If no, return 0.
||$*USER-END*$||\
"""
