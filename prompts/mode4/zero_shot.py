LLAVA_M4_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
### Definitions:
1. **ego**: The vehicle holding the camera and capturing the scene (not visible in the image).
2. **OBJECT**: Refers to one of the following:
- Person
- Bicycle
- Bus
- Car
- Construction_vehicle
- Emergency_vehicle
- Motorcycle
- Trailer_truck
- Truck
3. **RELATIONSHIP**: Refers to one of the following:
- Positional: `in_front_of`, `to_left_of`, `to_right_of`
- Distance-based: `within_25m` (within 25 meters), `between_25m_and_40m` (between 25 and 40 meters), `between_40m_and_60m` (between 40 and 60 meters)

### Instructions:
Is the OBJECT in the red bounding box ||$*REL_QUESTION*$||???
Answer yes/no.
||$*IMAGE*$||\
||$*USER-END*$||\
"""

PALIGEMMA_M4_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
### Definitions:
1. **ego**: The vehicle holding the camera and capturing the scene (not visible in the image).
2. **OBJECT**: Refers to one of the following:
- Person
- Bicycle
- Bus
- Car
- Construction_vehicle
- Emergency_vehicle
- Motorcycle
- Trailer_truck
- Truck
3. **RELATIONSHIP**: Refers to one of the following:
- Positional: `in_front_of`, `to_left_of`, `to_right_of`
- Distance-based: `within_25m` (within 25 meters), `between_25m_and_40m` (between 25 and 40 meters), `between_40m_and_60m` (between 40 and 60 meters)

### Instructions:
Is the OBJECT in the red bounding box ||$*REL_QUESTION*$||???
Please limit your answer and ONLY output yes/no.
||$*USER-END*$||\
"""

LLAVA_FT_M4_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is the object in the red bounding box ||$*REL_QUESTION*$||?
||$*USER-END*$||\
"""

CAMBRIAN_M4_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Please only reply yes or no.
Is the OBJECT in the bounding box ||$*REL_QUESTION*$||???
Don't explain why, just return yes or no answers.
||$*USER-END*$||\
"""

LLAVA_1_6_VICUNA_M4_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Is the OBJECT in the red bounding box ||$*REL_QUESTION*$||?
Please only answer 'yes' / 'no'
||$*USER-END*$||\
"""