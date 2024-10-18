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

GPT_M4_ZERO_SHOT_PROMPT = \
"""\
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
"""