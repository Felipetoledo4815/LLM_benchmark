LLAVA_M3_ZERO_SHOT_PROMPT = \
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
1. Given the red bounding box, detect which OBJECT is in it.
2. Generate two RELATIONSHIPS between the detected OBJECT in the red bounding box and ego:
- One positional RELATIONSHIP (`in_front_of`, `to_left_of`, `to_right_of`)
- One distance-based RELATIONSHIP (`within_25m`, `between_25m_and_40m`, `between_40m_and_60m`)
3. Use the following structured format for your output.
### Output Template:
```
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
]
```
Using the above prompt format, please analyze the following image and generate the output according to the template:
||$*IMAGE*$||\
||$*USER-END*$||\
"""

PALIGEMMA_M3_ZERO_SHOT_PROMPT = \
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
1. Given the red bounding box, detect which OBJECT is in it.
2. Generate two RELATIONSHIPS between the detected OBJECT in the red bounding box and **ego** only:
- One positional RELATIONSHIP (`in_front_of`, `to_left_of`, `to_right_of`)
- One distance-based RELATIONSHIP (`within_25m`, `between_25m_and_40m`, `between_40m_and_60m`)
3. Use the following structured format for your output.

### Output Template:
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
]

### Example:
Given an image with a red bounding box with a bus in it, the output should be:
[
(bus, in_front_of, ego),
(bus, between_25m_and_40m, ego),
]
Note that I am only interested in RELATIONSHIPS with ego, therefore ego always goes last in the tuple.

### Image Analysis:
Using the above prompt format, please analyze the following image and generate the output according to the template:
||$*USER-END*$||\
"""

LLAVA_FT_M3_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
Give me the triplets of the red bounding box.
||$*USER-END*$||\
"""

CAMBRIAN_M3_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
Please do not explain anything and only return a list of triplets!

DEFINITIONS:
1. **ego**: The vehicle holding the camera and capturing the scene (not visible in the image).
2. **OBJECT**: Refers to one of the following: "Person", "Bicycle", "Bus", "Car", "Construction_vehicle", "Emergency_vehicle", "Motorcycle", "Trailer_truck", "Truck"
3. **RELATIONSHIP**: Refers to one of the following: `in_front_of`, `to_left_of`, `to_right_of`, `within_25m` (within 25 meters), `between_25m_and_40m` (between 25 and 40 meters), `between_40m_and_60m` (between 40 and 60 meters)

Given the red bounding box, generate two RELATIONSHIPS between the detected OBJECT and ego:
- One positional RELATIONSHIP (`in_front_of`, `to_left_of`, `to_right_of`)
- One distance-based RELATIONSHIP (`within_25m`, `between_25m_and_40m`, `between_40m_and_60m`)
Use the following structured format for your output.
### Output Template:
```
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
]
```
Remember to only return a list of triplets, and do not explain anything!
||$*IMAGE*$||\
||$*USER-END*$||\
"""

LLAVA_1_6_VICUNA_M3_ZERO_SHOT_PROMTP = \
"""\
||$*USER*$||\
DEFINITIONS:
1. **ego**: The vehicle holding the camera and capturing the scene (not visible in the image).
2. **OBJECT**: Only refers to one of the following: "Person", "Bicycle", "Bus", "Car", "Construction_vehicle", "Emergency_vehicle", "Motorcycle", "Trailer_truck", "Truck"
3. **RELATIONSHIP**: Only refers to one of the following: `in_front_of`, `to_left_of`, `to_right_of`, `within_25m` (within 25 meters), `between_25m_and_40m` (between 25 and 40 meters), `between_40m_and_60m` (between 40 and 60 meters)

Given the red bounding box, generate two RELATIONSHIPS between the detected OBJECT and ego:
- Positional: `in_front_of`, `to_left_of`, `to_right_of`
- Distance-based: `within_25m` (within 25 meters), `between_25m_and_40m` (between 25 and 40 meters), `between_40m_and_60m` (between 40 and 60 meters)

OUTPUT TEMPLATE
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
]

EXAMPLE
Given an image with a red bounding box with a bus in it, the output should be:
[
(bus, in_front_of, ego),
(bus, between_25m_and_40m, ego),
]

Please only return a list of triplets, and do not explain anything!
||$*IMAGE*$||\
||$*USER-END*$||\
"""
