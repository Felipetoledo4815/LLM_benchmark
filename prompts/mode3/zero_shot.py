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

GPT_M3_ZERO_SHOT_PROMPT = \
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
Using the above prompt format, please analyze the following image and generate the output according to the template.
"""

