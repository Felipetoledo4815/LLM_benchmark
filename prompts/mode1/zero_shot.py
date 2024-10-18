LLAVA_M1_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||
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
1. Identify all OBJECTS in the scene. Only include objects that are clearly visible, identifiable, and relevant in the image. Do not include objects that are not present.
2. For each OBJECT, generate two RELATIONSHIPS with the ego:
- One positional RELATIONSHIP (`in_front_of`, `to_left_of`, `to_right_of`)
- One distance-based RELATIONSHIP (`within_25m`, `between_25m_and_40m`, `between_40m_and_60m`)
3. Use the following structured format for your output.
4. Verify that each OBJECT is clearly visible and identifiable in the image before including it in the output.

### Output Template:
```
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
...
]
```

### Example:
Given an image with one bus and one car, the output should be:
```
[
(bus, in_front_of, ego),
(bus, between_25m_and_40m, ego),
(car, to_left_of, ego),
(car, within_25m, ego)
]
```

### Important:
- Do not include any objects that are not visible in the scene.
- Ensure that all identified objects are clearly distinguishable and relevant.
- Verify the presence and relevance of each object before including it in the output.

---

### Image Analysis:
Using the above prompt format, please analyze the following image and generate the output according to the template, ensuring only visible, identifiable, and relevant objects are included:
||$*IMAGE*$||\
||$*USER-END*$||\
"""

LLAVA_FT_M1_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||
Give me the triplets.\
||$*USER-END*$||\
"""

#TODO: Implement the following prompt
PALIGEMMA_M1_ZERO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE*$||\
What do you see?\
||$*USER-END*$||\
"""

GPT_M1_ZERO_SHOT_PROMPT = \
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
1. Identify all OBJECTS in the scene. Only include objects that are clearly visible, identifiable, and relevant in the image. Do not include objects that are not present.
2. For each OBJECT, generate two RELATIONSHIPS with the ego:
- One positional RELATIONSHIP (`in_front_of`, `to_left_of`, `to_right_of`)
- One distance-based RELATIONSHIP (`within_25m`, `between_25m_and_40m`, `between_40m_and_60m`)
3. Use the following structured format for your output.
4. Verify that each OBJECT is clearly visible and identifiable in the image before including it in the output.

### Output Template:
```
[
(OBJECT, RELATIONSHIP, ego),
(OBJECT, RELATIONSHIP, ego),
...
]
```

### Example:
Given an image with one bus and one car, the output should be:
```
[
(bus, in_front_of, ego),
(bus, between_25m_and_40m, ego),
(car, to_left_of, ego),
(car, within_25m, ego)
]
```

### Important:
- Do not include any objects that are not visible in the scene.
- Ensure that all identified objects are clearly distinguishable and relevant.
- Verify the presence and relevance of each object before including it in the output.

---

### Image Analysis:
Using the above prompt format, please analyze the following image and generate the output according to the template, ensuring only visible, identifiable, and relevant objects are included.
"""