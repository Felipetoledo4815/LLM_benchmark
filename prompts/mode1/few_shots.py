LLAVA_M1_ONE_SHOT_PROMPT = \
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
||$*IMAGE: ./prompts/images/img1.jpg*$||\
||$*USER-END*$||\

||$*SYSTEM*$||
[('truck', 'within_25m', 'ego'), ('truck', 'to_left_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'to_right_of', 'ego')]
||$*SYSTEM-END*$||

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


LLAVA_M1_TWO_SHOTS_PROMPT = \
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
||$*IMAGE: ./prompts/images/img1.jpg*$||\
||$*USER-END*$||\

||$*SYSTEM*$||
[('truck', 'within_25m', 'ego'), ('truck', 'to_left_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'to_right_of', 'ego')]
||$*SYSTEM-END*$||

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
||$*IMAGE: ./prompts/images/img2.jpg*$||\
||$*USER-END*$||\

||$*SYSTEM*$||
[('car', 'within_25m', 'ego'), ('car', 'to_left_of', 'ego'), ('person', 'within_25m', 'ego'), ('person', 'to_left_of', 'ego'), ('car', 'between_40m_and_60m', 'ego'), ('car', 'in_front_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'to_right_of', 'ego'), ('motorcycle', 'between_40m_and_60m', 'ego'), ('motorcycle', 'to_right_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'in_front_of', 'ego')]
||$*SYSTEM-END*$||

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

FLAMINGO_M1_TWO_SHOT_PROMPT = \
"""\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img1.jpg*$||\
An image of [('truck', 'within_25m', 'ego'), ('truck', 'to_left_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'to_right_of', 'ego')]\
||$*USER-END*$||\
||$*USER*$||\
||$*IMAGE: ./prompts/images/img2.jpg*$||\
An image of [('car', 'within_25m', 'ego'), ('car', 'to_left_of', 'ego'), ('person', 'within_25m', 'ego'), ('person', 'to_left_of', 'ego'), ('car', 'between_40m_and_60m', 'ego'), ('car', 'in_front_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'to_right_of', 'ego'), ('motorcycle', 'between_40m_and_60m', 'ego'), ('motorcycle', 'to_right_of', 'ego'), ('car', 'within_25m', 'ego'), ('car', 'in_front_of', 'ego')]\
||$*USER-END*$||\
||$*USER*$||\
||$*IMAGE*$||\
An image of \
||$*USER-END*$||\
"""
