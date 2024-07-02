#########################
### Zero shot prompts ###
#########################

zero_shot_prompts = {
### List of triplets ###
"list_of_triplets": [
{"prompt":
"""
### Definitions:
1. **ego**: The vehicle holding the camera and capturing the scene (not visible in the image).
2. **OBJECT**: Refers to one of the following:
- Person
- Bicycle
- Bus
- Car
- Construction vehicle
- Emergency vehicle
- Motorcycle
- Trailer truck
- Truck
3. **RELATIONSHIP**: Refers to one of the following:
- Positional: `inFrontOf`, `toLeftOf`, `toRightOf`
- Distance-based: `very_near` (within 10 meters), `near` (within 25 meters), `visible` (within 50 meters)

### Instructions:
1. Identify all OBJECTS in the scene. Only include objects that are clearly visible, identifiable, and relevant in the image. Do not include objects that are not present.
2. For each OBJECT, generate two RELATIONSHIPS with the ego:
- One positional RELATIONSHIP (`inFrontOf`, `toLeftOf`, `toRightOf`)
- One distance-based RELATIONSHIP (`very_near`, `near`, `visible`)
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
(bus, inFrontOf, ego),
(bus, near, ego),
(car, toLeftOf, ego),
(car, very_near, ego)
]
```

### Important:
- Do not include any objects that are not visible in the scene.
- Ensure that all identified objects are clearly distinguishable and relevant.
- Verify the presence and relevance of each object before including it in the output.

---

### Image Analysis:

Using the above prompt format, please analyze the following image and generate the output according to the template, ensuring only visible, identifiable, and relevant objects are included:
"""
},
{
"prompt":
"""
From now on, whenever I type 'ego', I am referring to the vehicle holding the camera and capturing the scene, therefore it does not appear in the image.
From now on, whenever I type OBJECT, I am referring to one of the following OBJECTS: person, bicycle, bus, car, construction vehicle, emergency vehicle, motorcycle, trailer truck, truck.
From now on, whenever I ask for a RELATIONSHIP, I am asking for one of the following RELATIONSHIPS: inFrontOf, toLeftOf, toRightOf, very_near, near, visible. Where the last three RELATIONSHIPS describe distances of 10, 25, and 50 meters between an OBJECT and the ego.
I want you to generate only 2 RELATIONSHIPS betwewn the OBJECT and ego for each OBJECT in the scene. The first relationship should be one of the three RELATIONSHIPS (inFrontOf, toLeftOf, toRightOf) and the second relationship should be one of the three RELATIONSHIPS (very_near, near, visible). For example: if there is one OBJECT in the scene you should output 2 RELATIONSHIPS, if there are 3 OBJECTS you should output 6 RELATIONSHIPS, if there are n OBJECTS you should output 2n RELATIONSHIPS.
I am going to provide a template for your output. Every word in capital letters is a placeholder. Anytime that you generate text, fit it into one of the placeholders that I list. Preserve the formatting and overall template:
[OBJECT, RELATIONSHIP, ego]
"""
}
],
### List of objects ###
"list_of_objects": [
{
"prompt":
"""
From now on, whenever I type 'ego', I am referring to the vehicle holding the camera and capturing the scene, therefore it does not appear in the image.
From now on, whenever I type 'object', I am referring to one of the following objects: person, bicycle, bus, car, construction vehicle, emergency vehicle, motorcycle, trailer truck, truck.
I want you to list all the object that you see in the image. If an object appears more than once, list it as many times as it appears.
I am going to provide a template for your output. Every word in capital letters is a placeholder. Anytime that you generate text, fit it into one of the placeholders that I list. Preserve the formatting and overall template:
[OBJECT]
"""
}
]
}

########################
### Few shot prompts ###
########################

few_shot_prompts = {
"list_of_triplets": [
{
}
]
}