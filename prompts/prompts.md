The prompts need to use the following identifiers:
- ||$*USER*$||: Placeholder for user role. Message send by user.
- ||$*USER-END*$||: Placeholder for end of user role.
- ||$*SYSTEM*$||: Placeholder for user role. Message send by system.
- ||$*SYSTEM-END*$||: Placeholder for end of system role.
- ||$*IMAGE*$||: Placeholder for image. This will be parsed in runtime with the desired image.
- ||$*IMAGE: path_to_image*$||: Placeholder for image. This will be parsed with the prompt, and the image_path will be used to load the image. This can be used when doing few-shots
- ||$*BOUNDING-BOX*$||: Placeholder for bounding box. This will be parsed in runtime with the desired bounding box.
- ||$*REL_QUESTION*$||: Placeholder for relationship. This will be parsed in runtime with the desired relationship.

Also note that since the prompts are being defined using **docstrings**, even though there are no newline characters '\n' in the prompts definition, they will be rendered with newline characters when compiled. If you do not want a newline character to appear in the prompt you will have to use '\'.