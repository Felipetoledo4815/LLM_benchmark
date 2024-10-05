# Download Models

## SpaceLlava
Create a folder `./models/spacellava` and download the following files inside of it:
- [ggml-model-q4_0.gguf](https://huggingface.co/remyxai/SpaceLLaVA/blob/main/ggml-model-q4_0.gguf)
- [mmproj-model-f16.gguf](https://huggingface.co/remyxai/SpaceLLaVA/blob/main/mmproj-model-f16.gguf)
Do not change the names of the files.

## Llava 1.5 7b (Hugging Face)
This model should be downloaded automatically when executing the benchmark.

## Llava 1.6 Mistral 7b (Hugging Face)
This model should be downloaded automatically when executing the benchmark.

## RoadScene2Vec
This is a scene graph generator (SGG) from https://github.com/AICPS/roadscene2vec.
Create a folder `./external_code`, move to that folder, and then clone the repository inside of it:
```bash
mkdir -p external_code
cd external_code
git clone https://github.com/AICPS/roadscene2vec.git
```
After that you need to copy the `./models/roadscene2vec/setup.py` file to the `./external_code/roadscene2vec` folder and run the following command to install the roadscene2vec package:
```bash
cd roadscene2vec
pip install -e .
```