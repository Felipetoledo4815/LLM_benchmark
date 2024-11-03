# LLM_benchmark

## Getting Started
1. Create a conda environment
```bash
conda create --prefix .llm_benchmark python=3.10 -y
```
2. Activate the environment
```bash
conda activate llm_benchmark
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
4. Download the dataset [LLM_SRP](https://github.com/Felipetoledo4815/LLM_SRP). Make sure it is cloned inside this directory.
```bash
git clone git@github.com:Felipetoledo4815/LLM_SRP.git
```
5. Follow the instructions in the LLM_SRP repository to set up the datasets.
6. Once the datasets are up and running, within the LLM_SRP folder, install the package in the environment by running
```bash
pip install -e .
```

## Download models
Please follow the instructions in this [README](./models/README.md).

## Prompt engineering
There is a python module called `./tests/run_prompt_engineering.py` that can be used to generate prompts for the models. Essentially, this module loads a model and a mode in memory, and takes in a text file path, where you can update the prompt, and after clicking enter, the model will use the updated prompt from the text file to generate the output. The module can be used by running:
```bash
python -m tests.run_prompt_engineering --model llava_1.5 --mode 2 --prompt_path ./tests/prompt.txt
```
Note that the module will first load whatever is in the text file, used it as a prompt for the specified mode, and generate the results. After the results are generated, the module will wait for you to update the prompt in the text file, and press enter to generate the next results.