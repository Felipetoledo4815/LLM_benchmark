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

## Runing the benchmark

Use `run.py` to evaluate a model on the LLM_SRP dataset with your chosen prompt mode and shot setting. Here’s an example:

```bash
# Evaluate a fine-tuned LLaVA 1.5 on Mode 1 with zero-shot over 100 images
python run.py \
	--model llava_1.5 \
	--log_folder logs/study/ \
	--mode 1 \
	--shot zero \
	--nr_images 100
```

Arguments overview
- --model: Which VLM to use. Supported values include:
- --mode: Prompting mode (1|2|3|4). Each mode uses a different prompt template defined in `prompts/mode{1..4}` and query logic in `utils/query_mode.py`.
- --shot: Few‑shot setting for the prompt templates. One of `zero`, `one`, `two`.
- --nr_images: Number of images sampled from the dataset for evaluation (default 10). Increase to evaluate more samples; set higher than the dataset size to effectively run on all available test images.
- --log_folder: Output directory for results (default `./logs/test/`). The runner creates a subfolder per experiment and writes:
	- `log.txt`: Per‑image predictions, parsed triplets, and per‑image metrics.
	- `metrics.json`: List of per‑image metric records, including triplet‑level stats. This one is then used by the monitor.
	- `metrics.pkl`: Serialized `Metrics` object for later analysis.
	- `heatmaps.png`: Precision/recall heatmaps by relationship and entity.

Notes
- Ensure you’ve set up the datasets as described above and installed `LLM_SRP` with `pip install -e .` inside its folder.
- The script prints aggregate metrics (recall, precision, macro/micro F1, TP/FP/FN) on completion, and writes the detailed logs/plots into the experiment folder under `--log_folder`.