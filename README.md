# LLM_benchmark

## Getting Started
1. Create a conda environment
```bash
conda create -n LLM_benchmark python=3.10
```
2. Activate the environment
```bash
conda activate LLM_benchmark
```
3. Install the requirements
```bash
pip install -r requirements.txt
```
4. Download the dataset [LLM_SRP](https://github.com/Felipetoledo4815/LLM_SRP)
```bash
git clone git@github.com:Felipetoledo4815/LLM_SRP.git
```
5. Follow the instructions in the LLM_SRP repository to set up the datasets.
6. Once the datasets are up and running, within the LLM_SRP folder, install the package in the environment by running
```bash
pip install -e .
```