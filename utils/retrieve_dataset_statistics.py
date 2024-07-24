import argparse
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append('../LLM_SRP/')
from dataset.llm_srp_dataset import LLMSRPDataset


def main():
    parser = argparse.ArgumentParser("Statistics for LLM_SRP Dataset.")
    parser.add_argument('--csv_path', type=str, default=None, help='Path to save the statistics into a CSV file.')
    args = parser.parse_args()

    llm_srp_dataset = LLMSRPDataset(['nuscenes'])
    entity_names = llm_srp_dataset.get_entity_names()
    relationship_names = llm_srp_dataset.get_relationship_names()

    df = pd.DataFrame(index=[rn for rn in relationship_names])
    for entity in entity_names[:-1]:
        df[entity] = 0.0

    for i in tqdm(range(len(llm_srp_dataset))):
        _, triplets = llm_srp_dataset[i]
        for e, r, _ in triplets:
            df.loc[r, e] += 1

    # Calculate totals for columns and rows
    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)

    print(df)

    if args.csv_path is not None:
        df.to_csv(args.csv_path)


if __name__ == '__main__':
    main()
