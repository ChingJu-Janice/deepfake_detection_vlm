import json
import glob
import pandas as pd


def main():
    all_results = []
    for fpath in glob.glob('./results/ablation/*.json'):
        with open(fpath) as f:
            all_results.append(json.load(f))

    df = pd.DataFrame(all_results)
    df.to_csv("./results/ablation/ablation_results.csv", index=False)
    
if __name__ == "__main__":
    main()