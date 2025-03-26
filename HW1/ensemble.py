import os
import pandas as pd

root_path = "./result/ensemble"

csv_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(".csv") and f != "prediction.csv"]
dfs = [pd.read_csv(f) for f in csv_files]
print(csv_files)

ensemble_df = dfs[0][['image_name']].copy()
for i, df in enumerate(dfs):
    ensemble_df[f'pred_{i+1}'] = df['pred_label']


def majority_vote(row):
    return row.value_counts().idxmax()


pred_cols = [f'pred_{i+1}' for i in range(len(dfs))]
ensemble_df['pred_label'] = ensemble_df[pred_cols].apply(majority_vote, axis=1)


final_result = ensemble_df[['image_name', 'pred_label']]
final_result.to_csv(os.path.join(root_path, "prediction.csv"), index=False)
