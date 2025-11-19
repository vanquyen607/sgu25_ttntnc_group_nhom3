import pandas as pd
from pathlib import Path
from Naive_Bayes import Naive_Bayes

# build dataset path relative to project root (one level above code/)
base = Path(__file__).resolve().parent.parent
dataset_path = base / '/code/dataset' / 'Iris.csv'

if not dataset_path.exists():
    # fallback: create dataset folder and save sklearn iris to CSV
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df_tmp = pd.DataFrame(iris.data, columns=iris.feature_names)
        # normalize column names to match common Iris.csv (no units)
        df_tmp.columns = [c.replace(' (cm)', '').strip() for c in df_tmp.columns]
        df_tmp['Species'] = [iris.target_names[i] for i in iris.target]
        df_tmp.to_csv(dataset_path, index=False)
        print(f"Saved sklearn Iris dataset to {dataset_path}")
    except Exception as e:
        raise FileNotFoundError(f"Dataset not found and failed to generate fallback: {dataset_path}") from e

df = pd.read_csv(dataset_path)
# drop Id column if present
if 'Id' in df.columns:
    df.drop(['Id'], axis=1, inplace=True)

# shuffle rows deterministically (optional)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split
train_df = df.iloc[:120].reset_index(drop=True)
test_df = df.iloc[120:].reset_index(drop=True)

# CLASSIFIER
nb = Naive_Bayes(train_df)
nb.test(test_df)