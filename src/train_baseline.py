# src/train_baseline.py
from sklearn.pipeline import Pipeline
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

METRICS = {}


FEATURES = [
    'averageRating','numVotes','years_since_release','leadness','is_movie','is_tvseries','genre_count'
]




def accuracy_at_4(df: pd.DataFrame) -> float:
    # For each nconst, pick top-4 by pred and compare to label==1 set
    accs = []
    for nconst, g in df.groupby('nconst'):
        true_set = set(g.loc[g['label']==1, 'tconst'])
        if len(true_set) == 0:
            continue
        pred_top4 = list(g.sort_values('pred', ascending=False)['tconst'][:4])
        hit = len(set(pred_top4) & true_set)
        accs.append(hit/4)
    return float(np.mean(accs)) if accs else 0.0




def mean_jaccard_at_4(df: pd.DataFrame) -> float:
    vals = []
    for nconst, g in df.groupby('nconst'):
        true_set = set(g.loc[g['label']==1, 'tconst'])
        if not true_set:
            continue
        pred_set = set(g.sort_values('pred', ascending=False)['tconst'][:4])
        inter = len(true_set & pred_set)
        union = len(true_set | pred_set)
        vals.append(inter/union if union else 0.0)
    return float(np.mean(vals)) if vals else 0.0




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()


    df = pd.read_parquet(args.data)
    # Only keep actors who have at least 1 positive label (otherwise undefined)
    mask_has_pos = df.groupby('nconst')['label'].transform('sum') > 0
    df = df[mask_has_pos].copy()


    X = df[FEATURES].values
    y = df['label'].values


    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(max_iter=200, n_jobs=1, class_weight='balanced')),
    ])
    pipe.fit(X, y)


    # Predict probabilities for ranking
    df['pred'] = pipe.predict_proba(X)[:,1]


    metrics = {
        'n_rows': int(len(df)),
        'n_actors': int(df['nconst'].nunique()),
        'accuracy_at_4': accuracy_at_4(df),
        'mean_jaccard_at_4': mean_jaccard_at_4(df),
    }


    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)


    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()  