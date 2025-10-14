# src/build_dataset.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Utility to read TSV.GZ with IMDb's \N as NA


def read_tsv(path: str, usecols=None):
    return pd.read_csv(path, sep='\t', na_values='\\N', usecols=usecols, low_memory=False)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imdb_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()


    imdb_dir = Path(args.imdb_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load core tables
    names = read_tsv(imdb_dir / 'name.basics.tsv.gz', usecols=['nconst', 'primaryName', 'knownForTitles'])
    principals = read_tsv(imdb_dir / 'title.principals.tsv.gz', usecols=['tconst','nconst','category','ordering'])
    basics = read_tsv(imdb_dir / 'title.basics.tsv.gz', usecols=['tconst','primaryTitle','titleType','startYear','genres'])
    ratings = read_tsv(imdb_dir / 'title.ratings.tsv.gz', usecols=['tconst','averageRating','numVotes'])


    # Build labels (nconst, tconst, rank)
    labels = (
        names.dropna(subset=['knownForTitles'])
            .assign(tlist=lambda d: d['knownForTitles'].str.split(','))
            .explode('tlist')
            .rename(columns={'tlist':'tconst'})[['nconst','tconst']]
    )
    # Add rank by preserving original order
    rank_rows = []
    for nconst, kf in names.dropna(subset=['knownForTitles'])[['nconst','knownForTitles']].itertuples(index=False):
        parts = str(kf).split(',')
        for rank, t in enumerate(parts):
            rank_rows.append((nconst, t, rank))
    labels_ranked = pd.DataFrame(rank_rows, columns=['nconst','tconst','rank_in_known_for'])


    # Define actor universe: all titles a person is credited on (focus on cast; tweak as needed)
    cast = principals[principals['category'].isin(['actor','actress','self'])].copy()


    # Join features onto titles
    title_feats = (basics.merge(ratings, on='tconst', how='left'))


    # Candidate pairs (nconst, tconst) with role ordering for potential "leadness"
    # Lower ordering often indicates higher billing
    ordering_min = (principals.groupby(['tconst','nconst'], as_index=False)['ordering'].min())


    pairs = (cast[['nconst','tconst']]
                .drop_duplicates()
                .merge(ordering_min, on=['nconst','tconst'], how='left')
                .merge(title_feats, on='tconst', how='left'))


    # Label positives
    pairs['label'] = pairs.merge(labels_ranked[['nconst','tconst']], on=['nconst','tconst'], how='left')['tconst'].notna().astype(int)


    # Feature engineering (minimal)
    # recency: years since release (assume current year 2025; or compute dynamically)
    pairs['startYear'] = pd.to_numeric(pairs['startYear'], errors='coerce')
    CURRENT_YEAR = 2025
    pairs['years_since_release'] = CURRENT_YEAR - pairs['startYear']


    # leadness: inverse of billing order (smaller ordering => more lead). Fill with large for missing
    pairs['ordering'] = pd.to_numeric(pairs['ordering'], errors='coerce')
    pairs['leadness'] = 1 / (pairs['ordering'].fillna(pairs['ordering'].max() + 5))


    # Title type flags (example: movie vs tv)
    pairs['is_movie'] = (pairs['titleType'] == 'movie').astype(int)
    pairs['is_tvseries'] = (pairs['titleType'] == 'tvSeries').astype(int)


    # Handle genres (simple: count genres)
    pairs['genre_count'] = pairs['genres'].fillna('').str.count(',') + (pairs['genres'].notna()).astype(int)


    # Fill numeric NAs
    for col in ['averageRating','numVotes','years_since_release','leadness','genre_count']:
        pairs[col] = pd.to_numeric(pairs[col], errors='coerce').fillna(0)


    # Keep useful columns
    keep_cols = [
        'nconst','tconst','primaryTitle','titleType','startYear','genres',
        'averageRating','numVotes','years_since_release','leadness','is_movie','is_tvseries','genre_count','label'
    ]
    pairs = pairs.merge(basics[['tconst','primaryTitle']], on='tconst', how='left')
    pairs = pairs[keep_cols].drop_duplicates()


    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out_path, index=False)
    print(f"Wrote {len(pairs):,} rows -> {out_path}")


if __name__ == '__main__':
    main()