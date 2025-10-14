import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Optional imports (only if --node2vec)
try:
    import networkx as nx
    from node2vec import Node2Vec
except Exception:
    Node2Vec = None
    nx = None

CURRENT_YEAR = 2025
EMB_DIM_DEFAULT = 64


def load_base(base_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(base_path)
    # Basic sanity
    req = {
        "nconst",
        "tconst",
        "primaryTitle",
        "titleType",
        "startYear",
        "genres",
        "averageRating",
        "numVotes",
        "category",
        "min_ordering",
        "same_job_count_on_title",
        "label",
    }
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"modeling_base missing columns: {missing}")
    return df


def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Recency
    out["recency"] = CURRENT_YEAR - pd.to_numeric(
        out["startYear"], errors="coerce"
    ).fillna(CURRENT_YEAR)

    # Leadness and billing prominence
    mo = pd.to_numeric(out["min_ordering"], errors="coerce")
    out["leadness"] = 1.0 / mo.replace({0: np.nan})
    out["leadness"] = out["leadness"].fillna(0.0)

    sj = pd.to_numeric(out["same_job_count_on_title"], errors="coerce").replace(
        {0: np.nan}
    )
    out["billing_prominence"] = 1.0 / sj
    out["billing_prominence"] = out["billing_prominence"].fillna(0.0)

    # Popularity transforms
    out["numVotes"] = pd.to_numeric(out["numVotes"], errors="coerce").fillna(0)
    out["averageRating"] = pd.to_numeric(out["averageRating"], errors="coerce").fillna(
        0.0
    )
    out["log_numVotes"] = np.log1p(out["numVotes"])
    out["popularity_score"] = out["averageRating"] * out["log_numVotes"]

    # One-hots: titleType, category (keep all; or restrict to top-K if you want)
    out = pd.get_dummies(
        out, columns=["titleType", "category"], prefix=["type", "job"], dtype=np.uint8
    )

    # Simple multi-hot genres (common ones; adjust list as you like)
    common_genres = [
        "Drama",
        "Comedy",
        "Action",
        "Adventure",
        "Romance",
        "Thriller",
        "Crime",
        "Horror",
        "Sci-Fi",
        "Fantasy",
        "Animation",
    ]
    gser = out["genres"].fillna("")
    for g in common_genres:
        out[f"genre_{g}"] = gser.str.contains(rf"\b{g}\b", regex=True).astype(np.uint8)

    return out


def build_graph_from_principals(imdb_dir: Path, cast_only=True) -> pd.DataFrame:
    """Return edges dataframe: nconst, tconst, leadness (1/min ordering)."""
    import duckdb

    categories = (
        ("'actor','actress','self'")
        if cast_only
        else ("'actor','actress','self','director','writer','producer'")
    )
    q = f"""
    SELECT tconst, nconst, MIN(try_cast(ordering AS INTEGER)) AS min_ordering
    FROM read_csv_auto('{(imdb_dir / "title.principals.tsv.gz").as_posix()}', delim='\t', nullstr='\\N')
    WHERE category IN ({categories})
    GROUP BY 1,2;
    """
    lead = duckdb.query(q).to_df()
    lead["min_ordering"] = pd.to_numeric(lead["min_ordering"], errors="coerce")
    lead["leadness"] = (1.0 / lead["min_ordering"].replace({0: np.nan})).fillna(0.0)
    return lead[["nconst", "tconst", "leadness"]].dropna()


def build_graph_from_base(mb: pd.DataFrame) -> pd.DataFrame:
    """Fallback: edges from modeling_base rows."""
    tmp = mb[["nconst", "tconst", "min_ordering"]].drop_duplicates()
    tmp["min_ordering"] = pd.to_numeric(tmp["min_ordering"], errors="coerce")
    tmp["leadness"] = (1.0 / tmp["min_ordering"].replace({0: np.nan})).fillna(0.0)
    return tmp[["nconst", "tconst", "leadness"]].dropna()


def run_node2vec(
    edges: pd.DataFrame,
    emb_dim: int,
    p=1.0,
    q=1.0,
    walk_length=20,
    num_walks=200,
    workers=4,
):
    if Node2Vec is None or nx is None:
        raise RuntimeError(
            "node2vec/networkx not installed. Run: pip install node2vec networkx"
        )
    G = nx.Graph()
    G.add_nodes_from(edges["nconst"].unique(), bipartite="person")
    G.add_nodes_from(edges["tconst"].unique(), bipartite="title")
    G.add_weighted_edges_from(
        edges[["nconst", "tconst", "leadness"]].itertuples(index=False, name=None)
    )
    n2v = Node2Vec(
        G,
        dimensions=emb_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
        weight_key="weight",
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, sg=1, epochs=1)
    return model


def vectors_to_df(model, ids, id_col):
    dim = model.wv.vector_size
    rows = []
    for i in ids:
        if i in model.wv:
            rows.append((i, model.wv[i].astype(np.float32)))
        else:
            rows.append((i, np.zeros(dim, dtype=np.float32)))
    arr = np.vstack([r[1] for r in rows])
    df = pd.DataFrame(arr, columns=[f"emb_{k}" for k in range(arr.shape[1])])
    df.insert(0, id_col, [r[0] for r in rows])
    return df


def compute_costar_features(
    mb: pd.DataFrame, credits_edges: pd.DataFrame, person_vec_df: pd.DataFrame
):
    """Return DF with emb_self_norm, emb_costar_norm, emb_self_costar_cosine per (nconst,tconst)."""
    from numpy.linalg import norm

    dim = len([c for c in person_vec_df.columns if c.startswith("emb_")])
    # map person -> vector
    pvec = person_vec_df.set_index("nconst")[
        [f"emb_{k}" for k in range(dim)]
    ].to_numpy()
    pidx = {k: i for i, k in enumerate(person_vec_df["nconst"].tolist())}

    # build cast list per title
    cast_by_title = credits_edges.groupby("tconst")["nconst"].apply(list).to_dict()

    def get_vec(n):
        i = pidx.get(n, None)
        if i is None:
            return np.zeros(dim, dtype=np.float32)
        return pvec[i]

    def mean_costar(t, self_n):
        cast = [n for n in cast_by_title.get(t, []) if n != self_n]
        if not cast:
            return np.zeros(dim, dtype=np.float32)
        M = np.stack([get_vec(n) for n in cast], axis=0)
        return M.mean(0)

    def cosine(a, b, eps=1e-8):
        na, nb = norm(a), norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(a @ b / (na * nb))

    rows = []
    for r in mb[["nconst", "tconst"]].itertuples(index=False):
        sv = get_vec(r.nconst)
        cv = mean_costar(r.tconst, r.nconst)
        rows.append(
            (r.nconst, r.tconst, float(norm(sv)), float(norm(cv)), cosine(sv, cv))
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "nconst",
            "tconst",
            "emb_self_norm",
            "emb_costar_norm",
            "emb_self_costar_cosine",
        ],
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="data/processed/modeling_base.parquet",
        help="input base parquet",
    )
    ap.add_argument(
        "--out", default="data/processed/modeling_ml.parquet", help="output ML parquet"
    )
    ap.add_argument(
        "--node2vec", action="store_true", help="compute Node2Vec co-star features"
    )
    ap.add_argument(
        "--imdb_dir",
        default="data/imdb",
        help="for --node2vec, path to TSVs (title.principals.tsv.gz)",
    )
    ap.add_argument("--emb_dim", type=int, default=EMB_DIM_DEFAULT)
    ap.add_argument(
        "--save-embeddings",
        action="store_true",
        help="persist person embeddings parquet for reuse",
    )
    ap.add_argument(
        "--person_emb_path", default="data/processed/person_embeddings.parquet"
    )
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    imdb_dir = Path(args.imdb_dir)

    df = load_base(base)
    df_feats = add_tabular_features(df)

    if not args.node2vec:
        df_feats.to_parquet(out, index=False)
        print(f"[OK] Wrote {out} with tabular features only: {df_feats.shape}")
        return

    # Build edges (prefer principals so graph is broader than your training subset)
    edges = build_graph_from_principals(imdb_dir, cast_only=True)

    # If cached person embeddings exist, reuse
    person_emb_path = Path(args.person_emb_path)
    if person_emb_path.exists():
        person_vec_df = pd.read_parquet(person_emb_path)
        if not all(
            c.startswith("emb_") for c in person_vec_df.columns if c != "nconst"
        ):
            raise ValueError("Cached embeddings file has unexpected schema.")
    else:
        model = run_node2vec(edges, emb_dim=args.emb_dim)
        # Save person embeddings (we only need person vectors for co-star means)
        person_ids = edges["nconst"].unique()
        person_vec_df = vectors_to_df(model, person_ids, "nconst")
        if args.save_embeddings:
            person_emb_path.parent.mkdir(parents=True, exist_ok=True)
            person_vec_df.to_parquet(person_emb_path, index=False)

    # Compute co-star features for each (nconst,tconst) in the modeling base
    emb_feats = compute_costar_features(df, edges, person_vec_df)

    # Merge into feature table
    df_all = df_feats.merge(emb_feats, on=["nconst", "tconst"], how="left")
    for c in ["emb_self_norm", "emb_costar_norm", "emb_self_costar_cosine"]:
        if c not in df_all:
            df_all[c] = 0.0
        df_all[c] = df_all[c].fillna(0.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out, index=False)
    print(f"[OK] Wrote {out} with Node2Vec features: {df_all.shape}")


if __name__ == "__main__":
    main()
