# --- config ---
IMDB_DIR = data/imdb
PROC_DIR = data/processed
SQL_DIR = sql
PY = python
DUCKDB = duckdb


# IMDb dataset URLs (non-commercial TSVs)
IMDB_BASE = https://datasets.imdbws.com
BASICS_URL = $(IMDB_BASE)/title.basics.tsv.gz
RATINGS_URL = $(IMDB_BASE)/title.ratings.tsv.gz
PRINCIPALS_URL= $(IMDB_BASE)/title.principals.tsv.gz
NAMES_URL = $(IMDB_BASE)/name.basics.tsv.gz


.PHONY: all setup download-imdb build-modeling-base build-dataset train clean


all: setup download-imdb build-modeling-base build-dataset train


setup:
	mkdir -p $(IMDB_DIR) $(PROC_DIR)


# Download the four core TSVs
download-imdb: setup
	curl -L $(BASICS_URL) -o $(IMDB_DIR)/title.basics.tsv.gz
	curl -L $(RATINGS_URL) -o $(IMDB_DIR)/title.ratings.tsv.gz
	curl -L $(PRINCIPALS_URL) -o $(IMDB_DIR)/title.principals.tsv.gz
	curl -L $(NAMES_URL) -o $(IMDB_DIR)/name.basics.tsv.gz


# Build labels (nconst, tconst, rank_in_known_for) using DuckDB
build-modeling-base: $(SQL_DIR)/build_modeling_base.sql
	$(DUCKDB) < $(SQL_DIR)/build_modeling_base.sql \
	> /dev/null


# Build modeling table (actor–title pairs + features + label)
# build-dataset: src/build_dataset.py
# 	$(PY) src/build_dataset.py --imdb_dir $(IMDB_DIR) --out $(PROC_DIR)/modeling_table.parquet

build-features:
	$(PY) src/build_features.py $(PROC_DIR)/modeling_base.parquet $(PROC_DIR)/modeling_ml.parquet


# Train a simple baseline (logistic regression) and write metrics
train: src/train_baseline.py
	$(PY) src/train_baseline.py $(PROC_DIR)/modeling_table.parquet $(PROC_DIR)/baseline_metrics.json


clean:
	rm -rf $(PROC_DIR)/*.parquet $(PROC_DIR)/*.json