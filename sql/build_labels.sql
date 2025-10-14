-- sql/build_labels.sql
-- Run via: duckdb -c "ATTACH ':memory:' AS db; \r sql/build_labels.sql;"


.mode csv
.headers on


-- Create tables directly from gz TSVs (DuckDB can read gzipped CSV/TSV)
CREATE OR REPLACE TABLE name_basics AS
SELECT * FROM read_csv_auto('data/imdb/name.basics.tsv.gz', delim='\t', nullstr='\\N');


-- Explode knownForTitles to (nconst, tconst, rank)
CREATE OR REPLACE TABLE known_for_labels AS
WITH exploded AS (
    SELECT
     nconst,
     unnest(string_split(knownForTitles, ',')) AS tconst,
     row_number() OVER (PARTITION BY nconst ORDER BY 1) - 1 AS rank_in_known_for
    FROM name_basics
    WHERE knownForTitles IS NOT NULL
)
SELECT * FROM exploded;


-- Export a clean CSV/Parquet for downstream steps
COPY (SELECT * FROM known_for_labels) TO 'data/processed/known_for_labels.parquet' (FORMAT 'parquet');