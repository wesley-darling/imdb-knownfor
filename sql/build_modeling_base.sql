---------------------------------------------
-- Build modeling_base.parquet for IMDb KF
-- DuckDB reads .tsv.gz directly
---------------------------------------------
PRAGMA threads=auto;
SET enable_progress_bar = true;

-- 1) Load IMDb TSVs
CREATE OR REPLACE TABLE name_basics AS
SELECT * FROM read_csv_auto('data/imdb/name.basics.tsv.gz', delim='\t', nullstr='\\N');

CREATE OR REPLACE TABLE title_basics AS
SELECT * FROM read_csv_auto('data/imdb/title.basics.tsv.gz', delim='\t', nullstr='\\N');

CREATE OR REPLACE TABLE title_ratings AS
SELECT * FROM read_csv_auto('data/imdb/title.ratings.tsv.gz', delim='\t', nullstr='\\N');

CREATE OR REPLACE TABLE title_principals AS
SELECT * FROM read_csv_auto('data/imdb/title.principals.tsv.gz', delim='\t', nullstr='\\N');

-- 2) Labels: explode knownForTitles (only needed to mark positives)
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

COPY (SELECT * FROM known_for_labels) TO 'data/processed/known_for_labels.parquet' (FORMAT 'parquet');

-- 3) Candidate universe: which person worked on which title
--    Include common creative roles; expand if you want more
CREATE OR REPLACE TABLE credits AS
SELECT
  tconst,
  nconst,
  category,
  try_cast(ordering AS INTEGER) AS ordering
FROM title_principals
WHERE category IN ('actor','actress','self','director','writer','producer');

-- best billing position for that person/title
CREATE OR REPLACE TABLE min_billing AS
SELECT tconst, nconst, MIN(ordering) AS min_ordering
FROM credits
GROUP BY tconst, nconst;

-- how many people share the same job on the title (for prominence)
CREATE OR REPLACE TABLE same_job_counts AS
SELECT tconst, category, COUNT(*) AS same_job_count_on_title
FROM credits
GROUP BY tconst, category;

-- title-level features
CREATE OR REPLACE TABLE title_features AS
SELECT
  b.tconst,
  b.primaryTitle,
  b.titleType,
  try_cast(b.startYear AS INTEGER) AS startYear,
  b.genres,
  r.averageRating,
  r.numVotes
FROM title_basics b
LEFT JOIN title_ratings r USING (tconst);

-- 4) Assemble base table: one row per person–title with raw fields
CREATE OR REPLACE TABLE modeling_base AS
SELECT
  c.nconst,
  nb.primaryName,                
  tf.tconst,
  tf.primaryTitle,
  tf.titleType,
  tf.startYear,
  tf.genres,
  coalesce(tf.averageRating, 0.0) AS averageRating,
  coalesce(tf.numVotes, 0)        AS numVotes,
  c.category,
  mb.min_ordering,
  coalesce(sjc.same_job_count_on_title, 1) AS same_job_count_on_title,
  (kfl.tconst IS NOT NULL)::INTEGER AS label
  kfl.rank_in_known_for   
FROM credits c
JOIN title_features tf
  ON c.tconst = tf.tconst
LEFT JOIN min_billing mb
  ON c.tconst = mb.tconst AND c.nconst = mb.nconst
LEFT JOIN same_job_counts sjc
  ON c.tconst = sjc.tconst AND c.category = sjc.category
LEFT JOIN known_for_labels kfl
  ON c.nconst = kfl.nconst AND c.tconst = kfl.tconst
LEFT JOIN name_basics nb                 
  ON c.nconst = nb.nconst;


-- keep only people who have at least one positive label
CREATE OR REPLACE TABLE modeling_base_filtered AS
SELECT mb.*
FROM modeling_base mb
JOIN (
  SELECT nconst
  FROM modeling_base
  GROUP BY nconst
  HAVING SUM(label) >= 1
) keep USING (nconst);

-- 5) Write artifact for Python
COPY (SELECT * FROM modeling_base_filtered)
TO 'data/processed/modeling_base.parquet' (FORMAT PARQUET);

-- quick sanity
-- SELECT COUNT(*) AS rows,
--        COUNT(DISTINCT nconst) AS persons,
--        COUNT(DISTINCT tconst) AS titles
-- FROM modeling_base_filtered;

-- SELECT label, COUNT(*) AS cnt
-- FROM modeling_base_filtered
-- GROUP BY 1 ORDER BY 1;