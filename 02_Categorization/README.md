# Categorization

this folder holds the code that is used to turn the extracted data into a categorized dataset. The categorization is done by using article cleaning -> deterministic prefiltering -> AI classification.

- main.py: the main script that runs the categorization. It reads `out_small.parquet` (the output.parquet file from the extraction step, renamed/pointed at via `INPUT_FILE`), cleans and keyword-prefilters the articles, then submits the survivors to the Claude Message Batches API for classification. Writes `filtered_articles.parquet` and `batch_id.txt`. When `DRY_RUN` is set to `True`, the script only runs the prefiltering and writes a cost preview to `dry_run_preview.jsonl` instead of calling the API.
- prompt.py: the Claude system prompt used by main.py to classify each article (ESG relevance, asset category, sentiment, region, source reach, etc.), kept separate so main.py reads as pipeline logic rather than prompt text.
- batch_collect.py: run after the batches submitted by main.py finish processing (~15min–1hr). Polls the batch status, collects and parses the results, joins them onto `filtered_articles.parquet`, and writes the final ESG-relevant dataset to `results_final.csv`.

You need to supply your own anthropic API key in the environment if you want to use this script
