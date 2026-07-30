import json
import time

import anthropic
import pandas as pd

BATCH_ID_FILE = "batch_id.txt"
FILTERED_FILE = "filtered_articles.parquet"
OUTPUT_FILE = "results_final.csv"
POLL_INTERVAL_S = 60

DEFAULT_RESULT = {
    "esg_relevant": False,
    "asset_category": "neutral",
    "sentiment": "neutral",
    "region": None,
    "combined_label": "unclassified",
    "company_mentioned": None,
    "source": None,
    "publication_type": None,
    "reach": None,
    "reach_region": None,
    "reason": "parse_error",
}


def load_batch_ids(path: str) -> list[str]:
    """Load the batch IDs written by main.py after submitting the classification batches."""
    with open(path) as f:
        meta = json.load(f)
    return meta["batch_ids"]


def wait_for_batches(client: anthropic.Anthropic, batch_ids: list[str]) -> None:
    """Poll the Claude Message Batches API until every batch has finished processing.

    Prints a status line per batch on each poll so a long-running wait is
    visibly making progress rather than looking hung.
    """
    print(f"Waiting for {len(batch_ids)} batch(es) to complete...")
    for batch_id in batch_ids:
        while True:
            batch = client.beta.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts
            print(
                f"  [{batch_id}] status={status} | "
                f"succeeded={counts.succeeded} "
                f"errored={counts.errored} "
                f"processing={counts.processing}"
            )
            if status == "ended":
                break
            time.sleep(POLL_INTERVAL_S)


def parse_batch_result(raw_text: str) -> dict:
    """Parse one batch result's JSON payload, stripping any markdown code fences."""
    cleaned = (
        raw_text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    return json.loads(cleaned)


def collect_batch_results(client: anthropic.Anthropic, batch_ids: list[str]) -> dict[int, dict]:
    """Fetch and parse every result across all batches, keyed by the original row index.

    The row index was stored as custom_id when the request was built (see
    build_request in main.py), which is how results are joined back to the
    source DataFrame. Rows that fail to parse fall back to DEFAULT_RESULT so
    a single malformed response does not abort the whole run.
    """
    esg_results = {}
    for batch_id in batch_ids:
        for result in client.beta.messages.batches.results(batch_id):
            idx = int(result.custom_id)
            try:
                data = parse_batch_result(result.result.message.content[0].text)
            except Exception as e:
                print(f"  Parse error for idx {idx}: {e}")
                data = DEFAULT_RESULT
            esg_results[idx] = data
    print(f"Collected {len(esg_results)} results")
    return esg_results


def merge_results_into_df(df: pd.DataFrame, esg_results: dict[int, dict]) -> pd.DataFrame:
    """Attach each classification field from esg_results onto df as its own column."""
    df = df.copy()
    fields = [
        ("esg_relevant", False),
        ("asset_category", "neutral"),
        ("sentiment", "neutral"),
        ("region", None),
        ("combined_label", "unclassified"),
        ("source", "unclassified"),
        ("publication_type", "unclassified"),
        ("reach", "unclassified"),
        ("reach_region", "unclassified"),
        ("company_mentioned", None),
    ]
    for field, default in fields:
        df[field] = df.index.map(lambda i, field=field, default=default: esg_results.get(i, {}).get(field, default))
    return df


def print_summary(df: pd.DataFrame, df_esg: pd.DataFrame, output_file: str) -> None:
    """Print counts and distributions for the final ESG-relevant dataset."""
    print(f"\nTotal articles processed:  {len(df)}")
    print(f"ESG relevant:              {len(df_esg)}")
    print(f"\nAsset category distribution:\n{df_esg['asset_category'].value_counts()}")
    print(f"\nSentiment distribution:\n{df_esg['sentiment'].value_counts()}")
    print(f"\nCombined label distribution:\n{df_esg['combined_label'].value_counts()}")
    print(f"\nFinal results saved to {output_file}")


def main():
    """Wait for the batches submitted by main.py to finish, collect and parse
    their results, join them onto the filtered articles, and write the final
    ESG-relevant dataset to OUTPUT_FILE."""
    client = anthropic.Anthropic()
    batch_ids = load_batch_ids(BATCH_ID_FILE)

    wait_for_batches(client, batch_ids)
    print("\nAll batches complete. Collecting results...")
    esg_results = collect_batch_results(client, batch_ids)

    df = pd.read_parquet(FILTERED_FILE)
    df = merge_results_into_df(df, esg_results)
    df_esg = df[df["esg_relevant"]].reset_index(drop=True)
    df_esg.to_csv(OUTPUT_FILE, index=False)

    print_summary(df, df_esg, OUTPUT_FILE)


if __name__ == "__main__":
    main()
