import json
import time

import anthropic
import pandas as pd
from prompt import SYSTEM_PROMPT
from utils.filter import clean_scraped_text, is_esg_keyword_match

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
INPUT_FILE = "output.parquet"
BATCH_ID_FILE = "batch_id.txt"
DRY_RUN = False
DRY_RUN_FILE = "dry_run_preview.jsonl"
CONTENT_MAX_CHARS = 100_000
MIN_BODY_CHARS = 300
BATCH_SIZE = 100_000

PRICE_IN_PER_TOK = 0.50 / 1_000_000
PRICE_OUT_PER_TOK = 2.50 / 1_000_000


def build_request(idx: int, row: dict) -> dict:
    """Build a single Claude Message Batches API request object.

    Uses the pre-cleaned, truncated body in CONTENT_CLEAN (computed once in
    main()), so cleaning is NOT repeated here. The idx is stored as custom_id so
    batch_collect.py can join results back to the source DataFrame by index.
    """
    url = row.get("SOURCEURL", "")
    content = str(row.get("CONTENT_CLEAN", ""))[:CONTENT_MAX_CHARS]
    return {
        "custom_id": str(idx),
        "params": {
            "model": CLAUDE_MODEL,
            "max_tokens": 300,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": f"URL: {url}\n\nArticle:\n{content}"}
            ],
        },
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate for cost preview only (≈ 4 chars per token)."""
    return max(1, len(text) // 4)


def write_dry_run(requests_list: list[dict], path: str) -> None:
    """Write a preview JSONL of exactly what would be sent, plus a cost estimate.

    Prints progress every 5,000 rows so a long write is visibly making progress
    rather than looking hung.
    """
    sys_tokens = estimate_tokens(SYSTEM_PROMPT)

    total_in = 0
    total_out = 0
    n = len(requests_list)
    t0 = time.time()
    with open(path, "w") as f:
        for k, req in enumerate(requests_list, 1):
            user_text = req["params"]["messages"][0]["content"]
            total_in += sys_tokens + estimate_tokens(user_text)
            total_out += req["params"]["max_tokens"]
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            if k % 5_000 == 0:
                print(f"  ...wrote {k:,}/{n:,} ({time.time() - t0:.1f}s)", flush=True)

    est_in_cost = total_in * PRICE_IN_PER_TOK
    est_out_cost_max = total_out * PRICE_OUT_PER_TOK

    print("\n── DRY RUN ───────────────────────────────────────────────────────")
    print(f"  Requests that WOULD be sent : {n:,}")
    print(f"  System prompt (est. tokens) : {sys_tokens:,} per request")
    print(f"  Est. input tokens (total)   : {total_in:,}")
    print(f"  Est. output tokens (max cap): {total_out:,}")
    print("  ─────────────────────────────────────────")
    print(f"  Est. input cost             : ${est_in_cost:,.2f}")
    print(f"  Est. output cost (max)      : ${est_out_cost_max:,.2f}")
    print(f"  Est. TOTAL (upper bound)    : ${est_in_cost + est_out_cost_max:,.2f}")
    print("  NOTE: token counts are a ~4-chars/token heuristic, not exact.")


def main():
    """Run the full categorization pipeline: load scraped articles, clean and
    filter them by ESG keyword relevance, then either preview the resulting
    batch requests (DRY_RUN) or submit them to the Claude Message Batches API."""
    t0 = time.time()
    df = pd.read_parquet(INPUT_FILE)
    df = df[df["CONTENT"].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} rows with content ({time.time() - t0:.1f}s)", flush=True)

    t1 = time.time()
    df["CONTENT_CLEAN"] = (
        df["CONTENT"].astype(str).str[: CONTENT_MAX_CHARS * 4].apply(clean_scraped_text)
    )
    print(f"Cleaned text ({time.time() - t1:.1f}s)", flush=True)

    before = len(df)
    df = df[df["CONTENT_CLEAN"].str.len() >= MIN_BODY_CHARS].reset_index(drop=True)
    print(
        f"After body-length filter (>= {MIN_BODY_CHARS}): {len(df)} / {before}",
        flush=True,
    )

    t2 = time.time()
    df["keyword_match"] = df["CONTENT_CLEAN"].apply(
        lambda x: is_esg_keyword_match(x[:CONTENT_MAX_CHARS])
    )
    df_filtered = df[df["keyword_match"]].reset_index(drop=True)
    print(
        f"After keyword pre-filter: {len(df_filtered)} / {len(df)} ({time.time() - t2:.1f}s)",
        flush=True,
    )

    requests_list = [build_request(i, row) for i, row in df_filtered.iterrows()]

    if DRY_RUN:
        write_dry_run(requests_list, DRY_RUN_FILE)
        df_filtered.to_parquet("filtered_articles.parquet", index=False)
        print("\nFiltered articles saved to filtered_articles.parquet")
        print(f"Dry run complete in {time.time() - t0:.1f}s. No batches submitted.")
        return

    client = anthropic.Anthropic()
    batch_ids = []

    for i in range(0, len(requests_list), BATCH_SIZE):
        chunk = requests_list[i : i + BATCH_SIZE]
        batch = client.beta.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"Submitted batch {batch.id} with {len(chunk)} requests")

    with open(BATCH_ID_FILE, "w") as f:
        json.dump({"batch_ids": batch_ids, "total_rows": len(df_filtered)}, f)

    df_filtered.to_parquet("filtered_articles.parquet", index=False)
    print(f"\nDone. Batch IDs saved to {BATCH_ID_FILE}")
    print("Filtered articles saved to filtered_articles.parquet")
    print("Now wait ~15min–1hr, then run batch_collect.py")


if __name__ == "__main__":
    main()
