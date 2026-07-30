import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import gdelt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

gd = gdelt.gdelt()
relevant_cols = ["SQLDATE", "Actor1Name", "Actor2Name", "SOURCEURL"]
start_date = date(2021, 1, 1)
end_date = date(2025, 12, 31)

SCRAPE_TIMEOUT = 5
MAX_SCRAPE_WORKERS = 30

DAILY_DIR = "daily"
OUTPUT_PATH = "output.parquet"


def date_range(start, end):
    """Yield consecutive dates between start and end as GDELT-formatted strings.

    Args:
        start: start date (inclusive) as a datetime.date object
        end: end date (inclusive) as a datetime.date object

    Yields:
        str: each date formatted as "DD MMMM YYYY" (e.g. "01 January 2025"),
             which is the format required by the GDELT Search API
    """
    current = start
    while current <= end:
        yield current.strftime("%d %B %Y")
        current += timedelta(days=1)


def day_path(i):
    """Return the parquet path for the i-th day (zero-padded for sortability)."""
    return os.path.join(DAILY_DIR, f"day_{i:04d}.parquet")


def run_query(date_str, retries=3):
    """Query the GDELT database for all events on a single day.

    Adds a 0.3 s sleep before each attempt to avoid overloading the GDELT
    endpoint. Returns an empty DataFrame after exhausting all retries so the
    caller can skip the day rather than abort.

    Args:
        date_str: date string in "DD MMMM YYYY" format as returned by date_range
        retries: maximum number of attempts before giving up (default 3)

    Returns:
        pd.DataFrame: rows from relevant_cols with duplicate SOURCEURL values
            removed; empty DataFrame if all attempts fail
    """
    for attempt in range(retries):
        try:
            time.sleep(0.3)
            res = gd.Search(date=date_str)
            return res[relevant_cols].drop_duplicates(subset="SOURCEURL")
        except Exception as e:
            print(f"[GDELT] {date_str} failed (attempt {attempt + 1}): {e}")
    return pd.DataFrame()


def extract_content(url):
    """Fetch and parse the plain-text body of a single article URL.

    Non-HTML responses (PDFs, RSS feeds, binary content) are detected via
    the Content-Type header and skipped. Script, style, nav, and footer
    elements are stripped before text extraction to reduce boilerplate.
    SCRAPE_TIMEOUT is set to 5 s; values below ~2 s cause excessive
    failures on slow news sites. All exceptions are silenced because a
    high failure rate is expected when scraping thousands of URLs.

    Args:
        url: fully-qualified URL of the article to fetch

    Returns:
        tuple[str, str | None]: (url, text) pair where text is the
            whitespace-normalised plain-text body, or None if the URL is
            non-HTML, unreachable, or raises any exception
    """
    try:
        res = requests.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        if "text/html" not in res.headers.get("Content-Type", ""):
            return (url, None)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return (url, soup.get_text(separator=" ", strip=True))
    except Exception:
        return (url, None)


def scrape_all(urls, max_workers=MAX_SCRAPE_WORKERS):
    """Scrape a list of article URLs concurrently using a thread pool.

    Args:
        urls: iterable of fully-qualified article URLs to scrape
        max_workers: size of the thread pool (default MAX_SCRAPE_WORKERS)

    Returns:
        dict[str, str | None]: mapping from URL to extracted plain-text body,
            or None for every URL that failed or returned non-HTML content
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_content, url): url for url in urls}
        for future in as_completed(futures):
            url, content = future.result()
            results[url] = content
    return results


def combine_daily(output_path=OUTPUT_PATH):
    """Concatenate all per-day parquet files into a single output file.

    Empty placeholder files (days with no usable content) are skipped.

    Args:
        output_path: destination path for the combined parquet file
    """
    parts = []
    for p in sorted(glob.glob(os.path.join(DAILY_DIR, "day_*.parquet"))):
        part = pd.read_parquet(p)
        if not part.empty:
            parts.append(part)

    if not parts:
        print("No data to combine.")
        return

    df_final = pd.concat(parts, ignore_index=True)
    df_final.to_parquet(output_path)
    print(f"Done. {len(df_final)} rows saved to {output_path}.")


def main():
    """Run the full extraction pipeline: query GDELT day by day, scrape each
    day's article URLs, cache the results per day, then combine everything
    into a single output file."""
    os.makedirs(DAILY_DIR, exist_ok=True)
    dates = list(date_range(start_date, end_date))

    for i, date_str in enumerate(tqdm(dates, desc="Processing days")):
        if os.path.exists(day_path(i)):
            continue

        df = run_query(date_str)
        if df.empty:
            pd.DataFrame(columns=relevant_cols + ["CONTENT"]).to_parquet(day_path(i))
            continue

        results = scrape_all(df["SOURCEURL"].tolist())
        df["CONTENT"] = df["SOURCEURL"].map(results)
        df = df[df["CONTENT"].notna()]

        tmp = day_path(i) + ".tmp"
        df.to_parquet(tmp)
        os.replace(tmp, day_path(i))

    combine_daily()


if __name__ == "__main__":
    main()
