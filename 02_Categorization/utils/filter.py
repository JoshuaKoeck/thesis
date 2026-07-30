import re

NAV_CUT_MARKERS = [
    "skip to main content",
    "skip to content",
    "related news",
    "related stories",
    "most read",
    "trending now",
    "you may also like",
    "recommended for you",
    "recommended articles",
    "more from",
    "advertisement",
]

DATESTAMP_RE = re.compile(
    r"\b\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+\d{4}(?:,?\s+\d{1,2}:\d{2}\s*(?:am|pm))?",
    flags=re.IGNORECASE,
)

HEADER_ZONE_CHARS = 2500

MAX_HEAD_TRIM = 1200

PROSE_RE = re.compile(r"[A-Z][^.!?]{80,}?[a-z]{3,}[^.!?]*[.!?]")

STRONG_KEYWORDS = [
    "climate change",
    "climate risk",
    "global warming",
    "greenhouse gas",
    "greenhouse gases",
    "carbon emissions",
    "carbon neutral",
    "carbon neutrality",
    "carbon footprint",
    "carbon tax",
    "carbon pricing",
    "carbon credit",
    "carbon credits",
    "carbon offset",
    "carbon offsets",
    "carbon capture",
    "net zero",
    "net-zero",
    "decarbonization",
    "decarbonisation",
    "decarbonize",
    "decarbonise",
    "emissions trading",
    "emissions reduction",
    "emission targets",
    "emissions target",
    "paris agreement",
    "scope 1",
    "scope 2",
    "scope 3",
    "methane emissions",
    "ghg emissions",
    "co2 emissions",
    "renewable energy",
    "renewables",
    "clean energy",
    "clean tech",
    "cleantech",
    "green energy",
    "green power",
    "energy transition",
    "solar power",
    "solar energy",
    "solar panel",
    "solar panels",
    "photovoltaic",
    "wind energy",
    "wind farm",
    "wind farms",
    "wind power",
    "offshore wind",
    "onshore wind",
    "green hydrogen",
    "clean hydrogen",
    "biofuel",
    "biofuels",
    "geothermal",
    "electric vehicle",
    "electric vehicles",
    "green bond",
    "green bonds",
    "sustainable finance",
    "sustainability-linked",
    "battery storage",
    "energy storage",
    "grid storage",
    "fossil fuel",
    "fossil fuels",
    "coal-fired",
    "coal plant",
    "coal plants",
    "coal power",
    "coal mine",
    "coal mining",
    "thermal coal",
    "oil and gas",
    "oil & gas",
    "crude oil",
    "natural gas",
    "shale gas",
    "shale oil",
    "tar sands",
    "oil sands",
    "fracking",
    "hydraulic fracturing",
    "offshore drilling",
    "oil drilling",
    "oil spill",
    "gas flaring",
    "stranded assets",
    "petrochemical",
    "petrochemicals",
    "lng terminal",
    "lng export",
    "esg",
    "greenwashing",
    "transition risk",
    "physical climate risk",
    "sustainability report",
    "sustainability disclosure",
    "csrd",
    "sfdr",
    "eu taxonomy",
    "tcfd",
    "sbti",
    "science-based target",
    "science-based targets",
    "just transition",
    "divestment",
    "just transition",
    "divestment",
    "low-carbon",
    "low carbon",
    "climate transition",
    "climate target",
    "climate targets",
    "climate neutral",
    "climate neutrality",
    "emission reduction",
    "energy efficiency",
    "renewable power",
    "hydropower",
    "tidal energy",
    "wave energy",
    "nuclear power",
    "nuclear energy",
    "carbon dioxide",
    "carbon market",
    "carbon markets",
    "coal-fired power",
    "lignite",
    "diesel emissions",
    "tailpipe emissions",
    "methane leak",
    "methane leaks",
    "pipeline project",
    "refinery",
    "refineries",
    "drilling permit",
    "drilling permits",
    "sustainable investing",
    "responsible investing",
    "climate disclosure",
    "transition plan",
    "green transition",
    "clean power",
]

FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "that",
    "with",
    "as",
    "by",
    "it",
    "is",
    "was",
    "were",
    "has",
    "have",
    "had",
    "said",
    "after",
    "from",
    "its",
    "at",
    "would",
    "will",
    "which",
    "they",
    "this",
    "but",
    "their",
    "been",
    "than",
    "into",
    "over",
    "amid",
}

CONTEXT_WINDOW = 160
PROSE_MIN = 0.55

SENTENCE_END_RE = re.compile(r"[a-z]{2,}\. ")


def _compile(words: list[str]) -> re.Pattern:
    """Build a case-insensitive, word-boundary alternation regex from the given
    words. Phrases and hyphens are escaped to match literally, word boundaries
    keep short tokens from matching inside larger words, and longer alternatives
    are listed first so the engine prefers the most specific match."""
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


STRONG_RE = _compile(STRONG_KEYWORDS)


def _cut_after_last(text: str, markers: list[str], zone: int) -> int:
    """Return the index just after the last marker found within the header zone,
    or 0 if no marker is present."""
    head = text[:zone].lower()
    best = 0
    for m in markers:
        i = head.rfind(m)
        if i != -1:
            best = max(best, i + len(m))
    return best


def clean_scraped_text(text: str) -> str:
    """Strip leading site chrome from flattened scraped article text so the real
    body survives truncation instead of being pushed past the model window by the
    header. Best-effort, applied in order: cut past the last nav/related-news
    marker in the header zone, then start the body after the first datestamp in
    that zone (typically the publish date, right before the body; preferred over
    a later updated-date or "related stories" teaser date), falling back to the
    first long prose sentence, then collapse whitespace. Returns "" for empty
    input.

    The proper fix is DOM-based extraction (trafilatura) on the raw HTML at scrape
    time; these heuristics are salvage for text that was already flattened and no
    longer has tags to separate menu from body. They reduce header pollution, they
    do not eliminate it.

    All leading cuts are bounded by MAX_HEAD_TRIM: a heuristic that would advance
    the start past that cap is treated as a false positive (more likely an inline
    ad marker or a teaser datestamp deep in the page than the end of the real
    header) and is ignored, so the lead paragraph is never trimmed away."""
    if not text:
        return ""

    t = text

    cut = _cut_after_last(t, NAV_CUT_MARKERS, HEADER_ZONE_CHARS)
    if 0 < cut <= MAX_HEAD_TRIM:
        t = t[cut:]

    stamps = [m.end() for m in DATESTAMP_RE.finditer(t[:HEADER_ZONE_CHARS])]
    first_stamp = next((s for s in stamps if s <= MAX_HEAD_TRIM), None)
    if first_stamp is not None:
        t = t[first_stamp:]
    else:
        m = PROSE_RE.search(t[:HEADER_ZONE_CHARS])
        if m and m.start() <= MAX_HEAD_TRIM:
            t = t[m.start() :]

    return re.sub(r"\s+", " ", t).strip()


def _prose_score(window: str) -> float:
    """Return a rough 0..1 measure of how prose-like a text window is. Menu and
    sidebar text is short Title-Case fragments with few lowercase function words
    and little sentence punctuation; article prose is the opposite. Combines the
    lowercase-token fraction, function-word presence, and comma/period density."""
    if not window.strip():
        return 0.0
    tokens = window.split()
    if len(tokens) < 8:
        return 0.0
    lower = sum(1 for t in tokens if t.isalpha() and t.islower())
    funcs = sum(1 for t in tokens if t.lower() in FUNCTION_WORDS)
    punct = window.count(",") + window.count(". ")
    return (
        (lower / len(tokens)) * 0.5 + min(funcs / 8, 1) * 0.4 + min(punct / 3, 1) * 0.1
    )


def _has_sentence_shape(window: str) -> bool:
    """Return True if the window looks like a clause rather than a list of section
    names. Requires at least two function words and a real sentence ending (a
    lowercase word followed by a period); a navigation bar satisfies neither."""
    tokens = window.split()
    funcs = sum(1 for t in tokens if t.lower() in FUNCTION_WORDS)
    return funcs >= 2 and bool(SENTENCE_END_RE.search(window))


def is_esg_keyword_match(text: str) -> bool:
    """Keep an article iff a STRONG ESG keyword appears in prose, not in chrome. A
    bare keyword hit is not enough: scraped pages embed section names like "Oil &
    gas industry" or "Electric Vehicles" in their navigation menus, which would
    otherwise pass any plain keyword filter. For each keyword hit the surrounding
    window is inspected and required to look like article prose (function words,
    sentence punctuation) rather than a capitalised list of menu links. This
    rejects the common false-positive classes while keeping genuine ESG coverage.
    Match on the cleaned, truncated text you actually send to the model."""
    if not text:
        return False
    for m in STRONG_RE.finditer(text):
        s, e = m.span()
        window = text[max(0, s - CONTEXT_WINDOW) : e + CONTEXT_WINDOW]
        if _prose_score(window) >= PROSE_MIN and _has_sentence_shape(window):
            return True
    return False
