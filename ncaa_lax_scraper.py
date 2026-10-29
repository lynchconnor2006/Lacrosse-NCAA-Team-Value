"""
NCAA lacrosse team-level scraper (polite / cached / robots-aware).

Pulls team stat tables and ranking polls from ncaa.com, builds a team x stat
matrix, converts it to percentiles + z-scores for modeling, and keeps rankings
in a SEPARATE file for validation only (never as model inputs -> no circularity).

Usage:
    pip install requests pandas beautifulsoup4 lxml scipy
    python ncaa_lax_scraper.py --sport lacrosse-men --division d1 --season current

Before running:
  1. Read https://www.ncaa.com/terms-of-use (and stats.ncaa.org terms) and confirm
     your use (academic / club research) is permitted. Don't redistribute raw data.
  2. Set CONTACT below to a real email so site admins can reach you.
  3. Do a tiny test first:  --max-stats 1 --max-pages 1
"""

import argparse
import hashlib
import random
import re
import time
from io import StringIO
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CONTACT = "your_email@syr.edu"  # <-- CHANGE ME
USER_AGENT = f"SU-Lacrosse-Analytics-Club-Research/1.0 (academic; contact: {CONTACT})"
BASE = "https://www.ncaa.com"

MIN_DELAY = 4.0       # seconds between live requests
JITTER = 2.0          # + random 0..JITTER seconds
MAX_RETRIES = 3
TIMEOUT = 30

CACHE_DIR = Path("data/raw_cache")
OUT_DIR = Path("data/processed")

# Stats where a LOWER value is better (flipped before percentile/z-scoring).
# Extend after you see the real column names.
LOWER_IS_BETTER = ("turnover", "caused_by", "against", "allowed", "penalt", "foul")

# Ranking polls to pull as VALIDATION benchmarks (slugs as used in ncaa.com URLs).
# Verify the slugs by browsing https://www.ncaa.com/rankings/<sport>/<div>
RANKING_POLLS = ["iwlca-coaches", "inside-lacrosse", "rpi"]


# --------------------------------------------------------------------------- #
# Polite HTTP client
# --------------------------------------------------------------------------- #
class StopScraping(Exception):
    """Raised when the server tells us to back off (403/429). We do NOT push through."""


class PoliteFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.robots = robotparser.RobotFileParser()
        self.robots.set_url(urljoin(BASE, "/robots.txt"))
        try:
            self.robots.read()
        except Exception:
            # If robots.txt can't be read, fail closed rather than assume allowed.
            self.robots = None
        self._last_request = 0.0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def allowed(self, url: str) -> bool:
        return self.robots is not None and self.robots.can_fetch(USER_AGENT, url)

    def _cache_path(self, url: str) -> Path:
        return CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")

    def get(self, url: str, refresh: bool = False) -> str | None:
        if not self.allowed(url):
            print(f"[robots.txt] disallowed or unverifiable, skipping: {url}")
            return None

        cache = self._cache_path(url)
        if cache.exists() and not refresh:
            return cache.read_text(encoding="utf-8")

        for attempt in range(1, MAX_RETRIES + 1):
            wait = MIN_DELAY + random.uniform(0, JITTER) - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()

            resp = self.session.get(url, timeout=TIMEOUT)
            if resp.status_code in (403, 429):
                raise StopScraping(
                    f"Got {resp.status_code} from {url}. Stopping; wait and retry later "
                    "(or contact NCAA for data access)."
                )
            if resp.status_code == 404:
                return None
            if resp.status_code >= 500:
                time.sleep(2 ** attempt * 5)  # exponential backoff
                continue
            resp.raise_for_status()
            cache.write_text(resp.text, encoding="utf-8")
            return resp.text
        return None


# --------------------------------------------------------------------------- #
# Discovery + parsing
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def discover_team_stats(fetcher: PoliteFetcher, sport: str, division: str, season: str) -> dict:
    """
    Find team-stat pages from the site's own navigation instead of hard-coding IDs.
    Returns {stat_label: url}. Verify the output once by eye.
    """
    landing = f"{BASE}/stats/{sport}/{division}/{season}/team"
    html = fetcher.get(landing)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(rf"/stats/{sport}/{division}/[^/]+/team/(\d+)")
    found = {}
    for tag in soup.find_all(["a", "option"]):
        href = tag.get("href") or tag.get("value") or ""
        m = pattern.search(href)
        if m:
            label = tag.get_text(strip=True) or f"stat_{m.group(1)}"
            found[label] = urljoin(BASE, href.split("/p")[0])  # strip any page suffix
    return found


def parse_table(html: str) -> pd.DataFrame | None:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    for t in tables:
        if any(str(c).strip().lower() in ("team", "school") for c in t.columns):
            return t
    return None


def scrape_paginated(fetcher: PoliteFetcher, base_url: str, max_pages: int) -> pd.DataFrame:
    """ncaa.com paginates as <url>/p2, /p3, ... ; stop when a page is empty/repeats."""
    frames, prev_signature = [], None
    for page in range(1, max_pages + 1):
        url = base_url if page == 1 else f"{base_url}/p{page}"
        html = fetcher.get(url)
        if not html:
            break
        df = parse_table(html)
        if df is None or df.empty:
            break
        signature = tuple(df.iloc[:, 1].astype(str).head(3))
        if signature == prev_signature:  # site returned same page again
            break
        prev_signature = signature
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tidy_stat_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    team_col = next(c for c in df.columns if c.lower() in ("team", "school"))
    df = df.rename(columns={team_col: "team"})
    df["team"] = df["team"].astype(str).str.strip()
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("rank", "rnk", "#")], errors="ignore")
    keep_shared = {"team", "g", "gp", "cl", "conf", "conference"}
    renamed = {
        c: c if c.lower() in keep_shared else f"{slugify(label)}__{slugify(c)}"
        for c in df.columns
    }
    return df.rename(columns=renamed).drop_duplicates(subset="team").set_index("team")


def build_team_matrix(fetcher, sport, division, season, max_stats, max_pages) -> pd.DataFrame:
    stats = discover_team_stats(fetcher, sport, division, season)
    if not stats:
        raise RuntimeError("No team stat pages discovered; open the stats page in a browser and "
                           "adjust discover_team_stats() to match the markup.")
    print(f"Discovered {len(stats)} team stats")
    matrix = None
    for i, (label, url) in enumerate(stats.items()):
        if max_stats and i >= max_stats:
            break
        print(f"[{i+1}/{len(stats)}] {label}")
        raw = scrape_paginated(fetcher, url, max_pages)
        if raw.empty:
            continue
        tidy = tidy_stat_table(raw, label)
        shared = [c for c in tidy.columns if "__" not in c]
        if matrix is not None:  # avoid duplicate shared columns on join
            tidy = tidy.drop(columns=[c for c in shared if c in matrix.columns])
        matrix = tidy if matrix is None else matrix.join(tidy, how="outer")
    return matrix.reset_index()


# --------------------------------------------------------------------------- #
# Feature engineering (percentiles + z-scores; model learns the weights)
# --------------------------------------------------------------------------- #
def to_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "team":
            continue
        cleaned = out[c].astype(str).str.replace(r"[%,]", "", regex=True).str.strip()
        out[c] = pd.to_numeric(cleaned, errors="coerce")
    return out


def add_percentiles_and_z(df: pd.DataFrame) -> pd.DataFrame:
    num = to_numeric_stats(df)
    feats = pd.DataFrame({"team": num["team"]})
    stat_cols = [c for c in num.columns if "__" in c and num[c].notna().sum() >= 10]
    for c in stat_cols:
        s = num[c]
        if any(k in c for k in LOWER_IS_BETTER):
            s = -s  # flip so higher always = better
        feats[f"{c}__pctile"] = s.rank(pct=True) * 100
        std = s.std(ddof=0)
        feats[f"{c}__z"] = (s - s.mean()) / std if std else 0.0
    return feats


def add_derived_efficiency(num: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Derived efficiency stats. Fill `colmap` with the REAL column names after your first
    scrape (the NCAA doesn't publish every ratio directly), e.g.:
        {"goals": "scoring_offense__goals", "shots": "shots_per_game__total", ...}
    Any ratio whose inputs are missing is skipped.
    """
    d = num.copy()

    def ratio(name, a, b):
        if a in colmap and b in colmap and colmap[a] in d and colmap[b] in d:
            d[name] = d[colmap[a]] / d[colmap[b]].replace(0, pd.NA)

    ratio("shooting_pct", "goals", "shots")
    ratio("shots_per_possession", "shots", "possessions")   # possessions often must be estimated
    ratio("turnovers_per_possession", "turnovers", "possessions")
    ratio("clear_pct", "clears_made", "clear_attempts")
    ratio("ride_pct", "rides_made", "ride_attempts")         # = 1 - opp clear % if not published
    ratio("man_up_pct", "man_up_goals", "man_up_opps")
    ratio("man_down_pct", "man_down_stops", "man_down_opps")
    return d


# --------------------------------------------------------------------------- #
# Rankings (validation only)
# --------------------------------------------------------------------------- #
def scrape_rankings(fetcher, sport, division) -> pd.DataFrame:
    frames = []
    for poll in RANKING_POLLS:
        html = fetcher.get(f"{BASE}/rankings/{sport}/{division}/{poll}")
        df = parse_table(html) if html else None
        if df is None:
            print(f"  no table for poll '{poll}' (check the slug)")
            continue
        df = df.rename(columns=lambda c: str(c).strip())
        team_col = next(c for c in df.columns if c.lower() in ("team", "school"))
        rank_col = next((c for c in df.columns if c.lower() in ("rank", "rnk", "#")), df.columns[0])
        out = df[[team_col, rank_col]].rename(columns={team_col: "team", rank_col: f"rank_{poll}"})
        out["team"] = out["team"].astype(str).str.replace(r"\s*\(\d+\)$", "", regex=True).str.strip()
        frames.append(out.set_index("team"))
    return pd.concat(frames, axis=1).reset_index() if frames else pd.DataFrame()


def validate_against_rankings(scores: pd.DataFrame, rankings: pd.DataFrame, score_col: str):
    """Spearman between your model score and each external rank (negative = agreement)."""
    from scipy.stats import spearmanr
    merged = scores.merge(rankings, on="team", how="inner")
    for c in [c for c in rankings.columns if c.startswith("rank_")]:
        sub = merged[[score_col, c]].dropna()
        if len(sub) >= 10:
            rho, p = spearmanr(sub[score_col], sub[c])
            print(f"{c}: rho={rho:.3f} (n={len(sub)}, p={p:.3g})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="lacrosse-men", help="lacrosse-men | lacrosse-women")
    ap.add_argument("--division", default="d1")
    ap.add_argument("--season", default="current")
    ap.add_argument("--max-stats", type=int, default=0, help="0 = all (use 1 for a test run)")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--skip-rankings", action="store_true",
                    help="Rankings URLs are not season-specific, so skip them for historical seasons")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.sport}_{args.division}_{args.season}"
    fetcher = PoliteFetcher()

    try:
        matrix = build_team_matrix(fetcher, args.sport, args.division, args.season,
                                   args.max_stats, args.max_pages)
        matrix.to_csv(OUT_DIR / f"team_stats_raw_{tag}.csv", index=False)

        feats = add_percentiles_and_z(matrix)
        feats.to_csv(OUT_DIR / f"team_features_{tag}.csv", index=False)

        rankings = pd.DataFrame() if args.skip_rankings else scrape_rankings(
            fetcher, args.sport, args.division)
        if not rankings.empty:
            rankings.to_csv(OUT_DIR / f"rankings_VALIDATION_ONLY_{tag}.csv", index=False)
    except StopScraping as e:
        print(f"\nSTOPPED: {e}")

    print("Done. Raw HTML is cached in data/raw_cache, so re-runs make no new requests.")


if __name__ == "__main__":
    main()
