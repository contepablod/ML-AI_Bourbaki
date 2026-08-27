import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def tmdb_request(
    path,
    api_key=None,
    bearer_token=None,
    params=None,
    timeout=20,
):
    if not api_key and not bearer_token:
        return None

    params = params or {}

    url = f"https://api.themoviedb.org/3{path}"
    headers = {"accept": "application/json"}

    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

        if params:
            url = url + "?" + urlencode(params)

    else:
        params = {
            **params,
            "api_key": api_key,
        }

        url = url + "?" + urlencode(params)

    request = Request(
        url,
        headers=headers,
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def runtime_to_bucket(runtime):
    if runtime is None or pd.isna(runtime):
        return pd.NA

    runtime = int(runtime)

    if runtime < 90:
        return "short"

    if runtime <= 150:
        return "medium"

    return "long"


def load_cache(cache_path):
    if not cache_path.exists():
        return {}

    try:
        text = cache_path.read_text(encoding="utf-8")

        if text.startswith("version https://git-lfs.github.com/spec/v1"):
            print("TMDb cache: Git LFS pointer detected.")
            return {}

        if not text.strip():
            return {}

        cache = json.loads(text)

        return cache if isinstance(cache, dict) else {}

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}


def save_cache(cache, cache_path):
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def enrich_with_tmdb(
    catalog_df,
    cache_path,
    api_key=None,
    bearer_token=None,
):
    cache = load_cache(cache_path)

    can_fetch_tmdb = bool(api_key or bearer_token)

    if cache:
        print(f"TMDb source: LOCAL CACHE " f"({len(cache)} entries)")

    elif can_fetch_tmdb:
        print("TMDb source: API")

    else:
        print("TMDb source: NONE")

    tmdb_ids = catalog_df.loc[
        catalog_df["tmdbId"].notna(),
        ["movieId", "tmdbId"],
    ].copy()

    if can_fetch_tmdb:

        missing_tmdb_ids = [
            str(int(tmdb_id))
            for tmdb_id in tmdb_ids["tmdbId"]
            if str(int(tmdb_id)) not in cache
        ]

        print(
            "TMDb entries missing from cache:",
            len(missing_tmdb_ids),
        )

        for tmdb_id in missing_tmdb_ids:

            try:
                payload = tmdb_request(
                    f"/movie/{tmdb_id}",
                    api_key=api_key,
                    bearer_token=bearer_token,
                    params={"append_to_response": "credits,keywords"},
                )

                cache[tmdb_id] = (
                    payload
                    if payload is not None
                    else {"_error": "TMDb request returned None"}
                )

            except Exception as exc:
                cache[tmdb_id] = {"_error": str(exc)}

        if missing_tmdb_ids:
            save_cache(
                cache,
                cache_path,
            )

    directors = []
    keywords = []
    cast_names = []
    writer_names = []
    countries = []
    original_languages = []
    runtime_minutes = []
    runtime_buckets = []

    for _, row in catalog_df.iterrows():

        tmdb_id = row.get("tmdbId")

        if pd.isna(tmdb_id):

            directors.append(pd.NA)
            keywords.append([])
            cast_names.append([])
            writer_names.append([])
            countries.append([])
            original_languages.append(pd.NA)
            runtime_minutes.append(pd.NA)
            runtime_buckets.append(pd.NA)

            continue

        payload = cache.get(
            str(int(tmdb_id)),
            {},
        )

        if not isinstance(payload, dict):
            payload = {}

        credits = payload.get(
            "credits",
            {},
        )

        if not isinstance(credits, dict):
            credits = {}

        crew = credits.get(
            "crew",
            [],
        )

        cast = credits.get(
            "cast",
            [],
        )

        if not isinstance(crew, list):
            crew = []

        if not isinstance(cast, list):
            cast = []

        director_names = [
            p.get("name")
            for p in crew
            if isinstance(p, dict) and p.get("job") == "Director" and p.get("name")
        ]

        writer_list = [
            p.get("name")
            for p in crew
            if isinstance(p, dict)
            and p.get("job")
            in {
                "Writer",
                "Screenplay",
            }
            and p.get("name")
        ]

        cast_list = [
            p.get("name") for p in cast[:8] if isinstance(p, dict) and p.get("name")
        ]

        country_list = [
            p.get("name")
            for p in payload.get(
                "production_countries",
                [],
            )
            if isinstance(p, dict) and p.get("name")
        ]

        keywords_payload = payload.get(
            "keywords",
            {},
        )

        if not isinstance(
            keywords_payload,
            dict,
        ):
            keywords_payload = {}

        raw_keywords = (
            keywords_payload.get("keywords") or keywords_payload.get("results") or []
        )

        directors.append(director_names[0] if director_names else pd.NA)

        keywords.append(
            [
                item.get("name")
                for item in raw_keywords
                if isinstance(item, dict) and item.get("name")
            ]
        )

        cast_names.append(sorted(set(cast_list)))

        writer_names.append(sorted(set(writer_list)))

        countries.append(sorted(set(country_list)))

        original_languages.append(
            payload.get(
                "original_language",
                pd.NA,
            )
        )

        runtime = payload.get(
            "runtime",
            pd.NA,
        )

        runtime_minutes.append(runtime)

        runtime_buckets.append(runtime_to_bucket(runtime))

    enriched = catalog_df.copy()

    enriched["director"] = directors
    enriched["tmdb_keywords"] = keywords
    enriched["cast_names"] = cast_names
    enriched["writer_names"] = writer_names
    enriched["countries"] = countries
    enriched["original_language"] = original_languages
    enriched["runtime_minutes"] = runtime_minutes
    enriched["runtime_bucket"] = runtime_buckets

    return enriched, bool(cache)
