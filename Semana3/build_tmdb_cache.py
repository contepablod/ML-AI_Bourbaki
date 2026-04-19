import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


BASE_DIR = Path("Semana3") if Path("Semana3").exists() else Path(".")
DATA_DIR = BASE_DIR / "Data"
MOVIELENS_ZIP = DATA_DIR / "ml-latest-small.zip"
CACHE_PATH = DATA_DIR / "tmdb_movielens_cache.json"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


load_env_file(BASE_DIR / ".env")
load_env_file(Path(".env"))

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


def tmdb_request(path: str, params=None, timeout: int = 20):
    if not TMDB_API_KEY and not TMDB_BEARER_TOKEN:
        raise RuntimeError("Falta TMDB_API_KEY o TMDB_BEARER_TOKEN.")

    params = params or {}
    url = f"https://api.themoviedb.org/3{path}"
    headers = {"accept": "application/json"}

    if TMDB_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_BEARER_TOKEN}"
        if params:
            url = url + "?" + urlencode(params)
    else:
        params = {**params, "api_key": TMDB_API_KEY}
        url = url + "?" + urlencode(params)

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_movielens_catalog() -> pd.DataFrame:
    if not MOVIELENS_ZIP.exists():
        raise FileNotFoundError(f"No existe {MOVIELENS_ZIP}")

    with ZipFile(MOVIELENS_ZIP) as zf:
        root = "ml-latest-small/"
        movies = pd.read_csv(zf.open(root + "movies.csv"))
        ratings = pd.read_csv(zf.open(root + "ratings.csv"))
        links = pd.read_csv(zf.open(root + "links.csv"))

    agg = ratings.groupby("movieId").rating.agg(avg_rating="mean", rating_count="count").reset_index()
    catalog = movies.merge(agg, on="movieId", how="inner")
    catalog = catalog.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    catalog = catalog[catalog["tmdbId"].notna()].copy()
    catalog["tmdbId"] = catalog["tmdbId"].astype(int)
    catalog = catalog.sort_values(["rating_count", "avg_rating"], ascending=[False, False]).reset_index(drop=True)
    return catalog


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cache(limit=None, min_ratings=1, sleep_seconds=0.25, save_every=25, refresh_errors=False):
    catalog = load_movielens_catalog()
    catalog = catalog[catalog["rating_count"] >= min_ratings].copy()
    if limit is not None:
        catalog = catalog.head(limit).copy()

    cache = load_cache(CACHE_PATH)
    requested = 0
    skipped = 0
    failures = 0

    rows = catalog[["movieId", "title", "tmdbId", "rating_count"]].to_dict("records")
    total = len(rows)

    print(f"Peliculas candidatas: {total}")
    print(f"Cache actual: {len(cache)} entradas")
    print(f"Archivo cache: {CACHE_PATH}")

    for idx, row in enumerate(rows, start=1):
        tmdb_id = str(int(row["tmdbId"]))
        cached_payload = cache.get(tmdb_id)

        if cached_payload is not None and not (refresh_errors and "_error" in cached_payload):
            skipped += 1
            if idx % 200 == 0 or idx == total:
                print(f"[{idx}/{total}] skip tmdb_id={tmdb_id} title={row['title']}")
            continue

        try:
            payload = tmdb_request(
                f"/movie/{tmdb_id}",
                params={"append_to_response": "credits,keywords"},
            )
            cache[tmdb_id] = payload
            requested += 1
            print(f"[{idx}/{total}] ok   tmdb_id={tmdb_id} title={row['title']}")
        except Exception as exc:
            failures += 1
            cache[tmdb_id] = {"_error": str(exc)}
            print(f"[{idx}/{total}] fail tmdb_id={tmdb_id} title={row['title']} error={exc}")

        if requested % save_every == 0 and requested > 0:
            save_cache(cache, CACHE_PATH)
            print(f"Cache guardado parcialmente: {len(cache)} entradas")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    save_cache(cache, CACHE_PATH)
    print("Resumen final")
    print(f"- nuevas requests exitosas: {requested}")
    print(f"- entradas salteadas por cache: {skipped}")
    print(f"- requests con error: {failures}")
    print(f"- cache total guardado: {len(cache)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prearma el cache TMDb para la notebook de MovieLens.")
    parser.add_argument("--limit", type=int, default=None, help="Cantidad maxima de peliculas a consultar.")
    parser.add_argument("--min-ratings", type=int, default=1, help="Minimo de ratings para incluir una pelicula.")
    parser.add_argument("--sleep-seconds", type=float, default=0.25, help="Pausa entre requests a TMDb.")
    parser.add_argument("--save-every", type=int, default=25, help="Guardar cache cada N requests nuevas.")
    parser.add_argument(
        "--refresh-errors",
        action="store_true",
        help="Reintenta ids que ya estan en cache pero con _error.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_cache(
        limit=args.limit,
        min_ratings=args.min_ratings,
        sleep_seconds=args.sleep_seconds,
        save_every=args.save_every,
        refresh_errors=args.refresh_errors,
    )


if __name__ == "__main__":
    main()
