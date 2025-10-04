# cmr_granules_and_links.py
"""
Buscar colecciones y granules via CMR (sin earthaccess).
Imprime enlaces útiles (data, opendap, browse) para cada granule.

Uso:
    python cmr_granules_and_links.py --short_name GPM_3IMERGDF \
        --start 2024-08-10 --end 2024-08-12 \
        --bbox -103.6 20.3 -103.1 20.8 --page_size 5
"""
import os
import sys
import argparse
import requests
import json

CMR_BASE = "https://cmr.earthdata.nasa.gov/search"


def search_granules(short_name, start=None, end=None, bbox=None, page_size=10):
    params = {"short_name": short_name, "page_size": page_size}
    if start and end:
        params["temporal"] = f"{start}T00:00:00Z,{end}T23:59:59Z"
    if bbox:
        params["bounding_box"] = ",".join(map(str, bbox))  # w,s,e,n
    url = CMR_BASE + "/granules.json"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def print_links_from_granules(json_resp, max_items=10):
    entries = json_resp.get("feed", {}).get("entry", [])
    for i, e in enumerate(entries[:max_items]):
        print(f"\n=== GRANULE {i+1} ===")
        print("title:", e.get("title"))
        print("id:", e.get("id"))
        found = False
        for link in e.get("links", []):
            href = link.get("href")
            rel = link.get("rel", "")
            if href and any(k in (rel + href) for k in ["data", "opendap", "download", ".nc", ".hdf", ".h5"]):
                print(f"  {rel} -> {href}")
                found = True
        if not found:
            print("  all links:")
            for link in e.get("links", []):
                print("   ", link.get("rel"), "->", link.get("href"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--short_name", required=True)
    p.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="Bounding box: west south east north")
    p.add_argument("--page_size", type=int, default=10)
    args = p.parse_args()

    print(f"Searching granules for {args.short_name} ...")
    j = search_granules(args.short_name, start=args.start, end=args.end, bbox=args.bbox, page_size=args.page_size)
    print("Found entries:", len(j.get("feed", {}).get("entry", [])))
    print_links_from_granules(j, max_items=args.page_size)


if __name__ == "__main__":
    main()
