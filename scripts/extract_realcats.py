#!/usr/bin/env python3
"""Extract only labeled-address normal_transactions.json files from real-cats.zip.

Usage:
    python scripts/extract_realcats.py

Reads CE.tsv / BE.tsv from data/raw/realcats_dataset/ to determine which
addresses are labeled, then extracts only their normal_transactions.json from
data/raw/realcats_transactions/real-cats.zip into
data/raw/realcats_transactions/ETHdata/ETHdata/.

Already-extracted addresses are skipped. After extraction the zip can be
deleted to reclaim disk space (prompted).
"""

import os
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR    = os.path.join(PROJECT_ROOT, "data", "raw", "realcats_dataset")
ZIP_PATH     = os.path.join(PROJECT_ROOT, "data", "raw", "realcats_transactions", "real-cats.zip")
EXTRACT_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "realcats_transactions")
ETH_DIR      = os.path.join(EXTRACT_ROOT, "ETHdata", "ETHdata")

TX_FILE = "normal_transactions.json"   # only file we need for the motif pipeline


def load_labeled_addresses() -> set[str]:
    labeled = set()
    for tsv in ("CE.tsv", "BE.tsv"):
        path = os.path.join(LABEL_DIR, tsv)
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found — is the GitHub Real-CATS release in {LABEL_DIR}?")
        with open(path) as f:
            header = f.readline().strip().split("\t")
            addr_col = header.index("address")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > addr_col:
                    labeled.add(parts[addr_col].lower())
    return labeled


def already_extracted(labeled: set[str]) -> set[str]:
    if not os.path.isdir(ETH_DIR):
        return set()
    extracted = set()
    for d in os.listdir(ETH_DIR):
        addr = d.lower()
        if addr in labeled:
            tx_path = os.path.join(ETH_DIR, d, TX_FILE)
            if os.path.exists(tx_path):
                extracted.add(addr)
    return extracted


def main() -> None:
    if not os.path.exists(ZIP_PATH):
        sys.exit(f"ERROR: zip not found at {ZIP_PATH}")

    print("Loading labeled addresses from CE.tsv / BE.tsv ...")
    labeled = load_labeled_addresses()
    print(f"  {len(labeled):,} labeled Ethereum addresses")

    done = already_extracted(labeled)
    missing = {a for a in labeled if a not in done}
    print(f"  {len(done):,} already have {TX_FILE}")
    print(f"  {len(missing):,} need extraction")

    if not missing:
        print("Nothing to do — all labeled addresses already extracted.")
        _offer_zip_delete()
        return

    print(f"\nOpening {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # Build a lookup: lowercase address → exact zip entry name
        # Zip paths look like:  ETHdata/ETHdata/<addr>/normal_transactions.json
        print("  Indexing zip entries ...")
        target_entries: dict[str, str] = {}
        for name in zf.namelist():
            parts = name.split("/")
            # expect: ETHdata / ETHdata / <addr> / <file>
            if len(parts) == 4 and parts[3] == TX_FILE:
                addr = parts[2].lower()
                if addr in missing:
                    target_entries[addr] = name

        found    = set(target_entries)
        not_found = missing - found
        if not_found:
            print(f"  WARNING: {len(not_found):,} labeled addresses have no entry in zip "
                  f"(may have zero transactions).")

        print(f"  Extracting {len(found):,} entries ...")
        os.makedirs(ETH_DIR, exist_ok=True)

        for i, (addr, zip_entry) in enumerate(target_entries.items(), 1):
            dest_dir = os.path.join(ETH_DIR, zip_entry.split("/")[2])
            os.makedirs(dest_dir, exist_ok=True)
            dest_file = os.path.join(dest_dir, TX_FILE)
            with zf.open(zip_entry) as src, open(dest_file, "wb") as dst:
                dst.write(src.read())
            if i % 500 == 0 or i == len(found):
                print(f"  {i:,}/{len(found):,} extracted", end="\r", flush=True)

    print(f"\nDone. {len(found):,} {TX_FILE} files extracted to {ETH_DIR}")

    _offer_zip_delete()


def _offer_zip_delete() -> None:
    size_gb = os.path.getsize(ZIP_PATH) / 1e9
    answer = input(f"\nDelete {ZIP_PATH} ({size_gb:.1f} GB) to free disk space? [y/N] ").strip().lower()
    if answer == "y":
        os.remove(ZIP_PATH)
        print("Zip deleted.")
    else:
        print("Zip kept.")


if __name__ == "__main__":
    main()
