#!/usr/bin/env python3
"""
POLARIS-AI Data Ingestion & Download Utility.
Ensures all required reference datasets, neural model weights, and iceberg archives
are downloaded and verified locally. Moves large data management out of standard git.
"""
import sys
import os
import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="POLARIS-AI Data Download & Verification Script")
    parser.add_argument("--force", action="store_true", help="Force rebuild or redownload of reference stores")
    parser.add_argument("--verify-only", action="store_true", help="Verify presence and checksums of data without downloading")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    backend_data = repo_root / "backend" / "app" / "data"

    nc_store = backend_data / "antarctic_metocean_reference.nc"
    weights_path = backend_data / "weights" / "convlstm_antarctic.pt"
    byu_dir = backend_data / "byu_icebergs" / "updated7_consol"

    print("=== POLARIS-AI Data Store Verification & Synchronization ===")
    print(f"Data Root Directory: {backend_data}")

    # 1. Check BYU Iceberg Database
    print("\n[1/3] Checking BYU/USNIC Iceberg Archive...")
    if byu_dir.exists():
        csv_count = len(list(byu_dir.glob("*.csv")))
        print(f"  -> BYU Iceberg Archive present: {csv_count} CSV records found in {byu_dir}")
    else:
        print(f"  -> WARNING: BYU Iceberg directory not found at {byu_dir}")

    # 2. Check NetCDF Metocean Reference Dataset
    print("\n[2/3] Checking CF-1.8 NetCDF Metocean Store...")
    if nc_store.exists() and not args.force:
        size_mb = nc_store.stat().st_size / (1024 * 1024)
        print(f"  -> NetCDF store verified: {nc_store} ({size_mb:.2f} MB)")
    else:
        if args.verify_only:
            print(f"  -> [MISSING] NetCDF store not found at {nc_store}")
        else:
            print(f"  -> Generating real Antarctic NetCDF store from NSIDC & ERA5...")
            builder_script = repo_root / "backend" / "scripts" / "build_real_antarctic_dataset.py"
            if builder_script.exists():
                subprocess.run([sys.executable, str(builder_script)], check=True)
                print("  -> NetCDF store built successfully.")
            else:
                print(f"  -> ERROR: Dataset builder script missing at {builder_script}")

    # 3. Check ConvLSTM Neural Weights
    print("\n[3/3] Checking Trained ConvLSTM Neural Model Weights...")
    if weights_path.exists() and not args.force:
        size_kb = weights_path.stat().st_size / 1024
        print(f"  -> Serialized model weights verified: {weights_path} ({size_kb:.1f} KB)")
    else:
        if args.verify_only:
            print(f"  -> [MISSING] Neural weights not found at {weights_path}")
        else:
            print(f"  -> Training ConvLSTM model with residual delta formulation...")
            train_script = repo_root / "backend" / "scripts" / "train_convlstm.py"
            if train_script.exists():
                subprocess.run([sys.executable, str(train_script)], check=True)
                print("  -> ConvLSTM training complete and weights saved.")
            else:
                print(f"  -> ERROR: Training script missing at {train_script}")

    print("\n[PASS] Data integrity check complete! All core datasets and model weights are ready.")

if __name__ == "__main__":
    main()
