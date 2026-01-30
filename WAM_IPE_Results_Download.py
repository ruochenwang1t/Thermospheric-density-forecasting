import os
import re
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.config import Config


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def atomic_move(tmp_path: str, final_path: str):
    os.replace(tmp_path, final_path)


def iter_s3_objects(bucket: str, prefix: str, s3_client):
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            yield obj


def download_wam_ipe_range(
    start_date: str,
    end_date: str,
    include_end_date: bool = True,
    bucket_name: str = "noaa-nws-wam-ipe-pds",
    version_prefix: str = "v1.2/",
    hours=("00", "06", "12", "18"),
    data_root: str = "data_root",
):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if not include_end_date:
        end = end - timedelta(seconds=1)

    ts_pat = re.compile(r"(\d{8}_\d{6})\.nc$")

    ensure_dir(data_root)
    total_found = 0
    total_skipped = 0
    total_downloaded = 0

    cur = start
    while cur <= end:
        date_str = cur.strftime("%Y%m%d")
        for hh in hours:
            folder = f"wfs.{date_str}/{hh}/"
            s3_prefix = f"{version_prefix}{folder}"
            local_dir = os.path.join(data_root, version_prefix.rstrip("/"), folder)
            ensure_dir(local_dir)

            for obj in iter_s3_objects(bucket_name, s3_prefix, s3):
                key = obj["Key"]
                if not key.endswith(".nc"):
                    continue

                m = ts_pat.search(key)
                if not m:
                    continue

                try:
                    file_dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                if not (start <= file_dt <= end):
                    continue

                total_found += 1
                local_name = os.path.basename(key)
                local_path = os.path.join(local_dir, local_name)
                size_remote = obj.get("Size", 0)

                if os.path.exists(local_path) and os.path.getsize(local_path) == size_remote:
                    total_skipped += 1
                    print(f"[SKIP] {local_path} (exists, size match)")
                    continue

                tmp_path = local_path + ".part"
                print(f"[GET ] s3://{bucket_name}/{key}")
                s3.download_file(bucket_name, key, tmp_path)
                atomic_move(tmp_path, local_path)
                total_downloaded += 1
                print(f"[SAVE] {local_path}")

        cur += timedelta(days=1)

    print("\nDone.")
    print(f"  Found in range : {total_found}")
    print(f"  Downloaded     : {total_downloaded}")
    print(f"  Skipped (exist): {total_skipped}")


if __name__ == "__main__":
    download_wam_ipe_range(
        start_date="2024-05-24",
        end_date="2024-05-31",
        include_end_date=True,
        bucket_name="noaa-nws-wam-ipe-pds",
        version_prefix="v1.2/",
        hours=("00", "06", "12", "18"),
        data_root="data_root",
    )
