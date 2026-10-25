"""Download and extract the capstone's Kaggle datasets under ``data/``.

Set KAGGLE_API_TOKEN in the shell before running this script. The credential
is deliberately never stored in this source file or printed to the console.
"""

import os
from getpass import getpass
import sys

# Kaggle authenticates while its package is imported, so populate the expected
# environment variable first. Keep the value outside source control.
_kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
if not _kaggle_token:
    if sys.stdin.isatty():
        try:
            _kaggle_token = getpass("Kaggle API token (input hidden): ").strip()
        except (EOFError, OSError):
            _kaggle_token = ""
if not _kaggle_token:
    raise SystemExit("KAGGLE_API_TOKEN was not provided; no Kaggle package was imported.")
os.environ["KAGGLE_API_TOKEN"] = _kaggle_token
del _kaggle_token

import shutil
import stat
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
STAGING_ROOT = DATA_ROOT / ".kaggle_downloads"
DATASETS = (
    ("nih-chest-xrays/data", "nih_chest_xray"),
    ("salviohexia/isic-2019-skin-lesion-images-for-classification", "isic_2019"),
)


def _get_kaggle_api():
    """Return Kaggle's already-authenticated API client."""
    try:
        import kaggle
    except ImportError as exc:
        raise RuntimeError(
            "The Kaggle package is not installed. Install it with: python -m pip install kaggle"
        ) from exc

    # Current Kaggle releases authenticate the module-level client on import;
    # reusing it avoids authenticating a second time after token consumption.
    api = getattr(kaggle, "api", None)
    if api is not None:
        return api

    # Compatibility path for older releases without the module-level client.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> int:
    """Extract archive members while rejecting paths that escape destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    members = archive.infolist()
    extracted = 0

    for index, member in enumerate(members, start=1):
        relative_path = Path(member.filename)
        target = (destination / relative_path).resolve()
        try:
            inside_destination = os.path.commonpath((str(root), str(target))) == str(root)
        except ValueError:
            inside_destination = False
        if relative_path.is_absolute() or not inside_destination:
            raise ValueError(f"Unsafe archive path rejected: {member.filename!r}")

        # Do not create symbolic links from downloaded archive metadata.
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(f"Symbolic link in archive rejected: {member.filename!r}")

        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted += 1

        if index % 1000 == 0 or index == len(members):
            print(f"  Extracted {index:,}/{len(members):,} archive entries ({extracted:,} files)")

    return extracted


def download_and_extract(api, dataset_id: str, folder_name: str) -> None:
    """Download one dataset, extract it, then remove the downloaded ZIP."""
    destination = DATA_ROOT / folder_name
    staging = STAGING_ROOT / folder_name
    completion_marker = destination / ".kaggle_download_complete"
    if completion_marker.is_file():
        print(f"Already complete; leaving existing files in {destination} unchanged.")
        return

    staging.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {dataset_id} into {staging} ...")
    api.dataset_download_files(
        dataset_id,
        path=str(staging),
        unzip=False,
        quiet=False,
        force=True,
    )

    archives = sorted(staging.rglob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            f"Expected one downloaded ZIP for {dataset_id}, found {len(archives)} in {staging}"
        )
    zip_path = archives[0]
    print(f"Downloaded {zip_path.name} ({zip_path.stat().st_size / (1024 ** 3):.2f} GiB).")
    print(f"Extracting into {destination} ...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        file_count = _safe_extract(archive, destination)

    # Mark completion only after every archive member extracted successfully.
    completion_marker.write_text(dataset_id + "\n", encoding="utf-8")
    # Delete the archive only after every member was extracted successfully.
    zip_path.unlink()
    shutil.rmtree(staging, ignore_errors=True)
    try:
        STAGING_ROOT.rmdir()
    except OSError:
        pass

    print(f"Finished {dataset_id}: extracted {file_count:,} files to {destination}.")
    print(f"Removed downloaded archive {zip_path.name}.")


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        api = _get_kaggle_api()
    except Exception as exc:
        print(f"Kaggle setup failed: {exc}", file=sys.stderr)
        return 1

    failures = []
    for dataset_id, folder_name in DATASETS:
        try:
            download_and_extract(api, dataset_id, folder_name)
        except Exception as exc:
            failures.append((dataset_id, exc))
            print(f"ERROR downloading {dataset_id}: {exc}", file=sys.stderr)

    if failures:
        print("\nOne or more datasets did not complete:", file=sys.stderr)
        for dataset_id, exc in failures:
            print(f"  - {dataset_id}: {exc}", file=sys.stderr)
        print("Downloaded ZIP files are retained when extraction fails so you can retry.", file=sys.stderr)
        return 1

    print("\nAll requested Kaggle datasets are downloaded and extracted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
