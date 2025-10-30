from pathlib import Path

import zipfile


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        raise SystemExit(
            "Kaggle package not installed or import failed. Run: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files("mnassrib/telecom-churn-datasets", path=str(raw_dir), unzip=True)

    # Remove any leftover zip files if present
    for f in raw_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(f, "r") as zf:
                zf.extractall(raw_dir)
        except zipfile.BadZipFile:
            pass
        f.unlink(missing_ok=True)

    for f in sorted(raw_dir.iterdir()):
        print(f.name)


if __name__ == "__main__":
    main()


