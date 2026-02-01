from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
import webbrowser


KAGGLE_PAGE_URL = "https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature"
DEFAULT_FILENAME = "measures_v2.csv"


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def extract_zip(zip_path: Path, workdir: Path) -> None:
    info(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(workdir)


def find_file(root: Path, filename: str) -> Path | None:
    # Case-insensitive match
    filename_lower = filename.lower()
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() == filename_lower:
            return p
    return None


def move_into_place(found: Path, target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            info(f"Target already exists: {target}")
            info("Use --force to overwrite.")
            return
        info(f"Overwriting: {target}")
        target.unlink()

    info(f"Saving: {target}")
    shutil.move(str(found), str(target))
    info("Done.")


def main() -> None:
    """
    IMPORTANT LIMITATION
    --------------------
    Kaggle does NOT provide a stable public direct-download URL for datasets without authentication.
    Therefore, this script does NOT attempt to download the dataset automatically.

    Instead, it supports a clean workflow:
      1) Opens the Kaggle dataset page for manual download (one click in browser)
      2) You provide the downloaded ZIP path via --zip (or a CSV path via --csv)
      3) Script extracts/moves measures_v2.csv into data/raw/

    Usage examples:
      - Open page + instructions:
          python scripts/download_dataset.py

      - After you downloaded the ZIP:
          python scripts/download_dataset.py --zip "C:/Users/you/Downloads/electric-motor-temperature.zip"

      - If you already have the CSV:
          python scripts/download_dataset.py --csv "C:/path/to/measures_v2.csv"
    """
    parser = argparse.ArgumentParser(
        description="Prepare measures_v2.csv (Electric Motor Temperature dataset) WITHOUT Kaggle API."
    )
    parser.add_argument("--outdir", default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Expected CSV filename (default: measures_v2.csv)")
    parser.add_argument("--zip", dest="zip_path", default=None, help="Path to downloaded Kaggle ZIP file")
    parser.add_argument("--csv", dest="csv_path", default=None, help="Path to an existing measures_v2.csv")
    parser.add_argument("--open", action="store_true", help="Open Kaggle dataset page in your browser")
    parser.add_argument("--force", action="store_true", help="Overwrite if target file already exists")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    ensure_outdir(outdir)
    target = outdir / args.filename

    # If already exists, stop early unless --force
    if target.exists() and not args.force:
        info(f"Already present: {target}")
        info("Nothing to do.")
        return

    # If user provided a CSV directly
    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser().resolve()
        if not csv_path.exists():
            die(f"CSV path does not exist: {csv_path}")
        if csv_path.name.lower() != args.filename.lower():
            warn(f"CSV filename is '{csv_path.name}', expected '{args.filename}'. Proceeding anyway.")
        move_into_place(csv_path, target, force=args.force)
        return

    # If user provided a ZIP
    if args.zip_path:
        zip_path = Path(args.zip_path).expanduser().resolve()
        if not zip_path.exists():
            die(f"ZIP path does not exist: {zip_path}")
        if zip_path.suffix.lower() != ".zip":
            warn(f"File does not look like a ZIP: {zip_path}")

        # Extract into a temp folder under outdir/.tmp_extract (keeps it simple cross-platform)
        tmp_dir = outdir / ".tmp_extract"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        extract_zip(zip_path, tmp_dir)
        found = find_file(tmp_dir, args.filename)
        if not found:
            # Show some helpful context
            files = [p.name for p in tmp_dir.rglob("*") if p.is_file()]
            files_preview = "\n".join(sorted(files)[:60])
            die(
                f"Could not find '{args.filename}' inside extracted ZIP.\n"
                f"Files found (first 60):\n{files_preview}\n"
                "If Kaggle changed the package contents, pass --filename accordingly."
            )

        # Move the found CSV into final location
        move_into_place(found, target, force=args.force)

        # Clean up temp folder (optional, but nice)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # No inputs provided: open page + print instructions
    if args.open or True:
        info("Kaggle does not allow unauthenticated direct dataset downloads.")
        info("Manual download is required (one click).")
        info(f"Opening: {KAGGLE_PAGE_URL}")
        try:
            webbrowser.open(KAGGLE_PAGE_URL)
        except Exception:
            warn("Could not open a browser automatically. Please open the URL manually.")
        print("\nNext steps:\n")
        print("1) Click 'Download' on Kaggle (you may need to sign in).")
        print("2) After download finishes, run ONE of these:\n")
        print('   a) If you downloaded a ZIP:')
        print('      python scripts/download_dataset.py --zip "C:/path/to/downloaded.zip" --outdir data/raw\n')
        print('   b) If you already have measures_v2.csv:')
        print('      python scripts/download_dataset.py --csv "C:/path/to/measures_v2.csv" --outdir data/raw\n')
        print("Target expected at: data/raw/measures_v2.csv")
        return


if __name__ == "__main__":
    main()
