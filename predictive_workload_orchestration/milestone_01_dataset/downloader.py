"""
milestone_01_dataset/downloader.py
====================================
Handles downloading and extracting the Alibaba Cluster Trace 2018 dataset.

Design notes:
- Downloads are skipped if the raw file already exists (idempotent).
- Archives are extracted in-place; original .tar.gz files are preserved.
- A download manifest is written to data/raw/download_manifest.json so the
  pipeline can verify completeness without re-downloading.
"""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional
from urllib.error import URLError

from config.logger import get_logger
from config.settings import DATASET_BASE_URL, DATASET_FILES, RAW_DIR

logger = get_logger(__name__)


class DatasetDownloader:
    """
    Downloads and extracts Alibaba Cluster Trace 2018 files.

    Attributes
    ----------
    base_url : str
        Root URL for all dataset files.
    raw_dir : Path
        Local directory where files are saved.
    files : Dict[str, str]
        Mapping of logical name → remote filename.
    """

    def __init__(
        self,
        base_url: str = DATASET_BASE_URL,
        raw_dir: Path = RAW_DIR,
        files: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.files = files if files is not None else DATASET_FILES
        self._manifest: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_all(self, force: bool = False) -> Dict[str, Path]:
        """
        Download all registered dataset files.

        Parameters
        ----------
        force : bool
            If True, re-download even if the file already exists.

        Returns
        -------
        Dict[str, Path]
            Mapping of logical name → local extracted CSV path.
        """
        extracted_paths: Dict[str, Path] = {}
        for logical_name, filename in self.files.items():
            try:
                local_path = self._download_file(filename, force=force)
                csv_path = self._extract_archive(local_path)
                extracted_paths[logical_name] = csv_path
                self._manifest[logical_name] = {
                    "filename": filename,
                    "archive": str(local_path),
                    "extracted": str(csv_path),
                    "md5": self._md5(local_path),
                }
            except (URLError, OSError) as exc:
                logger.error("Failed to download %s: %s", filename, exc)
                raise

        self._write_manifest()
        logger.info("All dataset files downloaded and extracted.")
        return extracted_paths

    def download_file(self, logical_name: str, force: bool = False) -> Path:
        """Download a single file by its logical name."""
        filename = self.files[logical_name]
        local_path = self._download_file(filename, force=force)
        return self._extract_archive(local_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_file(self, filename: str, force: bool = False) -> Path:
        local_path = self.raw_dir / filename
        if local_path.exists() and not force:
            logger.info("Already exists, skipping download: %s", local_path.name)
            return local_path

        url = self.base_url + filename
        logger.info("Downloading %s → %s", url, local_path)
        urllib.request.urlretrieve(url, local_path, reporthook=self._progress_hook(filename))
        logger.info("Downloaded: %s (%.2f MB)", filename, local_path.stat().st_size / 1e6)
        return local_path

    def _extract_archive(self, archive_path: Path) -> Path:
        """
        Extract a .tar.gz archive into the same directory.

        Returns the path to the first .csv file found inside the archive.
        """
        if not tarfile.is_tarfile(archive_path):
            # Already extracted or not an archive — look for matching CSV
            csv_path = archive_path.with_suffix(".csv")
            if csv_path.exists():
                return csv_path
            logger.warning("%s is not a tar archive and no .csv found.", archive_path.name)
            return archive_path

        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
            csv_members = [m for m in members if m.endswith(".csv")]
            logger.info("Extracting %s (contains: %s)", archive_path.name, ", ".join(members))
            tar.extractall(path=self.raw_dir)

        if csv_members:
            extracted_csv = self.raw_dir / csv_members[0]
            logger.info("Extracted CSV: %s", extracted_csv.name)
            return extracted_csv

        # Fall back to the raw archive path if no CSV found
        return archive_path

    @staticmethod
    def _progress_hook(filename: str):
        """Return a urllib reporthook that logs download progress."""
        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                downloaded = block_num * block_size
                pct = min(100, downloaded * 100 / total_size)
                if block_num % 500 == 0:
                    logger.info("  %s: %.1f%%", filename, pct)
        return hook

    @staticmethod
    def _md5(path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hasher = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _write_manifest(self) -> None:
        manifest_path = self.raw_dir / "download_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2)
        logger.info("Download manifest written: %s", manifest_path)
