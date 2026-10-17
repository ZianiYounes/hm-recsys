"""Telechargement du dataset Kaggle H&M.

Prerequis :
  1. Authentification Kaggle, au choix :
     - `kaggle auth login` (connexion via le navigateur, le plus simple)
     - ou token de kaggle.com/settings/api dans la variable KAGGLE_API_TOKEN
     - ou ancien kaggle.json dans %USERPROFILE%\\.kaggle\\
  2. Accepter les regles de la competition sur sa page Kaggle ("Join" / "Late Submission")
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from loguru import logger


def _get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # l'import du paquet kaggle peut deja tenter l'authentification
        raise RuntimeError(
            "Impossible d'initialiser l'API Kaggle. Lance `kaggle auth login` puis accepte "
            "les regles de la competition sur sa page Kaggle."
        ) from e
    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_all(raw_dir: Path) -> None:
    for zpath in sorted(raw_dir.glob("*.zip")):
        logger.info(f"Decompression de {zpath.name} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(raw_dir)
        zpath.unlink()
        logger.info(f"{zpath.name} decompresse puis supprime")


def download_dataset(
    competition: str,
    raw_dir: Path,
    files: list[str],
    with_images: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = _get_api()

    if with_images:
        # Archive complete (~30 Go) : CSV + dossier images/
        logger.info("Telechargement complet (CSV + images), compte 30 Go et un peu de patience")
        api.competition_download_files(competition, path=str(raw_dir), quiet=False)
    else:
        for name in files:
            if (raw_dir / name).exists():
                logger.info(f"{name} deja present, ignore")
                continue
            logger.info(f"Telechargement de {name}")
            api.competition_download_file(competition, name, path=str(raw_dir), quiet=False)

    _unzip_all(raw_dir)

    missing = [f for f in files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Fichiers manquants apres telechargement : {missing}")
    logger.success("Donnees brutes pretes dans " + str(raw_dir))
