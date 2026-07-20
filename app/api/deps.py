from __future__ import annotations
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.core.config import SETTINGS, get_logger

_log = get_logger("api.deps")

def new_session_id() -> str:
    return uuid.uuid4().hex


def save_upload_file(upload_file: UploadFile) -> Path:
    dest = SETTINGS.raw_docs_dir / upload_file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    _log.info("Saved upload '%s' -> %s", upload_file.filename, dest)
    return dest
