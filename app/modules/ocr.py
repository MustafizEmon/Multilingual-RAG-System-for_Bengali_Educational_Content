from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
import json

from pdf2image import convert_from_path
from tqdm.auto import tqdm
import cv2
import numpy as np
import pytesseract

from app.core.config import SETTINGS, get_logger
from app.modules.utils import clean_page_text, free_memory

_log = get_logger("ocr")


@dataclass
class PageRecord:
    """Minimal metadata + cleaned text for a single OCR'd page."""
    document_name: str
    page_number: int
    text: str
    char_count: int
    ocr_lang: str


def _deskew(image: np.ndarray) -> np.ndarray:
    try:
        inverted = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(inverted > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 0.1:
            return image  # not worth the interpolation cost
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_matrix, (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as exc:
        _log.warning("Deskew failed, using original image: %s", exc)
        return image


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def _threshold(image: np.ndarray) -> np.ndarray:   
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_page_image(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = _deskew(gray)
    gray = _denoise(gray)
    return _threshold(gray)


def ocr_single_page(image_rgb: np.ndarray, lang: str = SETTINGS.ocr_lang) -> str:
    processed = preprocess_page_image(image_rgb)
    text = pytesseract.image_to_string(processed, lang=lang, config="--psm 4")
    free_memory(processed)
    return text


def iter_pdf_pages(pdf_path: Path, dpi: int = SETTINGS.ocr_dpi) -> Iterator[tuple[int, np.ndarray]]:
    from pdf2image.pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    n_pages = info["Pages"]

    for page_num in range(1, n_pages + 1):
        images = convert_from_path(
            str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num
        )
        page_image = np.array(images[0].convert("RGB"))
        free_memory(images)  # drop the PIL image list immediately after conversion
        yield page_num, page_image
        free_memory(page_image)


def dump_document_raw_text(document_name: str, records: list[PageRecord]) -> Path:
    SETTINGS.raw_text_dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = SETTINGS.raw_text_dump_dir / f"{document_name}.txt"

    parts = []
    for rec in sorted(records, key=lambda r: r.page_number):
        parts.append(f"----- Page {rec.page_number} -----\n{rec.text}")
    out_path.write_text("\n\n".join(parts), encoding="utf-8")

    _log.info("Wrote raw pre-chunking text dump for '%s' -> %s", document_name, out_path)
    return out_path


def ingest_pdf(pdf_path: Path, document_name: str | None = None) -> list[PageRecord]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    document_name = document_name or pdf_path.stem

    records: list[PageRecord] = []
    out_dir = SETTINGS.page_text_dir / document_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_num, page_image in tqdm(iter_pdf_pages(pdf_path), desc=f"OCR {document_name}"):
        try:
            raw_text = ocr_single_page(page_image, lang=SETTINGS.ocr_lang)
            cleaned = clean_page_text(raw_text)
        except Exception as exc:
            _log.error("OCR failed on %s page %d: %s", document_name, page_num, exc)
            continue

        record = PageRecord(
            document_name=document_name,
            page_number=page_num,
            text=cleaned,
            char_count=len(cleaned),
            ocr_lang=SETTINGS.ocr_lang,
        )
        records.append(record)

        # Persist immediately; free the (small) text-only structures are cheap,
        # but we still avoid accumulating raw_text/page_image beyond this loop body.
        page_file = out_dir / f"page_{page_num:04d}.json"
        page_file.write_text(json.dumps(asdict(record), ensure_ascii=False), encoding="utf-8")
        free_memory(raw_text)

    _log.info("Ingested %d pages for document '%s' -> %s", len(records), document_name, out_dir)
    if records:
        dump_document_raw_text(document_name, records)
    return records


def ingest_image_file(image_path: Path, document_name: str, page_number: int = 1) -> PageRecord:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise cv2.error(f"Could not decode image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    free_memory(bgr)

    raw_text = ocr_single_page(rgb, lang=SETTINGS.ocr_lang)
    cleaned = clean_page_text(raw_text)
    free_memory(rgb, raw_text)

    record = PageRecord(
        document_name=document_name,
        page_number=page_number,
        text=cleaned,
        char_count=len(cleaned),
        ocr_lang=SETTINGS.ocr_lang,
    )

    out_dir = SETTINGS.page_text_dir / document_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"page_{page_number:04d}.json").write_text(
        json.dumps(asdict(record), ensure_ascii=False), encoding="utf-8"
    )
    return record
