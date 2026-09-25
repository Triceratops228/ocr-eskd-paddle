from pathlib import Path
import argparse
import json
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from tkinter import Tk, filedialog

import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

try:
    from tqdm.auto import tqdm
except ImportError:
    # Если tqdm не установлен, код всё равно будет работать, просто без красивой шкалы.
    def tqdm(iterable=None, total=None, desc=None, unit=None, leave=True, **kwargs):
        return iterable if iterable is not None else range(total or 0)


OUT_DIR_NAME = "eskd_paddle_result"
DPI = 400

import re
from rapidfuzz import process, fuzz


PHRASE_FIXES = {
    "Одозначение": "Обозначение",
    "Наименобание": "Наименование",
    "Приме- чание": "Примечание",
    "Приме-чание": "Примечание",

    "Сдорочный черпеж": "Сборочный чертеж",
    "Сдорочные единицы": "Сборочные единицы",
    "Сдорочные единицы": "Сборочные единицы",

    "Сиспема цдерживающая": "Система удерживающая",
    "Система цдерживающая": "Система удерживающая",
    "Сиспема удерживающая": "Система удерживающая",

    "Демпфер подвесной": "Демпфер подвесной",
    "Серьга-пряжка": "Серьга-пряжка",
    "Боковая подушка": "Боковая подушка",

    "Полотно трикотажное": "Полотно трикотажное",
    "полопна прикопажного": "полотна трикотажного",
    "полотна прикопажного": "полотна трикотажного",
    "полотно прикопажное": "полотно трикотажное",

    "сетки одъемной": "сетки объемной",
    "сетка одъемная": "сетка объемная",
    "с хорактеристиками": "с характеристиками",
    "предованиям": "требованиям",
    "ногрузка": "нагрузка",
    "должна дыть": "должна быть",
    "столдикам": "столбикам",
}


KNOWN_LINES = [
    "Формат",
    "Зона",
    "Поз.",
    "Обозначение",
    "Наименование",
    "Кол.",
    "Примечание",
    "Документация",
    "Сборочный чертеж",
    "Сборочные единицы",
    "Демпфер подвесной",
    "Система удерживающая",
    "Серьга-пряжка",
    "Прочие изделия",
    "Боковая подушка",
    "Полотно трикотажное (сетка объемная)",
    "ГОСТ 28554-2022",
    "Допускается применение",
    "полотна трикотажного",
    "сетки объемной",
    "с характеристиками",
    "аналогичными требованиям",
    "разрывная нагрузка",
    "должна быть не менее 80 Н",
    "поверхностная плотность",
    "должна быть не менее 180 г/м²",
    "Система фиксации",
]

TARGET_ROW_MIN = 2
TARGET_ROW_MAX = 27
TARGET_COL_MIN = 6
TARGET_COL_MAX = 8


def is_target_cell(cell):
    row = int(cell["row"])
    col = int(cell["col"])

    return (
        TARGET_ROW_MIN <= row <= TARGET_ROW_MAX
        and TARGET_COL_MIN <= col <= TARGET_COL_MAX
    )
def normalize_eskd_text(text: str) -> str:
    """
    Постобработка OCR под ЕСКД-спецификации.
    """
    if not text:
        return ""

    t = str(text).strip()

    # Убираем мусорные одиночные символы
    if len(t) == 1 and t in "|[]{}=_—`'.,:;":
        return ""

    # Нормализация пробелов
    t = re.sub(r"\s+", " ", t)

    # Частые ошибки в обозначении НМРБ
    t = re.sub(r"\b[HН][MМ][PР][5БBВ]\s*\.?", "НМРБ.", t)
    t = re.sub(r"\bИМР[5БBВ]\s*\.?", "НМРБ.", t)
    t = re.sub(r"\bНМР[5BВ]\s*\.?", "НМРБ.", t)

    # НМРБ..301524 -> НМРБ.301524
    t = re.sub(r"НМРБ\.+", "НМРБ.", t)

    # НМРБ. 301524. 038 -> НМРБ.301524.038
    t = re.sub(r"(НМРБ)\s*\.\s*", r"\1.", t)
    t = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", t)

    # НМРБ.301524.038CB -> НМРБ.301524.038 СБ
    t = re.sub(r"(НМРБ\.\d+(?:\.\d+)+)\s*[CС][BВБ]$", r"\1 СБ", t)

    # Частный случай из вашего листа
    t = t.replace("НМРБ.741358.063", "НМРБ.74.1358.063")

    # ГОСТ
    t = re.sub(r"\b[ГFТT][OО0][CСS][TТ]\b", "ГОСТ", t)
    t = t.replace("FОСТ", "ГОСТ")
    t = t.replace("ГОСГ", "ГОСТ")
    t = t.replace("Г0СТ", "ГОСТ")
    t = re.sub(r"ГОСТ\s*(\d+)\s*[-–—]\s*(\d+)", r"ГОСТ \1-\2", t)

    # Единицы
    t = re.sub(r"\bMM\b", "мм", t)
    t = re.sub(r"\bmM\b", "мм", t)
    t = re.sub(r"\bMМ\b", "мм", t)
    t = re.sub(r"(\d)\s*H\b", r"\1 Н", t)
    t = t.replace("г/м", "г/м²") if "180" in t and "г/м" in t and "²" not in t else t

    # AO "HПO..." -> АО "НПО..."
    t = re.sub(r"\bAO\b", "АО", t)
    t = re.sub(r"\bAО\b", "АО", t)
    t = re.sub(r"\bHПO\b", "НПО", t)
    t = re.sub(r"\bHПО\b", "НПО", t)

    # Прямые исправления фраз
    for wrong, right in PHRASE_FIXES.items():
        t = t.replace(wrong, right)

    # Частые буквенные ошибки
    word_fixes = {
        "цдерживающая": "удерживающая",
        "Сиспема": "Система",
        "Сдорочный": "Сборочный",
        "Сдорочные": "Сборочные",
        "черпеж": "чертеж",
        "и3д": "изд.",
        "Заум.": "Заим.",
        "Одозначение": "Обозначение",
        "Наименобание": "Наименование",
        "Допцскается": "Допускается",
        "полопна": "полотна",
        "прикопажного": "трикотажного",
        "хорактеристиками": "характеристиками",
        "предованиям": "требованиям",
        "ногрузка": "нагрузка",
        "одъемной": "объемной",
        "столдикам": "столбикам",
        "дыть": "быть",
        "докцм.": "докум.",
        "Подп.Дата": "Подп. Дата",
    }

    for wrong, right in word_fixes.items():
        t = t.replace(wrong, right)

    # Фаззи-исправление коротких строк, где OCR почти угадал строку
    if len(t) <= 60 and not re.search(r"\d{4,}", t):
        match = process.extractOne(t, KNOWN_LINES, scorer=fuzz.WRatio)
        if match:
            best, score, _ = match
            if score >= 82:
                t = best

    # Финальная чистка
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip("|[]{}=_—`'")

    return t
def render_pdf_to_png(pdf_path: str, out_dir: Path, dpi: int = 400) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    paths = []

    for i, page in tqdm(enumerate(doc, start=1), total=len(doc), desc="Рендер PDF", unit="стр."):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        path = out_dir / f"page_{i:03d}.png"
        pix.save(str(path))
        paths.append(path)

    return paths


def group_positions(indexes, max_gap=8):
    """
    Группирует близкие координаты линий в одну координату.
    """
    if len(indexes) == 0:
        return []

    groups = []
    current = [indexes[0]]

    for x in indexes[1:]:
        if x - current[-1] <= max_gap:
            current.append(x)
        else:
            groups.append(current)
            current = [x]

    groups.append(current)

    return [int(np.mean(g)) for g in groups]


def read_image(path: Path):
    """Читает изображение по Unicode-пути на Windows."""
    with path.open("rb") as image_file:
        encoded = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Не удалось прочитать изображение: {path}")
    return image


def write_image(path: Path, image) -> None:
    """Записывает изображение по Unicode-пути на Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Не удалось закодировать изображение: {path}")
    with path.open("wb") as image_file:
        image_file.write(encoded.tobytes())


def detect_table_grid(image_path: Path):
    """
    Находит вертикальные и горизонтальные линии таблицы.
    Возвращает координаты линий.
    """
    img = read_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Инвертированная бинаризация: линии/текст становятся белыми
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15,
    )

    h, w = binary.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(80, w // 30), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(80, h // 30))
    )

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_projection = horizontal.sum(axis=1)
    vertical_projection = vertical.sum(axis=0)

    y_indexes = np.where(horizontal_projection > horizontal_projection.max() * 0.35)[0]
    x_indexes = np.where(vertical_projection > vertical_projection.max() * 0.35)[0]

    y_lines = group_positions(y_indexes, max_gap=10)
    x_lines = group_positions(x_indexes, max_gap=10)

    return img, x_lines, y_lines


def crop_cells(img, x_lines, y_lines, min_w=40, min_h=25, pad=8):
    """
    Нарезает таблицу на ячейки.
    """
    cells = []
    h, w = img.shape[:2]

    for row in range(len(y_lines) - 1):
        for col in range(len(x_lines) - 1):
            x1, x2 = x_lines[col], x_lines[col + 1]
            y1, y2 = y_lines[row], y_lines[row + 1]

            if x2 - x1 < min_w or y2 - y1 < min_h:
                continue

            # Отступаем внутрь, чтобы не захватить линии таблицы
            xx1 = max(0, x1 + pad)
            yy1 = max(0, y1 + pad)
            xx2 = min(w, x2 - pad)
            yy2 = min(h, y2 - pad)

            if xx2 <= xx1 or yy2 <= yy1:
                continue

            crop = img[yy1:yy2, xx1:xx2]

            # Проверяем, есть ли внутри хоть какой-то тёмный текст
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            dark_ratio = np.mean(gray < 180)

            if dark_ratio < 0.005:
                continue

            cells.append({
                "row": row,
                "col": col,
                "x1": xx1,
                "y1": yy1,
                "x2": xx2,
                "y2": yy2,
                "image": crop,
            })

    return cells


def make_ocr():
    """
    PaddleOCR v3.
    Если будет ошибка по параметрам, см. комментарий ниже.
    """
    return PaddleOCR(
        lang="ru",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu",
    )


def parse_paddle_result(result):
    """
    Достаёт текст и confidence из результата PaddleOCR v3.
    """
    texts = []
    scores = []

    for res in result:
        data = None

        if isinstance(res, dict):
            data = res.get("res", res)
        elif hasattr(res, "json"):
            js = res.json
            if callable(js):
                js = js()
            if isinstance(js, dict):
                data = js.get("res", js)

        if not data:
            continue

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        for text, score in zip(rec_texts, rec_scores):
            text = str(text).strip()
            if text:
                texts.append(text)
                scores.append(float(score))

    return texts, scores


def prepare_crop_for_ocr(crop):
    """
    Подготовка отдельной ячейки для PaddleOCR.
    Важно: не удаляем агрессивно линии, а увеличиваем и улучшаем читаемость.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # Маленький текст увеличиваем сильнее
    if h < 60:
        scale = 4
    elif h < 120:
        scale = 3
    else:
        scale = 2

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Легкий контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Легкая резкость
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Белая рамка помогает OCR не резать крайние буквы
    sharp = cv2.copyMakeBorder(
        sharp,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return sharp


def ocr_cell(ocr, crop):
    prepared = prepare_crop_for_ocr(crop)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = Path(f.name)

    cv2.imwrite(str(temp_path), prepared)

    try:
        result = ocr.predict(str(temp_path))
        texts, scores = parse_paddle_result(result)
    finally:
        temp_path.unlink(missing_ok=True)

    if not texts:
        return "", 0.0

    text = " ".join(texts)
    score = max(scores) if scores else 0.0

    text = normalize_eskd_text(text)

    return text, score


def fix_eskd_text(text: str) -> str:
    """
    Постобработка под ЕСКД.
    Здесь можно добавлять свои замены.
    """
    text = text.strip()

    replacements = {
        "ИМРБ": "НМРБ",
        "ИМРВ": "НМРБ",
        "НМРВ": "НМРБ",
        "НМРЕ": "НМРБ",
        "HMРБ": "НМРБ",
        "HMPБ": "НМРБ",
        "ГОСГ": "ГОСТ",
        "Г0СТ": "ГОСТ",
        "ТОСТ": "ГОСТ",
        "ГOCT": "ГОСТ",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # НМРБ . 301524 . 038 -> НМРБ.301524.038
    text = re.sub(r"НМРБ\s*[\.,:;]?\s*", "НМРБ.", text)
    text = re.sub(r"(\d)\s*[\.,]\s*(\d)", r"\1.\2", text)

    # ГОСТ 28554 - 2022 -> ГОСТ 28554-2022
    text = re.sub(r"ГОСТ\s*(\d+)\s*[-–—]\s*(\d+)", r"ГОСТ \1-\2", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def draw_debug_grid(img, x_lines, y_lines, out_path: Path):
    debug = img.copy()

    for x in x_lines:
        cv2.line(debug, (x, 0), (x, debug.shape[0]), (0, 0, 255), 2)

    for y in y_lines:
        cv2.line(debug, (0, y), (debug.shape[1], y), (255, 0, 0), 2)

    write_image(out_path, debug)
def draw_debug_cells(img, cells, out_path):
    debug = img.copy()

    for cell in cells:
        x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
        row, col = cell["row"], cell["col"]

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            debug,
            f"{row},{col}",
            (x1 + 5, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    write_image(out_path, debug)

@dataclass
class Detail:
    assembly: str
    assembly_path: str
    pdf: str
    pdf_path: str
    page: int
    position: str
    decimal_number: str
    name: str
    quantity: str
    assembly_decimal: str = ""
    parent_assembly: str = ""
    parent_assembly_path: str = ""


def select_root_folder() -> Path:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title="Выберите корневую папку со спецификациями")
    root.destroy()
    if not selected:
        raise RuntimeError("Корневая папка не выбрана.")
    return Path(selected)


def is_decimal_number(text: str) -> bool:
    """Проверяет обозначение ЕСКД, не принимая обычное количество за обозначение."""
    value = re.sub(r"\s+", "", text.upper())
    return bool(
        re.fullmatch(r"[А-ЯA-Z0-9Ё-Я]{2,}(?:[.\-][А-ЯA-Z0-9Ё-Я]+)+", value)
        or re.fullmatch(r"[А-ЯA-ZЁ-Я]{2,}\.\d+(?:\.\d+)+", value)
    )


def is_position(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:[.,]\d+)?", text.strip()))


def decimal_from_text(text: str) -> str:
    match = re.search(r"[А-ЯA-ZЁ-Яа-яё]{2,}\.\d+(?:\.\d+)+", text)
    return match.group(0) if match else ""


def find_section(texts: list[str], section: str) -> bool:
    return any(section in normalize_eskd_text(text).casefold() for text in texts)


def row_cells(cells: list[dict], ocr) -> dict[int, list[dict]]:
    """Распознаёт целевой диапазон и группирует ячейки по строкам."""
    result = defaultdict(list)
    for cell in tqdm(cells, desc="OCR ячеек", unit="яч.", leave=False):
        if not is_target_cell(cell):
            continue
        text, score = ocr_cell(ocr, cell["image"])
        text = normalize_eskd_text(text)
        if text and (len(text) > 2 or text.isdigit()):
            result[cell["row"]].append({**cell, "text": text, "score": score})
    return result


def text_in_column(cells: list[dict], column: int) -> str:
    """Объединяет текст ячеек одного логического столбца."""
    values = [
        cell["text"] for cell in sorted(cells, key=lambda item: item["x1"])
        if cell["col"] == column
    ]
    return " ".join(values).strip()


def extract_details_from_page(row_map: dict[int, list[dict]], page: int,
                              assembly: str, assembly_path: str, pdf: str,
                              pdf_path: str) -> list[Detail]:
    rows = [row_map[row_number] for row_number in sorted(row_map)]
    current: Detail | None = None
    details = []
    in_details = False

    for cells in rows:
        decimal_number = text_in_column(cells, TARGET_COL_MIN)
        name = text_in_column(cells, TARGET_COL_MIN + 1)
        quantity = text_in_column(cells, TARGET_COL_MAX)

        if name.casefold().strip() == "детали":
            in_details = True
            continue
        if not in_details:
            continue

        if is_decimal_number(decimal_number):
            if current is not None:
                details.append(current)
            current = Detail(
                assembly=assembly,
                assembly_path=assembly_path,
                pdf=pdf,
                pdf_path=pdf_path,
                page=page,
                position="",
                decimal_number=decimal_number,
                name=name,
                quantity=quantity,
            )
        elif current is not None and name:
            current.name = f"{current.name} {name}".strip()

    if current is not None:
        details.append(current)
    return details


def process_pdf(pdf_path: Path, root_path: Path, output_dir: Path, ocr) -> list[Detail]:
    relative_assembly = pdf_path.parent.relative_to(root_path)
    assembly = pdf_path.parent.name
    pdf_output = output_dir / "documents" / relative_assembly / pdf_path.stem
    pages_dir = pdf_output / "pages"
    debug_dir = pdf_output / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Рендер PDF: {pdf_path}")
    page_paths = render_pdf_to_png(str(pdf_path), pages_dir, dpi=DPI)
    all_details = []

    for page_num, page_path in tqdm(
        enumerate(page_paths, start=1),
        total=len(page_paths),
        desc="Страницы",
        unit="стр."
    ):
        img, x_lines, y_lines = detect_table_grid(page_path)

        draw_debug_grid(
            img,
            x_lines,
            y_lines,
            debug_dir / f"page_{page_num:03d}_grid.png"
        )

        cells = crop_cells(img, x_lines, y_lines)

        tqdm.write(
            f"Страница {page_num}: найдено ячеек {len(cells)}"
        )
        draw_debug_cells(
            img,
            cells,
            debug_dir / f"page_{page_num:03d}_cells_numbered.png"
        )
        page_rows = row_cells(cells, ocr)
        all_details.extend(extract_details_from_page(
            page_rows,
            page_num,
            assembly,
            str(relative_assembly),
            pdf_path.name,
            str(pdf_path),
        ))
    return all_details


def choose_root_from_args() -> Path:
    parser = argparse.ArgumentParser(description="Распознавание структуры ЕСКД из папок и PDF.")
    parser.add_argument("root", nargs="?", help="Корневая папка со сборками")
    args = parser.parse_args()
    return Path(args.root) if args.root else select_root_folder()


def main():
    root_path = choose_root_from_args().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Папка не найдена: {root_path}")

    output_dir = root_path / OUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Загрузка PaddleOCR...")
    ocr = make_ocr()

    pdf_paths = sorted(
        path for path in root_path.rglob("*.pdf")
        if output_dir not in path.parents
    )
    if not pdf_paths:
        raise FileNotFoundError(f"В папке нет PDF-файлов: {root_path}")

    all_details = []
    for pdf_path in tqdm(pdf_paths, desc="PDF-файлы", unit="файл"):
        all_details.extend(process_pdf(pdf_path, root_path, output_dir, ocr))

    records = [asdict(detail) for detail in all_details]
    assembly_nodes = {
        str(folder.relative_to(root_path)): {
            "name": folder.name,
            "path": str(folder.relative_to(root_path)),
            "assembly_decimal": "",
            "parent_assembly": "",
            "parent_assembly_path": "",
            "included_in": None,
            "documents": [],
            "children": [],
        }
        for folder in [root_path, *root_path.rglob("*")]
        if folder.is_dir() and folder != output_dir and output_dir not in folder.parents
    }
    for pdf_path in pdf_paths:
        relative_folder = pdf_path.parent.relative_to(root_path)
        key = str(relative_folder)
        node = assembly_nodes[key]
        node["assembly_decimal"] = (
            node["assembly_decimal"]
            or decimal_from_text(pdf_path.stem)
            or decimal_from_text(node["name"])
        )
        assembly_nodes[key]["documents"].append({
            "name": pdf_path.name,
            "path": str(pdf_path.relative_to(root_path)),
            "details": [
                record for record in records
                if record["pdf_path"] == str(pdf_path)
            ],
        })

    # Связываем сборку с непосредственной родительской сборкой. Приоритет —
    # совпадение децимального обозначения дочернего PDF с деталью родителя,
    # затем совпадение названия папки с наименованием детали.
    for key, node in assembly_nodes.items():
        for record in records:
            if record["assembly_path"] == key:
                record["assembly_decimal"] = node["assembly_decimal"]
        if key == ".":
            continue
        parent_key = str(Path(key).parent)
        parent = assembly_nodes.get(parent_key)
        if parent is None:
            parent = assembly_nodes["."]
            parent_key = "."

        parent_records = [
            record for record in records
            if record["assembly_path"] == parent_key
        ]
        folder_name = node["name"].casefold()
        linked_record = next(
            (
                record for record in parent_records
                if node["assembly_decimal"]
                and record["decimal_number"] == node["assembly_decimal"]
            ),
            None,
        )
        if linked_record is None:
            linked_record = next(
                (
                    record for record in parent_records
                    if folder_name in record["name"].casefold()
                ),
                None,
            )

        node["parent_assembly"] = parent["name"]
        node["parent_assembly_path"] = parent_key
        if linked_record is not None:
            node["included_in"] = {
                "assembly": parent["name"],
                "assembly_path": parent_key,
                "decimal_number": linked_record["decimal_number"],
                "name": linked_record["name"],
                "quantity": linked_record["quantity"],
            }

        for record in records:
            if record["assembly_path"] == key:
                record["parent_assembly"] = parent["name"]
                record["parent_assembly_path"] = parent_key

    hierarchy = assembly_nodes["."]
    for key, node in sorted(assembly_nodes.items()):
        if key == ".":
            continue
        parent_key = str(Path(key).parent)
        parent = assembly_nodes.get(parent_key, hierarchy)
        parent["children"].append(node)

    (output_dir / "eskd_structure.json").write_text(
        json.dumps(hierarchy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(records, columns=list(Detail.__annotations__)).to_excel(
        output_dir / "eskd_details.xlsx", index=False
    )
    print(f"Готово. Структура: {output_dir / 'eskd_structure.json'}")
    print(f"Детали: {output_dir / 'eskd_details.xlsx'}")


if __name__ == "__main__":
    main()