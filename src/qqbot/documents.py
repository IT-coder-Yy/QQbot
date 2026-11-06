"""本地文档提取与视觉渲染，不执行文档宏。"""
from __future__ import annotations

import asyncio
import base64
import io
import threading
import zipfile
from contextlib import closing
from pathlib import Path

from PIL import Image, ImageOps
import pypdfium2 as pdfium
from docx import Document
from docx.table import Table
from openpyxl import load_workbook

from .context.sqlite_store import StorageError, TOO_LARGE, size

PDF_LOCK = threading.Lock()  # PDFium 即使操作不同文件也不保证线程安全。
OFFICE = {".doc", ".docx", ".xls", ".xlsx"}
SUPPORTED = OFFICE | {".pdf", ".txt", ".md", ".csv"}


async def blocking(function, *args):
    # 取消等待不会停止工作线程；等它释放文件后再清理暂存目录。
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise
    except Exception as exc:
        if isinstance(exc, StorageError):
            raise
        raise StorageError("附件读取或解析失败，文件可能已加密、损坏或格式不兼容。") from exc


def image_part(source, box=None):
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if box is not None:
            x0, y0, x1, y1 = validate_box(box)
            image = image.crop((int(x0 * image.width), int(y0 * image.height), int(x1 * image.width), int(y1 * image.height)))
        image.thumbnail((2048, 2048))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=90)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()}}


def validate_box(box):
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise StorageError("裁剪区域必须是 0 到 1 之间的 [左,上,右,下]。")
    return box


def render_pdf(path, page_number, box=None):
    with PDF_LOCK, pdfium.PdfDocument(path) as pdf:
        if not 1 <= page_number <= len(pdf):
            raise StorageError(f"页码超出范围，共 {len(pdf)} 页。")
        with closing(pdf[page_number - 1]) as page:
            w, h = page.get_size()
            x0, y0, x1, y1 = validate_box(box) if box else (0, 0, 1, 1)
            scale = min(3, 2048 / max(w * (x1 - x0), h * (y1 - y0)))
            # PDFium crop 顺序是左、下、右、上，单位为 PDF canvas units。
            crop = (x0 * w, (1-y1) * h, (1-x1) * w, y0 * h)
            with closing(page.render(scale=scale, crop=crop)) as bitmap:
                out = io.BytesIO()
                bitmap.to_pil().convert("RGB").save(out, format="JPEG", quality=90)
            return image_part(io.BytesIO(out.getvalue())), len(pdf)


def extract(path: Path, name: str, limit: int):
    suffix = Path(name).suffix.lower()
    parts, used = [], 0

    def add(text, **metadata):
        nonlocal used
        item = {"number": len(parts) + 1, "text": text, **metadata}
        used += size(item)
        if used > limit:
            raise StorageError(TOO_LARGE)
        parts.append(item)

    if suffix in {".docx", ".xlsx"}:
        with zipfile.ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > max(limit * 4, 1_000_000):
                raise StorageError(TOO_LARGE)
    if suffix == ".pdf":
        with PDF_LOCK, pdfium.PdfDocument(path) as pdf:
            for index in range(len(pdf)):
                with closing(pdf[index]) as page, closing(page.get_textpage()) as textpage:
                    text = textpage.get_text_range()
                    has_images = any(True for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))
                    add(text, page=index + 1, needs_ocr=not text.strip() or has_images, method="text")
    elif suffix == ".docx":
        doc = Document(path)
        for index, block in enumerate(doc.iter_inner_content(), 1):
            text = "\n".join("\t".join(c.text for c in row.cells) for row in block.rows) if isinstance(block, Table) else block.text
            if text.strip():
                add(text, block=index)
    elif suffix == ".xlsx":
        book = load_workbook(path, read_only=True, data_only=False, keep_links=False)
        cached = load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            for sheet in book:
                if sheet.max_row * sheet.max_column > 5_000_000:
                    raise StorageError(TOO_LARGE)
                lines, start = [], 1
                for row_index, (row, values) in enumerate(zip(sheet.iter_rows(), cached[sheet.title].iter_rows()), 1):
                    cells = []
                    for cell, value in zip(row, values):
                        if cell.value is None:
                            continue
                        text = str(cell.value)
                        if cell.data_type == "f":
                            text += f" [缓存值={value.value}]" if value.value is not None else " [无缓存值]"
                        cells.append(f"{cell.coordinate}={text}")
                    if cells:
                        lines.append("; ".join(cells))
                    if row_index % 100 == 0:
                        if lines:
                            add("\n".join(lines), sheet=sheet.title, rows=[start, row_index])
                        lines, start = [], row_index + 1
                if lines:
                    add("\n".join(lines), sheet=sheet.title, rows=[start, row_index])
        finally:
            book.close()
            cached.close()
    elif suffix in {".txt", ".md", ".csv"}:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        for start in range(0, len(text), 12000):
            add(text[start:start+12000], character_start=start)
    else:
        raise StorageError(f"暂不支持 {suffix or '无扩展名'} 格式，请发送 PDF、Word、Excel 或文本文件。")
    return parts


async def convert_office(path: Path, name: str, target: str, directory: Path, *, binary: str, timeout: float, limit: int):
    """独立 profile 避免连接桌面 Office；产物只放本次暂存目录。"""
    source = directory / ("source" + Path(name).suffix.lower())
    source.write_bytes(path.read_bytes())
    profile = directory / "profile"
    profile.mkdir(exist_ok=True)
    (profile / "user").mkdir(exist_ok=True)
    (profile / "user" / "registrymodifications.xcu").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
<item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="DisableMacrosExecution" oor:op="fuse"><value>true</value></prop><prop oor:name="DisableActiveContent" oor:op="fuse"><value>true</value></prop></item>
<item oor:path="/org.openoffice.Office.Writer/Content/Update"><prop oor:name="Link" oor:op="fuse"><value>2</value></prop></item>
<item oor:path="/org.openoffice.Office.Calc/Content/Update"><prop oor:name="Link" oor:op="fuse"><value>1</value></prop></item>
</oor:items>''')
    excel = source.suffix.lower() in {".xls", ".xlsx"}
    output_filter = target
    if target == "pdf":
        output_filter = "pdf:" + ("calc_pdf_Export" if excel else "writer_pdf_Export")
        if excel:
            output_filter += ':{"SinglePageSheets":{"type":"boolean","value":"true"}}'
    elif target == "docx":
        output_filter = "docx:Office Open XML Text"
    elif target == "xlsx":
        output_filter = "xlsx:Calc Office Open XML"
    try:
        process = await asyncio.create_subprocess_exec(
            binary, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--norestore",
            "--convert-to", output_filter, "--outdir", str(directory), str(source),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise StorageError("缺少 LibreOffice，暂时无法转换旧版 Office 文件或查看文档图表。") from exc
    try:
        async with asyncio.timeout(timeout):
            await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    result = directory / ("source." + target)
    if process.returncode or not result.is_file():
        raise StorageError("文档转换失败，文件可能已加密、损坏或格式不兼容。")
    if result.stat().st_size > limit:
        raise StorageError(TOO_LARGE)
    return result
