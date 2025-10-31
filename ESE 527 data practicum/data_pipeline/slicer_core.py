import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from tqdm import tqdm

# 静音
logging.getLogger("fitz").setLevel(logging.ERROR)

# ---------------- Utils ----------------

def md5(s: str) -> str:
    import hashlib as _hashlib
    return _hashlib.md5(s.encode("utf-8")).hexdigest()

def is_cjk(ch: str) -> bool:
    return any([
        "\u4e00" <= ch <= "\u9fff",
        "\u3400" <= ch <= "\u4dbf",
        "\u3040" <= ch <= "\u30ff",
        "\uac00" <= ch <= "\ud7af",
    ])

def normalize_space(s: str) -> str:
    s = s.replace("\u00A0", " ").replace("\u200b", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# ---------------- 版面重排 / 行修复 ----------------

def reflow_lines(lines: List[str]) -> str:
    """
    修复 PDF 抽取常见问题：
    - 连字符断词: finance-\nrelated -> finance related
    - 中文错误换行: “差收窄\n货币” -> “差收窄货币”
    - 英文小写/数字续行: “sector\nis” -> “sector is”
    - 保留段落边界：遇到项目符/编号/表图标题/章标题等不合并
    """
    out = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        if i == len(lines) - 1:
            out.append(cur)
            break

        nxt = lines[i+1].lstrip()

        # 强制断段：项目符/编号/表图/章标题
        if re.match(r"^([•·\-–—\*\u2022]|\(?\d+[\).、])\s+", nxt) or \
           re.match(r"^(表|图)\s*\d+", nxt) or \
           re.match(r"^第[一二三四五六七八九十百千]+章", nxt) or \
           re.fullmatch(r"[IVXLCDM]+[.)]?", nxt.strip()):
            out.append(cur); i += 1; continue

        # 1) 连字符断词
        if re.search(r"[A-Za-z]-\s*$", cur) and re.match(r"^[A-Za-z]", nxt):
            out.append(re.sub(r"-\s*$", "", cur) + nxt)
            i += 2; continue

        # 2) 中文行合并（上一行非句末标点，下一行中文开头）
        if (len(cur) > 0 and cur[-1] not in "。！？!?；;：:，,、)") and \
           (len(nxt) > 0 and is_cjk(nxt[0])):
            out.append(cur + nxt)
            i += 2; continue

        # 3) 英文/数字续行（上一行非句末标点，下一行小写/数字开头）
        if (len(cur) > 0 and cur[-1] not in ".!?;:)") and \
           re.match(r"^[a-z0-9]", nxt):
            out.append(cur + " " + nxt)
            i += 2; continue

        out.append(cur)
        i += 1

    return normalize_space("\n".join(out))

# ---------------- 页眉页脚 / 标题过滤 ----------------

_SENT_END = "。！？!?…"
_CH_NUM = "一二三四五六七八九十百千"
_ROMAN = "IVXLCDM"

def strip_repeating_headers_footers(pages_text: List[str]) -> List[str]:
    heads, tails = {}, {}
    for t in pages_text:
        ls = [x for x in t.splitlines() if x.strip()]
        if not ls: continue
        heads[ls[0]] = heads.get(ls[0], 0) + 1
        tails[ls[-1]] = tails.get(ls[-1], 0) + 1
    n = len(pages_text)
    head_cand = {k for k, v in heads.items() if v >= max(2, n // 3)}
    tail_cand = {k for k, v in tails.items() if v >= max(2, n // 3)}

    cleaned = []
    for t in pages_text:
        ls = t.splitlines()
        if ls and ls[0] in head_cand: ls = ls[1:]
        if ls and ls[-1] in tail_cand: ls = ls[:-1]
        cleaned.append("\n".join(ls))
    return cleaned

def is_heading_candidate(s: str) -> bool:
    t = s.strip()
    if not t: return False
    if t[-1] in _SENT_END: return False
    pats = [
        r"^\d+(\.\d+){0,3}\s*[、.)\-：:】)]?\s*\S{0,40}$",
        r"^[（(]?\d+[)）]\s*\S{0,40}$",
        rf"^[{_CH_NUM}]、\s*\S{{0,40}}$",
        r"^[（(][一二三四五六七八九十]+[)）]\s*\S{0,40}$",
        rf"^[{_CH_NUM}]{{1,3}}[、.]\s*\S{{0,40}}$",
        r"^(目录|前言|摘要|结论|附录|致谢|参考文献)\s*$",
        r"^(Abstract|Introduction|Conclusion|Contents|References|Appendix|Acknowledgements)\s*$",
        rf"^[{_ROMAN}]+[.)]\s*\S{{0,40}}$",
        r"^\S{1,40}[—\-－–—]{1,3}\S{1,40}$",
        r"^第[一二三四五六七八九十百千]+章\s*\S*$",
        r"^Page\s*\d+\s*/?\s*\d*\s*$",
        r"^(版权所有|保密|机密|©|Copyright).*$",
    ]
    for p in pats:
        if re.match(p, t, flags=re.IGNORECASE): return True
    if 2 <= len(t) <= 60:
        punct = sum(ch in "，。、；：:,.()/[]【】()—-－– " for ch in t)
        if punct / max(len(t), 1) < 0.02: return True
    letters = [ch for ch in t if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio >= 0.8 and 2 <= len(t) <= 60: return True
    return False

def drop_heading_lines(text: str) -> str:
    return "\n".join([ln for ln in text.splitlines() if ln.strip() and not is_heading_candidate(ln)])

# ---------------- 分句 & 2–3 句聚合 ----------------

_SENT_SPLIT_REGEX = re.compile(
    r"(?<=[。！？!?])\s+|"
    r"(?<=[。！？!?])(?=[“”\"'])|"
    r"(?<=[。！？!?])$"
)

def split_into_sentences(text: str) -> List[str]:
    t = text.strip()
    if not t: return []
    parts = _SENT_SPLIT_REGEX.split(t)
    sents = [s.strip() for s in parts if s and s.strip()]
    if len(sents) <= 1 and "\n" in t:
        sents = [p.strip() for p in t.split("\n") if p.strip()]
    return sents

def filter_heading_sents(sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        ss = s.strip()
        if not ss: continue
        if is_heading_candidate(ss): continue
        if len(ss) <= 3: continue
        out.append(ss)
    return out

def chunk_by_sentences(sentences: List[str], min_sent: int = 2, max_sent: int = 3) -> List[str]:
    if not sentences: return []
    min_sent = max(1, min_sent)
    max_sent = max(min_sent, max_sent)
    chunks, buf = [], []
    for s in sentences:
        buf.append(s)
        if len(buf) >= max_sent:
            chunks.append(" ".join(buf).strip())
            buf = []
    if buf:
        if len(buf) < min_sent and chunks:
            last = chunks.pop()
            chunks.append((last + " " + " ".join(buf)).strip())
        else:
            chunks.append(" ".join(buf).strip())
    return [c for c in chunks if len(c) > 5]

# ---------------- 读取 PDF → 清洗 → 切割 ----------------

def read_pdf_pages_text(pdf_path: str) -> List[str]:
    """用 PyMuPDF 读取每页文本，按块排序，再行级重排/修复。"""
    doc = fitz.open(pdf_path)
    pages_raw = []
    for p in doc:
        blocks = p.get_text("blocks") or []
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # by y, then x
        lines = []
        for b in blocks:
            txt = b[4] if len(b) > 4 else ""
            if not txt: continue
            for ln in txt.splitlines():
                ln = ln.rstrip()
                if ln: lines.append(ln)
            lines.append("")  # block gap
        pages_raw.append("\n".join(lines).strip())
    doc.close()

    pages_raw = strip_repeating_headers_footers(pages_raw)
    pages_clean = []
    for t in pages_raw:
        pages_clean.append(reflow_lines(t.splitlines()))
    return pages_clean

def process_pdf_to_chunks(pdf_path: str, out_dir: str, min_sent: int = 2, max_sent: int = 3) -> Tuple[int, int]:
    """
    处理单个 PDF，输出 out_dir/text_chunks.jsonl
    返回 (chunk_count, page_count)
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "text_chunks.jsonl"

    pages_text = read_pdf_pages_text(pdf_path)
    pages_text = [drop_heading_lines(t) for t in pages_text]

    n_chunks = 0
    doc_id = Path(pdf_path).stem
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for i, page_text in enumerate(pages_text, start=1):
            if not page_text.strip(): continue
            sents = split_into_sentences(page_text)
            sents = filter_heading_sents(sents)
            if not sents: continue
            chunks = chunk_by_sentences(sents, min_sent=min_sent, max_sent=max_sent)
            for j, ch in enumerate(chunks):
                rec = {
                    "id": md5(f"{doc_id}-p{i}-c{j}-{len(ch)}"),
                    "doc_id": doc_id,
                    "page": i,
                    "chunk_id": j,
                    "text": ch,
                    "source_path": str(Path(pdf_path).resolve())
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
    return n_chunks, len(pages_text)

def process_dir(pdf_dir: str, out_dir: str, min_sent: int = 2, max_sent: int = 3) -> int:
    """批量处理，返回总 chunk 数。"""
    pdfs = sorted([str(p) for p in Path(pdf_dir).glob("**/*.pdf")])
    total = 0
    for p in tqdm(pdfs, desc="Processing PDFs"):
        if not Path(p).is_file(): continue
        n_chunks, _ = process_pdf_to_chunks(p, out_dir, min_sent=min_sent, max_sent=max_sent)
        total += n_chunks
    return total