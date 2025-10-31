import argparse
from pathlib import Path
from slicer_core import process_pdf_to_chunks, process_dir

def build_parser():
    ap = argparse.ArgumentParser(
        description="读取PDF并按句切割（2~3句聚合），修复换行/断词，去页眉页脚/标题。"
    )
    ap.add_argument("--pdf", help="单个 PDF 文件路径")
    ap.add_argument("--pdf-dir", help="PDF 文件夹（批量）")
    ap.add_argument("--out-dir", default="./sliced_output", help="输出目录")
    ap.add_argument("--min-sentences-per-chunk", type=int, default=2, help="每段最少句子数")
    ap.add_argument("--max-sentences-per-chunk", type=int, default=3, help="每段最多句子数")
    ap.add_argument("--clean-out", action="store_true", help="运行前清空 text_chunks.jsonl")
    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    if not args.pdf and not args.pdf_dir:
        ap.error("必须提供 --pdf 或 --pdf-dir")

    out = Path(args.out-dir if hasattr(args, "out-dir") else args.out_dir)  # 兼容性保护
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "text_chunks.jsonl"
    if args.clean_out and jsonl.exists():
        jsonl.unlink()

    total = 0
    if args.pdf:
        n, pages = process_pdf_to_chunks(
            args.pdf, out_dir=args.out_dir,
            min_sent=args.min_sentences_per_chunk,
            max_sent=args.max_sentences_per_chunk
        )
        print(f"[OK] {args.pdf} pages={pages} chunks={n}")
        total += n
    if args.pdf_dir:
        total += process_dir(
            args.pdf_dir, out_dir=args.out_dir,
            min_sent=args.min_sentences_per_chunk,
            max_sent=args.max_sentences_per_chunk
        )

    print(f"[DONE] total_chunks={total}")
    print(f"[OUT]  {jsonl}")

if __name__ == "__main__":
    main()