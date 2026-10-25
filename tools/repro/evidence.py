"""Local diagnostic checksum selection. Git publication uses publication.py instead."""
ALLOWED_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".html", ".png", ".svg", ".patch", ".md"}


def evidence_files(directory):
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and not p.is_symlink()
                  and "generated" not in p.relative_to(directory).parts
                  and p.suffix in ALLOWED_SUFFIXES and p.stat().st_size < 10 * 1024 * 1024)
