"""
tools/make_manifest.py  -  refresh tools/FILES.sha256 after you change a code file (for maintainers).

tools/check_install.py compares every file listed in tools/FILES.sha256 with its checksum, so that a user notices an
incomplete or mixed-up download. Documentation (README, docs/, results/, models/) is not listed and can be edited
freely. After changing a listed file, or to list a new code file:

    python tools/make_manifest.py                      update the checksums of the files already listed
    python tools/make_manifest.py --add tools/new.py   also list new files (paths relative to the project folder)
    python tools/make_manifest.py --check              only report what is out of date (exit code 1 if anything is)
"""
import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "tools", "FILES.sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    entries = []
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, rel = line.rstrip("\n").split("  ", 1)
                entries.append([rel, digest])
    return entries


def main():
    ap = argparse.ArgumentParser(description="Refresh the checksums in tools/FILES.sha256.")
    ap.add_argument("--add", nargs="*", default=[], help="new files to list (relative to the project folder)")
    ap.add_argument("--check", action="store_true", help="only report what is out of date")
    args = ap.parse_args()

    entries = read_manifest()
    listed = {rel for rel, _ in entries}
    for path in args.add:
        rel = os.path.relpath(os.path.abspath(path), HERE).replace(os.sep, "/")
        if rel.startswith("../") or not os.path.isfile(os.path.join(HERE, rel)):
            raise SystemExit(f"--add {path}: not a file inside the project folder")
        if rel not in listed:
            entries.append([rel, ""])
            listed.add(rel)

    changed, missing = [], []
    for entry in entries:
        rel, old = entry
        path = os.path.join(HERE, *rel.split("/"))
        if not os.path.isfile(path):
            missing.append(rel)
            continue
        new = sha256(path)
        if new != old:
            changed.append(rel)
            entry[1] = new

    if missing:
        print("listed but missing (remove the line from tools/FILES.sha256 if the file was deleted on purpose):")
        print("  " + "\n  ".join(missing))
    if args.check:
        print(f"{len(entries)} files listed; out of date: {', '.join(changed) if changed else 'none'}")
        sys.exit(1 if (changed or missing) else 0)
    if missing:
        raise SystemExit("not written - fix the missing files first")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        for rel, digest in entries:
            fh.write(f"{digest}  {rel}\n")
    print(f"tools/FILES.sha256: {len(entries)} files, {len(changed)} updated"
          + (f" ({', '.join(changed)})" if changed else ""))


if __name__ == "__main__":
    main()
