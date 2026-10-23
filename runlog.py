"""
runlog.py  -  every script of the project also writes what it prints into outputs/logs/<script>.txt, so after a run
you can send that one file instead of copying the console. It also makes the console output safe on Windows
(a character the console cannot show is replaced instead of stopping the script).
"""
import datetime
import io
import os
import sys
import traceback


class _Tee(io.TextIOBase):
    def __init__(self, console, fh):
        self.console, self.fh = console, fh

    def write(self, s):
        try:
            self.console.write(s)
        except UnicodeEncodeError:
            enc = getattr(self.console, "encoding", None) or "ascii"
            self.console.write(s.encode(enc, errors="replace").decode(enc, errors="replace"))
        except (OSError, ValueError):
            pass
        try:
            self.fh.write(s.replace("\r", "\n"))
        except (OSError, ValueError):
            pass
        return len(s)

    def flush(self):
        for f in (self.console, self.fh):
            try:
                f.flush()
            except (OSError, ValueError):
                pass

    def isatty(self):
        try:
            return self.console.isatty()
        except (OSError, ValueError, AttributeError):
            return False

    @property
    def encoding(self):
        return getattr(self.console, "encoding", "utf-8")


class Log:
    """with runlog.Log("02_compute_soz"): main()   ->  outputs/logs/02_compute_soz.txt"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import config
        d = os.path.join(config.OUT_DIR, "logs")
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, f"{self.name}.txt")
        self.fh = open(self.path, "w", encoding="utf-8")
        self.fh.write(f"# {self.name}   {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   python {' '.join(sys.argv)}\n"
                      f"# project version {getattr(config, 'EZ_VERSION', '?')}\n\n")
        self.out, self.err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self.out, self.fh), _Tee(self.err, self.fh)
        return self

    def __exit__(self, et, ev, tb):
        try:
            if et is SystemExit:
                code = ev.code if ev is not None else None
                if isinstance(code, str):
                    self.fh.write("\n" + code + "\n")
            elif et is KeyboardInterrupt:
                self.fh.write("\nStopped by the user (Ctrl+C).\n")
            elif et is not None:
                self.fh.write("\n" + "".join(traceback.format_exception(et, ev, tb)))
        finally:
            sys.stdout, sys.stderr = self.out, self.err
            self.fh.close()
        return False
