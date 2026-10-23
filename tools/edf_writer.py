"""
edf_writer.py  -  a minimal EDF writer, used only to create the fake recordings of the smoke test
(the real dataset is only read, never written).
"""
import numpy as np


def _field(value, width):
    s = str(value)
    if len(s) > width:
        raise ValueError(f"EDF header field {s!r} longer than {width} characters")
    return s.ljust(width).encode("ascii")


def _num(v, width=8):
    s = f"{v:.2f}"
    return s if len(s) <= width else f"{v:.0f}"[:width]


def write_edf(path, data_uv, fs, ch_names, record_s=1, annotations=False):
    """data_uv: channels x samples (microvolts). fs must be an integer number of samples per record.
    annotations=True adds an 'EDF Annotations' signal (EDF+C, like the files of the expert benchmark)."""
    data_uv = np.asarray(data_uv, dtype=float)
    n_ch, n_samp = data_uv.shape
    spr = int(round(fs * record_s))
    n_rec = int(np.ceil(n_samp / spr))
    padded = np.zeros((n_ch, n_rec * spr))
    padded[:, :n_samp] = data_uv
    dmin, dmax = -32768, 32767
    pmin = np.floor(padded.min(axis=1)) - 1
    pmax = np.ceil(padded.max(axis=1)) + 1
    same = pmax - pmin < 2
    pmax[same] = pmin[same] + 2
    scale = (dmax - dmin) / (pmax - pmin)
    dig = np.clip(np.round((padded - pmin[:, None]) * scale[:, None] + dmin), dmin, dmax).astype("<i2")

    n_sig = n_ch + (1 if annotations else 0)
    ann_spr = 60
    labels = list(ch_names) + (["EDF Annotations"] if annotations else [])
    dims = ["uV"] * n_ch + ([""] if annotations else [])
    p_lo = [_num(v) for v in pmin] + (["-1"] if annotations else [])
    p_hi = [_num(v) for v in pmax] + (["1"] if annotations else [])
    nsamp = [spr] * n_ch + ([ann_spr] if annotations else [])
    hdr = b"".join([
        _field("0", 8), _field("X X X X", 80), _field("Startdate 01-JAN-2020 X X X", 80),
        _field("01.01.20", 8), _field("00.00.00", 8), _field(256 + 256 * n_sig, 8),
        _field("EDF+C" if annotations else "", 44),
        _field(n_rec, 8), _field(record_s, 8), _field(n_sig, 4),
    ])
    hdr += b"".join(_field(n, 16) for n in labels)
    hdr += b"".join(_field("", 80) for _ in labels)
    hdr += b"".join(_field(d, 8) for d in dims)
    hdr += b"".join(_field(v, 8) for v in p_lo)
    hdr += b"".join(_field(v, 8) for v in p_hi)
    hdr += b"".join(_field(dmin, 8) for _ in labels)
    hdr += b"".join(_field(dmax, 8) for _ in labels)
    hdr += b"".join(_field("", 80) for _ in labels)
    hdr += b"".join(_field(n, 8) for n in nsamp)
    hdr += b"".join(_field("", 32) for _ in labels)
    assert len(hdr) == 256 + 256 * n_sig
    with open(path, "wb") as fh:
        fh.write(hdr)
        for r in range(n_rec):
            fh.write(dig[:, r * spr:(r + 1) * spr].tobytes())
            if annotations:
                tal = f"+{r * record_s}\x14\x14\x00".encode("ascii")
                fh.write(tal.ljust(2 * ann_spr, b"\x00"))
