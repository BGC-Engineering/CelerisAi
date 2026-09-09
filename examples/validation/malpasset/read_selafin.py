"""Minimal Selafin (TELEMAC .slf, big-endian Fortran records) reader."""
import struct
import sys

import numpy as np


def _rec(f):
    """Read one Fortran record: 4-byte BE length, payload, 4-byte BE length."""
    n = struct.unpack(">i", f.read(4))[0]
    data = f.read(n)
    n2 = struct.unpack(">i", f.read(4))[0]
    assert n == n2, f"corrupt record: {n} != {n2}"
    return data


def read_selafin(path):
    """Return dict: title, varnames, nelem, npoin, ndp, ikle (0-based, nelem x ndp), ipobo, x, y,
    times (nt,), values (nt, nvar, npoin) float32, variables {name: (nt, npoin) view of values}."""
    with open(path, "rb") as f:
        title = _rec(f).decode("latin-1").strip()
        nbv1, nbv2 = struct.unpack(">2i", _rec(f))
        names = [_rec(f).decode("latin-1")[:16].strip() for _ in range(nbv1 + nbv2)]
        iparam = struct.unpack(">10i", _rec(f))
        if iparam[9] == 1:
            _rec(f)  # date record
        nelem, npoin, ndp, _one = struct.unpack(">4i", _rec(f))
        ikle = np.frombuffer(_rec(f), dtype=">i4").reshape(nelem, ndp) - 1
        ipobo = np.frombuffer(_rec(f), dtype=">i4")
        x = np.frombuffer(_rec(f), dtype=">f4").astype(np.float64)
        y = np.frombuffer(_rec(f), dtype=">f4").astype(np.float64)
        assert len(x) == npoin == len(y) == len(ipobo), (len(x), npoin)
        assert ikle.min() >= 0 and ikle.max() == npoin - 1, "ikle out of range"
        times, values = [], []
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            n = struct.unpack(">i", head)[0]
            times.append(struct.unpack(">f", f.read(n))[0])
            f.read(4)
            frame = [np.frombuffer(_rec(f), dtype=">f4") for _ in range(nbv1 + nbv2)]
            values.append(np.stack(frame))
    values = np.array(values, dtype=np.float32)
    return dict(
        title=title, varnames=names, nelem=nelem, npoin=npoin, ndp=ndp, ikle=ikle,
        ipobo=ipobo, x=x, y=y, times=np.array(times, dtype=np.float64), values=values,
        variables={n: values[:, k] for k, n in enumerate(names)},
    )


if __name__ == "__main__":
    m = read_selafin(sys.argv[1])
    print(m["title"], "|", m["varnames"], "| nelem", m["nelem"], "npoin", m["npoin"], "ndp", m["ndp"])
    print("x", m["x"].min(), m["x"].max(), "y", m["y"].min(), m["y"].max())
    for k, v in enumerate(m["varnames"]):
        print(v, "min", m["values"][:, k].min(), "max", m["values"][:, k].max(), "frames", len(m["times"]))
