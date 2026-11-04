"""Minimal SU reader/writer for the K-H test suite.

fdelfwi writes native-endian SU with a 240-byte header per trace.  Only 'ns'
(short at offset 114) is needed to parse; everything else is passed through
untouched so round-tripping a file cannot corrupt headers.
"""
import numpy as np
import struct

TRCBYTES = 240


def read(fn):
    """Return (data, headers) with data shape (ntrace, ns), headers a list of
    raw 240-byte header blobs."""
    raw = open(fn, 'rb').read()
    ns = struct.unpack_from('<h', raw, 114)[0]
    tl = TRCBYTES + 4 * ns
    if len(raw) % tl:
        raise ValueError('%s: size %d not a multiple of trace length %d (ns=%d)'
                         % (fn, len(raw), tl, ns))
    ntr = len(raw) // tl
    data = np.empty((ntr, ns), dtype=np.float32)
    hdrs = []
    for i in range(ntr):
        off = i * tl
        hdrs.append(raw[off:off + TRCBYTES])
        data[i] = np.frombuffer(raw, dtype='<f4', count=ns, offset=off + TRCBYTES)
    return data, hdrs


def write(fn, data, hdrs):
    """Write data (ntrace, ns) back out with the supplied headers."""
    assert len(hdrs) == data.shape[0], (len(hdrs), data.shape)
    with open(fn, 'wb') as fp:
        for i in range(data.shape[0]):
            fp.write(hdrs[i])
            fp.write(np.ascontiguousarray(data[i], dtype='<f4').tobytes())


def ricker(nt, dt, f0, tc):
    """Zero-phase Ricker, peak amplitude 1, centred at time tc."""
    t = np.arange(nt) * dt - tc
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def snapframes(fn, nx, nz):
    """Split a writeSnapTimes output file into frames of shape (nx, nz)."""
    data, _ = read(fn)
    ntr, ns = data.shape
    if ns != nz:
        raise ValueError('%s: ns=%d, expected nz=%d' % (fn, ns, nz))
    if ntr % nx:
        raise ValueError('%s: %d traces not a multiple of nx=%d' % (fn, ntr, nx))
    return data.reshape(ntr // nx, nx, nz)
