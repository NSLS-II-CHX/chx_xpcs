import io
import pytest
import numpy as np
from pathlib import Path
import uuid

from chx_compress.io.multifile.multifile import (MultifileAPS, MultifileBNL)

def test_read_write(tmpdir):
    f1 = tmpdir.mkdir("eiger")
    filename = Path(str(uuid.uuid4()) + ".h5")
    filename = Path(f1.dirname) / filename

    npixels = 10
    rows = 100
    cols = 100
    md = dict()
    md['beam_center_x'] = 0.
    md['beam_center_y'] = 0.
    md['count_time'] = 0.
    md['detector_distance'] = 0.
    md['frame_time'] = 0.
    md['incident_wavelength'] = 0.
    md['x_pixel_size'] = 0.
    md['y_pixel_size'] = 0.
    md['bytes'] = int(0)
    md['nrows'] = rows
    md['ncols'] = cols
    md['rows_begin'] = 0
    md['rows_end'] = rows-1
    md['cols_begin'] = 0
    md['cols_end'] = cols-1

    pos = np.arange(npixels)
    vals = np.random.randint(0, 10000, size=npixels)
    mf = MultifileAPS(filename, "wb")
    mf._write_header(npixels, rows, cols)
    mf.write_raw(pos, vals)
    mf.close()

    mf2 = MultifileAPS(filename, "rb")
    pos2, vals2 = mf2.rdrawframe(0)

