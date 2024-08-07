import tempfile
import os

import numpy as np

from tfcgan.cli import run


def test_cli_routine():
    """test the whole command line interface (cli) routine"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        prev_files = set(os.listdir(tmpdirname))
        N = 10
        run(['-m', '6', '-d', '50', '-n', str(N), '-v', '700', tmpdirname])
        files = set(os.listdir(tmpdirname)) - prev_files
        assert len(files) == 1
        file = list(files)[0]
        data = np.load(os.path.join(tmpdirname, file))
        assert data.shape[0] == N
        assert data.shape[1] == 4000
