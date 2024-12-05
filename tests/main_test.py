import tempfile
import os
from unittest.mock import patch

import numpy as np

from tfcgan.cli import run
from tfcgan.tfcgan import TFCGAN


def test_cli_routine():
    """test the whole command line interface (cli) routine"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        N = 7
        L = 3952
        file = os.path.abspath(os.path.join(tmpdirname, 'acca'))
        # try with args:
        run(['-m', '6', '-d', '50', '-n', str(N), '-v', '700', file])
        file = file + '.npy'
        assert os.path.isfile(file)
        data = np.load(file)
        assert data.shape == (N+1, L)

        with patch('builtins.input', return_value='y') as ipt:
            N = 1
            run(['-m', '6', '-d', '50', '-n', str(N), '-v', '700', file])
            ipt.assert_called_once()
            assert os.path.isfile(file)
            data = np.load(file)
            assert data.shape == (N+1, L)

            ipt.reset_mock()
            N += 1
            run(['-m', '6', '-d', '50', '-n', str(N), '-v', '700', '-q', file])
            ipt.assert_not_called()
            assert os.path.isfile(file)
            data = np.load(file)
            assert data.shape == (N+1, L)

        for fmt in ('.txt', '.gz'):
            file += fmt
            N = 1
            run(['-m', '6', '-d', '50', '-n', str(N), '-v', '700', file])
            assert os.path.isfile(file)
            data = np.loadtxt(file)
            assert data.shape == (N+1, L)

        time, data = data[0], data[1:]
        freq, fas = TFCGAN.get_fas_response(time[1] - time[0], data)
        assert len(freq) == len(time) // 2
        assert fas.shape == (data.shape[0], len(freq))
