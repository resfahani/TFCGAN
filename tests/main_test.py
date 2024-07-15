import tempfile
import os
from tfcgan.cli import run


def test_routine():
    with tempfile.TemporaryDirectory() as tmpdirname:
        run(['-m', '6', '-d', '50', '-n', '10', '-v', '700', tmpdirname])
        assert len(os.listdir(tmpdirname)) > 0
