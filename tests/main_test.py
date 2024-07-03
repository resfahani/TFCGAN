import tempfile
from tfcgan.cli import run


def test_routine():
    with tempfile.TemporaryDirectory() as tmpdirname:
        run(['-m', '6', '-d', '100', '-n', '10', '-v', '700', tmpdirname])