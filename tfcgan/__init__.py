import os
# disable tf printouts. For info see
# https://stackoverflow.com/a/42121886 and https://stackoverflow.com/a/65333085
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from .tfcgan import TFCGAN  # noqa

