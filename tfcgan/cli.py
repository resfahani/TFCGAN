"""cli (command line interface) module of the program

@author: Riccardo Z. <rizac@gfz-potsdam.de>
"""
import math
import sys
import os
import warnings
import argparse
from argparse import RawTextHelpFormatter

import numpy as np


#####################
# ArgumentParser code
#####################

def create_parser():
    parser = argparse.ArgumentParser(
        description="TFCGAN is a generator of synthetic seismic waveforms using "
                    "a pre-trained ANN model",
        formatter_class=RawTextHelpFormatter
    )
    # positional arguments:
    parser.add_argument(
        'output',  # <- positional argument
        # dest='output',
        type=str,
        metavar='output_directory',
        help="output directory")
    # add argument to ArgParse:
    parser.add_argument(
        "-m",
        type=float,
        dest="magnitude",
        help='The seismic event magnitude'
    )
    parser.add_argument(
        "-v",
        type=float,
        dest="vs30",
        help='The site Vs30'
    )
    parser.add_argument(
        "-d",
        type=float,
        dest="distance",
        help='The site distance (km)')
    parser.add_argument(
        "-n",
        type=int,
        dest="number_of_waveforms",
        help='number of generated synthetic waveforms'
    )
    parser.add_argument(
        "-q",
        action='store_true',
        help='quiet mode: overwrite existing output file(s) without prompting and '
             'do not print info'
    )

    return parser


def run(arguments):
    parser = create_parser()
    with warnings.catch_warnings(record=False) as wrn:  # noqa
        # Cause all warnings to always be triggered.
        warnings.simplefilter("ignore")
        # parse arguments and pass them to `process`
        # (here we see why names must match):
        args = parser.parse_args(arguments)
        verbose = not args.q
        if verbose:
            print('Loading TFCGAN model and libraries')
        from tfcgan.tfcgan import TFCGAN
        try:
            if verbose:
                print('Creating waveforms')
            tfc = TFCGAN().maker(
                args.magnitude,
                args.distance,
                args.vs30,
                args.number_of_waveforms)
            output_dir = args.output
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(output_dir)
            output_file = os.path.join(output_dir, 'tfcgan-dt0.01.npy')
            if verbose and os.path.isfile(output_file):
                answer = input('output file exists. Overwrite (y=yes)?')
                if answer != 'y':
                    if verbose:
                        print('Aborted by user')
                    sys.exit(0)
            x_hist = tfc[4]
            if verbose:
                print(f'Saving waveforms to {output_file}')
            np.save(file=output_file, arr=x_hist)
            # sys.exit(0)
        except Exception as exc:
            # raise
            print(f'{exc.__class__.__name__}: {str(exc)}', file=sys.stderr)
            # sys.exit(1)


if __name__ == '__main__':
    run(sys.argv[1:])
