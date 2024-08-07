"""cli (command line interface) module of the program

@author: Riccardo Z. <rizac@gfz-potsdam.de>
"""
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
        help=f'The output file. The data will be saved as matrix where each '
             f'row represent a time history (waveform) with its point arranged'
             f'in columns. The file format can be specified by its extension: '
             f'.txt and .gx -> save as text (less efficient '
             f'but can be read outside Python), otherwise save file as '
             f'.npy (numpy format). Numpy files can be opened in Python via: '
             f'data=numpy.load(<output>)',
        metavar="output file"
    )
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
        help='The site distance (km)'
    )
    parser.add_argument(
        "-n",
        type=int,
        dest="number_of_waveforms",
        help='number of generated synthetic waveforms'
    )
    parser.add_argument(
        "-q",
        action='store_true',
        help='quiet mode: overwrite existing output file(s) without asking '
             'and do not print info'
    )
    return parser


def run(arguments=None):
    """
    Run the main routine from the command line

    :param arguments: used only for testing, otherwise it defaults to sys.argv[1:]
    """
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
            output_file = args.output
            f_format = os.path.splitext(os.path.basename(output_file))[1].lower()
            if f_format not in ('.npy', '.txt', '.gz'):
                f_format = '.npy'
                output_file += f_format
            output_dir = os.path.dirname(output_file)
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(output_dir)
            if verbose and os.path.isfile(output_file):
                answer = input('output file exists. Overwrite (y=yes)?')
                if answer != 'y':
                    if verbose:
                        print('Aborted by user')
                    sys.exit(0)
            x_hist = tfc[4]
            if verbose:
                print(f'Saving waveforms to {output_file}')
            if f_format == '.npy':
                np.save(file=output_file, arr=x_hist)
            else:
                np.savetxt(fname=output_file, X=x_hist)
            # sys.exit(0)
        except Exception as exc:
            # raise
            print(f'{exc.__class__.__name__}: {str(exc)}', file=sys.stderr)
            # sys.exit(1)


if __name__ == '__main__':
    run()
