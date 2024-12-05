"""cli (command line interface) module of the program"""
import sys
import os
import warnings
import argparse
from argparse import RawTextHelpFormatter

import numpy as np


def create_parser():
    parser = argparse.ArgumentParser(
        description="TFCGAN is a generator of synthetic seismic waveforms using "
                    "a pre-trained ANN model"
    )
    # positional arguments:
    parser.add_argument(
        'output',  # <- positional argument
        type=str,
        help=f'The output file. The data will be saved as matrix where the '
             f'the first row represents the times (in s) and each subsequent row '
             f'a synthetic waveform (time history). The file format can be '
             f'specified by its extension: .gz, .txt and .ascii -> save as text (.gz '
             f'is compressed and recommended for large datasets. In general, use these '
             f'formats only if you need to open the files outside a python/numpy '
             f'environment), .npy and .npz -> save as binary in numpy format (for .npz, '
             f'the times are saved as "x" and the waveforms as "y"). See numpy save, '
             f'savez and load for details',
        metavar="output-file"
    )
    # add argument to ArgParse:
    parser.add_argument(
        "-n",
        type=int,
        required=True,
        dest="num_waveforms",
        help='number of generated synthetic waveforms'
    )
    parser.add_argument(
        "-m",
        type=float,
        required=True,
        dest="magnitude",
        help='The seismic event magnitude'
    )
    parser.add_argument(
        "-d",
        type=float,
        required=True,
        dest="distance",
        help='The site distance (km)'
    )
    parser.add_argument(
        "-v",
        type=float,
        dest="vs30",
        default=760,
        help='The site Vs30 (default when missing: 760)'
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
        from tfcgan import TFCGAN
        try:
            if verbose:
                print('Creating waveforms')

            # Model
            tfcgan = TFCGAN()

            # Generate waveform data
            t, data = tfcgan.get_ground_shaking_synthesis(
                args.num_waveforms, mw=args.magnitude, rhyp=args.distance,
                vs30=args.vs30
            )
            output_file = args.output
            f_format = os.path.splitext(os.path.basename(output_file))[1].lower()
            if f_format not in ('.npy', '.txt', '.gz', '.npz', '.ascii'):
                f_format = '.npy'
                output_file += f_format
            output_dir = os.path.dirname(output_file)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            if verbose and os.path.isfile(output_file):
                answer = input('output file exists. Overwrite (y=yes)?')
                if answer != 'y':
                    if verbose:
                        print('Aborted by user')
                    return 0

            if verbose:
                print(f'Saving waveforms to {output_file}')
            if f_format == '.npz':
                np.savez(file=output_file, x=t, y=data)
            elif f_format in ('.txt', '.ascii', '.npy', '.gz'):
                saved_data = np.vstack((t, data))
                if f_format == '.npy':
                    np.save(file=output_file, arr=saved_data)
                else:
                    np.savetxt(fname=output_file, X=saved_data)
            else:
                raise ValueError(f'Unrecognized file format "{f_format}"')
            return 0
        except Exception as exc:
            print(f'{exc.__class__.__name__}: {str(exc)}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    sys.exit(run())
