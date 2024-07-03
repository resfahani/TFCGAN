"""cli (command line interface) module of the program

@author: Riccardo Z. <rizac@gfz-potsdam.de>
"""
import sys
import warnings
import inspect
import re
import argparse
from argparse import RawTextHelpFormatter


def process(output_directory:str, mag:float, dist:float, vs30:float, number_of_waveforms:int, verbose=False):
    """TODO add doc

    :param verbose: (boolean flag) increase verbosity. When given, additional
        info and errors will be printed to stderr. Also in this case, the
        tabular output header will be printed to stdout instead of stderr,
        which is useful to create CSV files with headers
    """
    

def getdoc(param=None):
    """Parse the doc of the `process` function and returns the doc for the
    given param. If the latter is None, returns the doc for the whole
    function (portion of text from start until first occurrence of ":param "
    """
    flags = re.DOTALL  # @UndefinedVariable
    pattern = "^(.*?)\\n\\s*\\:param " if not param else \
        f"\\:param {param}: (.*?)(?:$|\\:param)"
    stripstart = "\n    " if not param else "\n        "
    try:
        return re.search(pattern, process.__doc__, flags).\
            group(1).strip().replace(stripstart, "\n") + '\n'
    except AttributeError:
        return 'No doc available'


def getdef(param):
    func = process
    signature = inspect.signature(func)
    val = signature.parameters[param]
    if val.default is not inspect.Parameter.empty:
        return val.default
    raise ValueError(f'"{param}" has no default')


#####################
# ArgumentParser code
#####################

def cli_entry_point():
    parser = argparse.ArgumentParser(
        description=getdoc(),
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('data',  # <- positional argument
                        type=str,
                        # dest='data',  # invalid for positional argument
                        metavar='data',
                        help=getdoc('data'))
    # optional arguments. Argparse argument here must have a match with
    # an argument of the `process` function above (the match is based on the
    # function name with the names defined below). To implement any new
    # optional argument, provide it in `process` WITH A DEFAULT (mandatory)
    # and a help in the function doc (recommended) and add the argument
    # name here below, with a correponding flag
    for flag, name in [
        ('-m', 'magnitude'),  # float
        ('-v', 'vs30'),  # in m/s float
        ('-d', 'distance'),  # in km, float
        ('-n', 'number_of_waveforms'),  # int
        ('-v', 'verbose')
    ]:
        param_default, param_doc = getdef(name), getdoc(name)
        kwargs = {
            'dest': name,
            'metavar': name,
            'help': param_doc
        }
        if param_default in (True, False):  # boolean flag
            kwargs['action'] = 'store_false' if param_default else 'store_true'
            kwargs.pop('metavar')  # invalid for store_true action
        else:  # no boolean flag
            kwargs['default'] = param_default
            kwargs['type'] = type(param_default)

        # add argument to ArgParse:
        parser.add_argument(flag, **kwargs)

    with warnings.catch_warnings(record=False) as wrn:  # noqa
        # Cause all warnings to always be triggered.
        warnings.simplefilter("ignore")
        # parse arguments and pass them to `process`
        # (here we see why names must match):
        args = parser.parse_args()
        try:
            process(**vars(args))
            sys.exit(0)
        except Exception as exc:
            # raise
            print(f'ERROR: {str(exc)}', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    cli_entry_point()
