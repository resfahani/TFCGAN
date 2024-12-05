from setuptools import setup, find_packages

_README = """
Python program and library for strong ground motion simulation in time-frequency 
domain (TF) based on conditional generative adversarial network and phase 
retrieval approaches
"""

setup(
    name='tfcgan',
    version='1.0.0',
    description=_README,
    url='https://github.com/resfahani/TFCGAN',
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='==3.10.0',
    # Minimal requirements, for a complete list see requirements-*.txt
    install_requires=[
        'numpy==1.26.1',
        'tensorflow==2.11.1',
        # 'obspy==1.4.1',
        'scipy==1.14.1'
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e ".[dev]"
    extras_require={
        # 'jupyter': [
        #     'jupyter>=1.1.0'
        # ],
        'dev': [
            'pytest'
        ]
    },
    author='Reza D. D. Esfahani, Riccardo Zaccarelli',
    author_email='',
    maintainer='Reza D. D. Esfahani, , Riccardo Zaccarelli',
    maintainer_email='',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        "seismic waveform",
        # TODO add keywords
    ],
    license="GPL3",
    platforms=["any"],  # FIXME: shouldn't be unix/macos? (shallow google search didn't help)
    # package_data={"key": [
    #    "README.md", "LICENSE"]},

    # make the installation process copy also the models (see MANIFEST.in)
    # for info see https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    include_package_data=True,
    zip_safe=False,
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'tfcgan=tfcgan.cli:run',
        ],
    },
)
