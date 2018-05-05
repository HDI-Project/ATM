"""
Stripped down and modified from the example at:
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='atm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='a multi-user, multi-data model exploration system',

    # The project's main homepage.
    url='https://github.com/HDI-project/ATM',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # TODO: python 3 support
    ],

    # What does your project relate to?
    keywords='machine learning hyperparameters tuning classification',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    python_requires='>=2.7',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'baytune==0.1.2',  # This one needs to be exact
        'boto>=2.48.0',
        'future>=0.16.0',
        'joblib>=0.11',
        'mysqlclient>=1.2',
        'numpy>=1.13.1',
        'pandas>=0.22.0',
        'pyyaml>=3.12',
        'requests>=2.18.4',
        'scikit-learn>=0.18.2',
        'scipy>=0.19.1',
        'sklearn-pandas>=1.5.0',
        'sqlalchemy>=1.1.14',
    ],

    # This variable is used to specify requirements for *this file* to run.
    setup_requires=[],

    test_suite='atm/tests',
    tests_require=[
        'mock>=2.0.0',
        'pytest-cov>=2.5.1',
        'pytest-runner>=3.0',
        'pytest-xdist>=1.20.1',
        'pytest>=3.2.3',
    ],
    include_package_data=True
)
