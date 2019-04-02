#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'baytune==0.2.5',
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
]

setup_requires = [
    'pytest-runner'
]

tests_require = [
    'mock>=2.0.0',
    'pytest-cov>=2.5.1',
    'pytest-runner>=3.0',
    'pytest-xdist>=1.20.1',
    'pytest>=3.2.3',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description="Auto Tune Models",
    entry_points={
        'console_scripts': [
            'atm=atm.cli:main'
        ]
    },
    extras_require={
        'dev': development_requires + tests_require,
        'tests': tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    keywords='machine learning hyperparameters tuning classification',
    name='atm',
    packages=find_packages(include=['atm', 'atm.*']),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-project/ATM',
    version='0.1.1-dev',
    zip_safe=False,
)
