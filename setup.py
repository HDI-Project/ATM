#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'baytune>=0.2.5,<0.3',
    'boto3>=1.9.146,<2',
    'future>=0.16.0,<0.18',
    'pymysql>=0.9.3,<0.10',
    'numpy>=1.13.1,<1.17',
    'pandas>=0.22.0,<0.25',
    'psutil>=5.6.1,<6',
    'python-daemon>=2.2.3,<3',
    'requests>=2.18.4,<3',
    'scikit-learn>=0.18.2,<0.22',
    'scipy>=0.19.1,<1.4',
    'sqlalchemy>=1.1.14,<1.4',
    'flask>=1.0.2,<2',
    'flask-restless>=0.17.0,<0.18',
    'flask-sqlalchemy>=2.3.2,<2.5',
    'flask-restless-swagger-2==0.0.3',
    'simplejson>=3.16.0,<4',
    'tqdm>=4.31.1,<5',
    'docutils>=0.10,<0.15',
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
    'google-compute-engine==2.8.12',    # required by travis
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
    'autodocsumm>=0.1.10',

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
    version='0.2.2',
    zip_safe=False,
)
