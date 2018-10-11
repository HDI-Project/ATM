#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'baytune>=0.2.2',
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

test_requirements = [
    'mock>=2.0.0',
    'pytest-cov>=2.5.1',
    'pytest-runner>=3.0',
    'pytest-xdist>=1.20.1',
    'pytest>=3.2.3',
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
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='machine learning hyperparameters tuning classification',
    name='atm',
    packages=find_packages(include=['atm', 'atm.*']),
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    setup_requires=setup_requires,
    test_suite='atm/tests',
    tests_require=test_requirements,
    url='https://github.com/HDI-project/ATM',
    version='0.1.0',
    zip_safe=False,
)
