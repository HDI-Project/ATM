.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-coverage clean-test clean-docs clean-data ## remove all build, test, coverage, docs and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean-data
clean-data: ## remove generated data
	rm -fr *.db
	rm -fr models/
	rm -fr metrics/
	rm -fr logs/

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -rf docs/build
	rm -f docs/atm.rst
	rm -f docs/atm.*.rst
	rm -f docs/modules.rst
	$(MAKE) -C docs clean

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 atm # tests
	isort -c --recursive atm # tests

.PHONY: fixlint
fixlint: ## fix lint issues using autoflake, autopep8, and isort
	find atm -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive atm
	isort --apply --atomic --recursive atm

	# find tests -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	# autopep8 --in-place --recursive --aggressive tests
	# isort --apply --atomic --recursive tests

.PHONY: test
test: ## run tests quickly with the default Python
	python -m pytest

.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox

.PHONY: coverage
coverage: clean-coverage ## check code coverage quickly with the default Python
	coverage run --source atm -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc -o docs/ atm
	$(MAKE) -C docs html

.PHONY: viewdocs
viewdocs: docs ## view docs in browser
	$(BROWSER) docs/build/index.html

.PHONY: servedocs
servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: release
release: dist ## package and upload a release
	twine upload dist/*

.PHONY: test-release
test-release: dist ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: install
install: clean ## install the package to the active Python's site-packages
	python setup.py install
