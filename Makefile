TEST_CMD=setup.py test --addopts --boxed
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '*.egg-info' -delete

lint:
	flake8 atm && isort --check-only --recursive atm

test: lint
	python $(TEST_CMD)

installdeps:
	pip install --upgrade pip
	ssh-keyscan -H github.com > /etc/ssh/ssh_known_hosts
	pip install -e . --process-dependency-links --quiet
	pip install -r requirements-dev.txt --quiet

