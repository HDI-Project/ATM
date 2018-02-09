Contributing to ATM and BTB
===========================

Ways to contribute
------------------
ATM is a research project under active development, and there's a *ton* of work
to do. To get started helping out, you can browse the `issues
<https://github.com/hdi-project/atm/issues>`_ page on
Github and look for issues tagged with "`help wanted
<https://github.com/hdi-project/atm/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_" or "`good first issue
<https://github.com/hdi-project/atm/issues?q=is%3Aissue+is%3Aopen+label%3A"good+first+issue">`_." An easy first pull request might flesh out the documentation for a
confusing feature or just fix a typo. You can also file an issue to report a
bug, suggest a feature, or ask a question. 

If you're looking to make a more in-depth contribution, check out our guides on
`adding a classification method <add_method.html>`_ and `adding a BTB Tuner or Selector <add_to_btb.html>`_.

Requirements
------------
If you'd like to contribute code or documentation, you should install the extra
requirements for testing, style checking, and building documents with::

    pip install -r requirements-dev.txt

Style
-----
We try to stick to the `Google style guide
<https://google.github.io/styleguide/pyguide.html>`_ where possible. We also use
`flake8 <http://flake8.pycqa.org/en/latest/>`_ (for Python best practices) and
`isort <https://pypi.python.org/pypi/isort>`_ (for organizing imports) to
enforce general consistency.

To check if your code passes a style sanity check, run ``make lint`` from the
main directory.

Tests
-----
We currently have a limited (for now!) suite of unit tests that ensure at least
most of ATM is working correctly. You can run the tests locally with ``pytest``
(which will use your local python environment) or ``tox`` (which will create a
new one from scratch); All tests should pass for every commit on master -- this 
means you'll have to update the code in ``atm/tests/unit_tests`` if you modify
the way anything works. In addition, you should create new tests for any new 
features or functionalities you add. See the `pytest documentation
<https://pytest.link>`_ and the existing tests for more information.

All unit and integration tests are run automatically for each pull request and
each commit on master with `CircleCI <https://circleci.com/>`_. We won't merge
anything that doesn't pass all the tests and style checks.

Docs
----
All documentation source files are in the ``docs/source/`` directory. To build
the docs after you've made a change, run ``make html`` from the ``docs/``
directory; the compiled HTML files will be in ``docs/build/``.
