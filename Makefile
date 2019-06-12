test:
	. venv/bin/activate && python -m pytest src

functional-test:
	. venv/bin/activate && python -m pytest functional_tests
