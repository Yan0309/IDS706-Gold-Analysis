.PHONY: install format format-check lint test check

install:
	python -m pip install -r requirements-dev.txt

format:
	black hw2_analysis.py hw3_analysis.py test_analysis.py polars_comparison.py

format-check:
	black --check hw2_analysis.py hw3_analysis.py test_analysis.py polars_comparison.py

lint:
	flake8 hw2_analysis.py hw3_analysis.py test_analysis.py polars_comparison.py

test:
	pytest

check: format-check lint test