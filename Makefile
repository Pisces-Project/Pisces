.PHONY: clean build docs test

clean:
	find pisces -name "*.so" -delete
	find pisces -name "*.pyc" -delete
	find pisces -name "__pycache__" -exec rm -rf {} +
	rm -rf build dist *.egg-info
	rm -f pisces/_version.py

build:
	python -m build

docs:
	sphinx-build -b html docs _build/html

test:
	pytest
