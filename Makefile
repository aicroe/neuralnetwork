start:
	pip install --editable .

test:
	python -Bm tests.run_tests

install:
	pip install .

clean:
	pip uninstall neuralnetwork
