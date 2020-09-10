###
# test, coverage and lint
###

.PHONY : help
help:  ## Use one of the following instructions:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY : lint
lint : ## Static code analysis with Pylint
	pylint -ry climada > pylint.log || true

.PHONY : unit_test
unit_test : ## Unit tests execution with coverage and xml reports
	python -m coverage run  tests_runner.py unit
	python -m coverage xml -o coverage.xml
	python -m coverage html -d coverage

.PHONY : install_test
install_test : ## Test installation was successful
	python tests_install.py report

.PHONY : data_test
data_test : ## Test data APIs
	python test_data_api.py

.PHONY : notebook_test
notebook_test : ## Test notebooks in doc/tutorial
	python test_notebooks.py

.PHONY : integ_test
integ_test : ## Integration tests execution with xml reports
	python -m coverage run --parallel-mode --concurrency=multiprocessing tests_runner.py integ
	python -m coverage combine
	python -m coverage xml -o coverage.xml
	python -m coverage html -d coverage

.PHONY : test
test : ## Unit and integration tests execution with coverage and xml reports
	python -m coverage run --parallel-mode --concurrency=multiprocessing tests_runner.py unit
	python -m coverage run --parallel-mode --concurrency=multiprocessing tests_runner.py integ
	python -m coverage combine
	python -m coverage xml -o coverage.xml
	python -m coverage html -d coverage

.PHONY : ci-clean
ci-clean :
	rm -rf tests_xml
	rm pylint.log

