###
# test, coverage and lint
###

PRE_CMD = uv run --

PYTEST_JUNIT_ARGS = --junitxml=tests_xml/tests.xml

PYTEST_COV_ARGS = \
	--cov --cov-config=.coveragerc --cov-report html:coverage \
	--cov-report xml:coverage.xml --cov-report term:skip-covered

PYTEST_ARGS = $(PYTEST_JUNIT_ARGS) $(PYTEST_COV_ARGS)

.PHONY : help
help:  ## Use one of the following instructions:
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY : lint
lint : ## Static code analysis with Pylint
	$(PRE_CMD) pylint -ry climada > pylint.log || true

.PHONY : unit_test
unit_test : ## Unit tests execution with coverage and xml reports
	$(PRE_CMD) pytest $(PYTEST_ARGS) --ignore=climada/test climada/

.PHONY : install_test
install_test : ## Test installation was successful
	$(PRE_CMD) pytest $(PYTEST_JUNIT_ARGS) --pyargs climada.engine.test.test_cost_benefit \
	climada.engine.test.test_impact

.PHONY : data_test
data_test : ## Test data APIs
	$(PRE_CMD) pytest $(PYTEST_JUNIT_ARGS) script/jenkins/test_data_api.py

.PHONY : notebook_test
notebook_test : ## Test notebooks in doc/tutorial
	$(PRE_CMD) pytest $(PYTEST_JUNIT_ARGS) script/jenkins/test_notebooks.py

.PHONY : integ_test
integ_test : ## Integration tests execution with xml reports
	$(PRE_CMD) pytest $(PYTEST_ARGS) climada/test/

.PHONY : test
test : ## Unit and integration tests execution with coverage and xml reports
	$(PRE_CMD) pytest $(PYTEST_ARGS) climada/

.PHONY : ci-clean
ci-clean :
	rm -rf tests_xml
	rm pylint.log coverage.xml
	rm -r coverage
