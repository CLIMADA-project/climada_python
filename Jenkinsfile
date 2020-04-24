pipeline {
  agent any

  stages {
    stage('ci') {
      parallel {

        stage('lint') {
          steps {
            sh '''#!/bin/bash
            export PATH=$PATH:$CONDAPATH
            source activate climada_env
            pylint -ry climada | tee pylint.log'''

            step([
                $class: 'WarningsPublisher',
                parserConfigurations: [[
                    parserName: 'PyLint',
                    pattern   : 'pylint.log'
                ]],
                unstableTotalHigh: '25',
                usePreviousBuildAsReference: true
            ])
          }
        }

        stage('unit_test') {
          steps {
            sh '''#!/bin/bash
            export PATH=$PATH:$CONDAPATH
            source activate climada_env
            python -m coverage run  tests_runner.py unit
            python -m coverage xml -o coverage.xml
            python -m coverage html -d coverage'''
          }
        }

      }
    }
  }

  post {
    always {
      junit 'tests_xml/*.xml'
      cobertura coberturaReportFile: 'coverage.xml'
    }
  }
}