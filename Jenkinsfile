pipeline {
  agent any
  stages {
    stage('lint') {
      parallel {
        stage('lint') {
          steps {
            sh '''\'\'\'#!/bin/bash
        export PATH=$PATH:$CONDAPATH
        source activate climada_env
        make lint\'\'\''''
          }
        }

        stage('unit_test') {
          steps {
            sh '''#!/bin/bash
        export PATH=$PATH:$CONDAPATH
        source activate climada_env
        make unit_test'''
          }
        }

      }
    }

  }
}