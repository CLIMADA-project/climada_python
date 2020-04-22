pipeline {
  agent any
  stages {
    stage('conda activate') {
      steps {
        sh '$CONDAPATH/conda activate climada_env'
      }
    }

    stage('lint') {
      parallel {
        stage('lint') {
          steps {
            sh 'make lint'
          }
        }

        stage('unit_test') {
          steps {
            sh 'make unit_test'
          }
        }

      }
    }

    stage('error') {
      steps {
        sh 'conda deactivate'
      }
    }

  }
}