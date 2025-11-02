pipeline {
  agent any

  environment {
    DOCKER_IMAGE = "your-docker-user/travel-ml-api:${env.BUILD_NUMBER}"
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Setup Python') {
      steps {
        sh '''
          python -m venv .venv
          . .venv/bin/activate
          pip install -r requirements.txt
        '''
      }
    }
    stage('Lint') {
      steps {
        sh '''
          . .venv/bin/activate
          flake8 src api streamlit_app
        '''
      }
    }
    stage('Test') {
      steps {
        sh '''
          . .venv/bin/activate
          pytest -q
        '''
      }
    }
    stage('Build Docker') {
      steps { sh "docker build -t ${DOCKER_IMAGE} ." }
    }
    stage('Push Docker') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'USER', passwordVariable: 'PASS')]) {
          sh '''
            echo "$PASS" | docker login -u "$USER" --password-stdin
            docker push ${DOCKER_IMAGE}
          '''
        }
      }
    }
    stage('Deploy to K8s') {
      steps {
        sh '''
          sed -i "s#your-docker-user/travel-ml-api:latest#${DOCKER_IMAGE}#g" k8s/deployment.yaml
          kubectl apply -f k8s/deployment.yaml
          kubectl apply -f k8s/service.yaml
          kubectl apply -f k8s/hpa.yaml
        '''
      }
    }
  }
  post {
    always {
      archiveArtifacts artifacts: 'models/**', allowEmptyArchive: true
    }
  }
}