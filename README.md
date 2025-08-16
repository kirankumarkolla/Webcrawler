# Webcrawler
### sudo yum update -y
### sudo amazon-linux-extras install docker
### sudo amazon-linux-2023 install docker
    4      sudo dnf update -y
### sudo dnf install -y docker
    6  sudo service docker start
    7      sudo usermod -aG docker ec2-user

    To copy files from local to ec2 
    8>pscp -i "ec2-docker.ppk" "K:\AI\Webcrawler\*" ec2-user@ec2-3-137-137-207.us-east-2.compute.amazonaws.com:/home/ec2-user/downloads

### Install docker-compose on EC2
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        docker-compose --version
