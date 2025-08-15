# Webcrawler
1  sudo yum update -y
    2  sudo amazon-linux-extras install docker
    3  sudo amazon-linux-2023 install docker
    4      sudo dnf update -y
    5      sudo dnf install -y docker
    6  sudo service docker start
    7      sudo usermod -aG docker ec2-user

    To copy files from local to ec2 
    8>pscp -i "ec2-docker.ppk" "K:\AI\Webcrawler\*" ec2-user@ec2-3-137-137-207.us-east-2.compute.amazonaws.com:/home/ec2-user/downloads
