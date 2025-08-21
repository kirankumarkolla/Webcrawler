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

### Cleanup space from docker images
        # Remove stopped containers
    docker container prune -f

# Remove unused images
    docker image prune -a -f

# Remove unused volumes
    docker volume prune -f

# Remove unused build cache
    docker builder prune -a -f


# Run postgres docker locally
    docker pull postgres
    docker run --name my_postgres -d -e POSTGRES_PASSWORD=kiran1234 -v my_pgdata:/var/lib/postgresql/data -p 5432:5432 postgres

## Connecting other apps
## You can also connect other applications to the PostgreSQL database running in the Docker container. You need to provide the following connection details:

    Host: localhost
    Port: 5432
    Username: postgres
    Password: (whatever you set it to)
    Database: postgres
or Connection String
        postgresql://postgres:mysecretpassword@localhost:5432/postgres
