docker build -f Dockerfile.manylinux -t torch-gmres:manylinux .

docker run -d --name torch-gmres-container torch-gmres:manylinux
docker cp torch-gmres-container:/output ./dist
docker stop torch-gmres-container
docker rm -f torch-gmres-container
