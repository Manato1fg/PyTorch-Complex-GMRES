# build.ps1 - Windows build script for torch-GMRES
# This script mimics the steps in build.sh using PowerShell

# Build Docker image
docker build -f Dockerfile.manylinux -t torch-gmres:manylinux .

# Run Docker container
docker run -d --name torch-gmres-container torch-gmres:manylinux

# Copy output from container to host
if (Test-Path -Path ./dist) {
    Remove-Item -Recurse -Force ./dist
}
docker cp torch-gmres-container:/output ./dist

# Stop and remove the container
docker stop torch-gmres-container
docker rm -f torch-gmres-container
