docker build -t digital-recognition-ml .

# Run the container with adding model.pth to /opt/ml/model.pth

docker run -p 8080:8080 -v $(pwd)/model.pth:/opt/ml/model/model.pth digital-recognition-ml

curl -X GET "http://localhost:8080/ping"

# serve command

docker run -p 8080:8080 -v $(pwd)/model.pth:/opt/ml/model/model.pth digital-recognition-ml serve

docker buildx create --use
docker buildx inspect --bootstrap
docker buildx build --platform linux/amd64 -t digital-recognition-ml --load .
