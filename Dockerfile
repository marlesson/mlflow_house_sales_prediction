# Build a Docker file to server mlflow pyfunc
#
# docker build -t model_mlflow .
# docker run -it -p 3001:80 model_mlflow


# Dockerfile
FROM continuumio/miniconda3:latest

# Model
ARG model_path=./models/44d4858b516e483d9ec827e6ada0592c/model

# build
COPY ${model_path} /app/model
RUN pip install mlflow==0.8.2

# Run
EXPOSE 80
ENTRYPOINT mlflow pyfunc serve -m /app/model --host 0.0.0.0 -p 80 --no-conda