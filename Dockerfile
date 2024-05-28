FROM python:3.8-slim

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt/program

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get install -y --no-install-recommends wget nginx git ffmpeg libsm6 libxext6 build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

# Install Poetry
RUN pip install poetry

# Copy the poetry files
COPY pyproject.toml /opt/program/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --only=main

# Install PyTorch
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# files
COPY src/inference /opt/program

# serve script
RUN chmod +x /opt/program/serve.sh

# Clean up cache
RUN rm -rf /root/.cache

EXPOSE 8080

ENTRYPOINT ["/opt/program/serve.sh"]
