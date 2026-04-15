FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    cmake ninja-build git build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install liboqs
RUN git clone --depth=1 https://github.com/open-quantum-safe/liboqs && \
    cd liboqs && \
    cmake -S . -B build -DBUILD_SHARED_LIBS=ON && \
    cmake --build build && \
    cmake --install build && \
    ldconfig

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install python wrapper
RUN pip install liboqs-python

# Copy app
COPY . /app

ENV PYTHONPATH=.
ENV CRYPTO_MODE=PQC

CMD ["python3", "main.py"]