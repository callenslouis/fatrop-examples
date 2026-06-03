
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    gfortran \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config \
    swig \
    libeigen3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build and install BLASFEO
RUN git clone https://github.com/giaf/blasfeo.git
RUN mkdir blasfeo/build && cd blasfeo/build && \
    cmake .. \
    -DTARGET=X64_AUTOMATIC \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install

# Build and install FATROP
# RUN git clone https://github.com/meco-group/fatrop.git
# RUN mkdir fatrop/build && cd fatrop/build && \
#     cmake .. \
#     -DCMAKE_INSTALL_PREFIX=/usr/local \
#     -DCMAKE_BUILD_TYPE=Release && \
#     make -j$(nproc) && make install
RUN git clone https://github.com/callenslouis/fatrop.git
RUN cd fatrop &&git checkout fatropv1-implicit
RUN mkdir fatrop/build && cd fatrop/build && \
    cmake .. \
    -DBUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install

# Clone CasADi
RUN git clone https://github.com/casadi/casadi.git

WORKDIR /casadi

# (Optional) checkout the nightly-fatropv1 commit/tag
# Replace with the actual tag or commit hash if needed
RUN git checkout nightly-fatropv1 || true

# Create build directory
RUN mkdir build
WORKDIR /casadi/build

# Configure build
RUN cmake .. \
    -DWITH_PYTHON=ON \
    -DWITH_PYTHON3=ON \
    -DWITH_FATROP=ON \
    -DWITH_BUILD_FATROP=OFF \
    -DWITH_BUILD_REQUIRED=ON \
    -DCMAKE_BUILD_TYPE=Release

# Compile
RUN make -j$(nproc)

# Install Python bindings
RUN make install

# Make sure Python can find it
WORKDIR /
RUN pip install matplotlib numpy scipy pyyaml --break-system-packages