
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
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Build and install BLASFEO
# Pinned: newer BLASFEO changed the blasfeo_drowpe/dcolpe signatures, which breaks fatrop
RUN git clone https://github.com/giaf/blasfeo.git && \
    git -C blasfeo checkout 60493284741b3e4e5b0bf25155221d4b5f0b2232
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
# The branch head is downloaded first, so the clone below is not taken from the build cache when the
# branch has new commits
ADD https://api.github.com/repos/callenslouis/fatrop/git/refs/heads/fatropv1-implicit /tmp/fatrop_ref.json
RUN git clone https://github.com/callenslouis/fatrop.git && \
    git -C fatrop checkout fatropv1-implicit && \
    git -C fatrop log -1 --format="fatrop at %h: %s"
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

# Support for implicit dynamics (Fatrop problem_type implicit_ocp_type)
COPY patches/casadi.patch /tmp/casadi.patch
RUN git apply /tmp/casadi.patch

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