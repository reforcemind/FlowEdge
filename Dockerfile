FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=clang
ENV CXX=clang++
ENV FLOWEDGE_BUILD_DIR=/workspace/build

RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99disable-check-valid-until \
    && echo 'Acquire::Check-Date "false";' >> /etc/apt/apt.conf.d/99disable-check-valid-until \
    && apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gnupg \
    ninja-build \
    pybind11-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Noble's fallback clang 18 lacks C++23 std::expected in its default standard library.
RUN set -eux; \
    if curl --connect-timeout 10 --max-time 60 --retry 5 --retry-delay 5 --retry-all-errors \
      -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /tmp/llvm-snapshot.gpg.key \
      && gpg --dearmor --yes -o /usr/share/keyrings/apt.llvm.org.gpg /tmp/llvm-snapshot.gpg.key \
      && echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-23 main" \
        > /etc/apt/sources.list.d/llvm.list \
      && apt-get update \
      && apt-get install -y clang-23 clang-format-23 clang-tidy-23; then \
        compiler=clang-23; \
        compiler_plus=clang++-23; \
        formatter=clang-format-23; \
        tidy=clang-tidy-23; \
      else \
        rm -f /usr/share/keyrings/apt.llvm.org.gpg /etc/apt/sources.list.d/llvm.list; \
        apt-get update; \
        apt-get install -y clang clang-format clang-tidy; \
        compiler=gcc; \
        compiler_plus=g++; \
        formatter=clang-format; \
        tidy=clang-tidy; \
    fi; \
    rm -rf /var/lib/apt/lists/* /tmp/llvm-snapshot.gpg.key; \
    if [ "$compiler" != "clang" ]; then update-alternatives --install /usr/bin/clang clang "/usr/bin/$compiler" 100; fi; \
    if [ "$compiler_plus" != "clang++" ]; then update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/$compiler_plus" 100; fi; \
    if [ "$formatter" != "clang-format" ]; then update-alternatives --install /usr/bin/clang-format clang-format "/usr/bin/$formatter" 100; fi; \
    if [ "$tidy" != "clang-tidy" ]; then update-alternatives --install /usr/bin/clang-tidy clang-tidy "/usr/bin/$tidy" 100; fi; \
    update-alternatives --install /usr/bin/cc cc "/usr/bin/$compiler" 100; \
    update-alternatives --install /usr/bin/c++ c++ "/usr/bin/$compiler_plus" 100

WORKDIR /workspace

COPY . .

RUN mkdir -p models \
    && curl --retry 5 --retry-delay 5 --retry-all-errors -fsSL "https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors" \
      -o models/mamba_flow.safetensors

RUN cmake -S . -B "$FLOWEDGE_BUILD_DIR" -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/clang \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DFLOWEDGE_TESTS=ON \
      -DFLOWEDGE_BENCH=ON \
      -DFLOWEDGE_RELAY=ON \
      -DFLOWEDGE_PYTHON=ON \
    && cmake --build "$FLOWEDGE_BUILD_DIR" \
    && ctest --test-dir "$FLOWEDGE_BUILD_DIR" --output-on-failure \
    && "$FLOWEDGE_BUILD_DIR/flowedge_job_queue_bench" 100 \
    && "$FLOWEDGE_BUILD_DIR/flowedge_job_qos_bench" 10000 \
    && "$FLOWEDGE_BUILD_DIR/mamba_relay_stream" models/mamba_flow.safetensors \
    && "$FLOWEDGE_BUILD_DIR/flowedge_mamba_stream_bench" models/mamba_flow.safetensors 100 \
    && "$FLOWEDGE_BUILD_DIR/flowedge_worker_drain_bench" 100 \
    && "$FLOWEDGE_BUILD_DIR/action_delivery_sample" \
    && "$FLOWEDGE_BUILD_DIR/flowedge_action_delivery_bench" 100 \
    && FLOWEDGE_BUILD_DIR="$FLOWEDGE_BUILD_DIR" ./tools/scripts/relay_demo.sh models/mamba_flow.safetensors \
    && FLOWEDGE_BUILD_DIR="$FLOWEDGE_BUILD_DIR" ./tools/scripts/job_demo.sh models/mamba_flow.safetensors

RUN python3 -m pip install --break-system-packages . huggingface_hub \
    && python3 -c "import flowedge; print('import OK:', flowedge.Engine)" \
    && python3 -m pip install --break-system-packages --no-deps -e integrations/lerobot \
    && python3 -m unittest discover -s integrations/lerobot/tests -v \
    && python3 -m unittest discover -s test -p benchmark_artifact_test.py -v

CMD ["ctest", "--test-dir", "/workspace/build", "--output-on-failure"]
