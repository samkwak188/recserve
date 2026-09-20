FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake ninja-build python3 ca-certificates
WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY apps/ apps/
COPY tests/ tests/
COPY data/quality_fixture.csv data/quality_fixture.csv
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4 && ctest --test-dir build --output-on-failure

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 python3 ca-certificates && useradd --uid 10001 --create-home recserve
COPY --from=build /src/build/recserve_server /usr/local/bin/recserve_server
COPY scripts/bundle.py /opt/recserve/bundle.py
USER 10001:10001
WORKDIR /data
EXPOSE 9400
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s --retries=3 CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9401/readyz',timeout=1).read()"]
ENTRYPOINT ["python3", "/opt/recserve/bundle.py", "serve", "--binary", "/usr/local/bin/recserve_server", "--bind", "0.0.0.0"]
CMD ["--manifest", "/data/wsl_small_bundle.json"]
