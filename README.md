# cubzl

This repository aims to illustrate how to use [bazel](https://bazel.build/) to build a C++ project with CUDA support.

To build the project, run:

```bash
bazel build //src:matmul
```

To run the project, run:

```bash
bazel run //src:matmul
```

or directly:

```bash
./bazel-bin/src/matmul
```

To build the docker image, run:

```bash
cp ./bazel-bin/src/matmul .
DOCKER_BUILDKIT=1 docker build -t machinelearning-one/cubzl:latest .
```

To run the docker image, run:

```bash
docker run --rm -it --gpus all machinelearning-one/cubzl:latest
```