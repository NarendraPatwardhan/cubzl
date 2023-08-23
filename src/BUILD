load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
    hdrs = ["kernel.h"],
)

cc_binary(
    name = "matmul",
    srcs = ["main.c"],
    deps = [":kernel"],
)
