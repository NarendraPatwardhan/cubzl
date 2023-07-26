# --------------------------
# machinelearning-one/cubzl
# --------------------------
# Set the base image
# -------------------
ARG CUDA_VERSION=12.1.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04

# Set the working directory
# --------------------------
WORKDIR /app

# Copy the files
# ---------------
COPY ./matmul .

# Set the command
# ----------------
CMD ["./matmul"]