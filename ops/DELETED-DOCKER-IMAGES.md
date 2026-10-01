# 已删除的 Docker 镜像留档（2026-09-29）

用户要求删除这两个镜像（各自约 14.7 GB）。本文件记录它们的身份与“配方”，
以免将来需要时无从下手。**本文件与 hong-pc 的 `ops/legacy-docker/IMAGE-PROVENANCE.md` 是两回事**。

| 标签 | Image ID | 可恢复性 |
|---|---|---|
| `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` | `sha256:64f19b8e99cd31cc4f5083284c48a25c047c23836939289e80bd7278766596de` | ✅ ghcr.io 上仍有（`docker manifest inspect` 成功，匿名/本地凭据可拉） |
| `qwen38-27b-3090:latest` | `sha256:61a1291720d4e117352e83264b7c483d433eadc3445935498bf978bc0e78e7e6` | ❌ **无 registry 副本**（`manifest inspect` 失败）⇒ 删除后不可恢复，只能按下文“配方”重建 |

## 逐项留档

### `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest`

- Image ID: `sha256:64f19b8e99cd31cc4f5083284c48a25c047c23836939289e80bd7278766596de`
- RepoDigests: `["ghcr.io/syv-ai/qwen38-27b-rtx3090@sha256:64f19b8e99cd31cc4f5083284c48a25c047c23836939289e80bd7278766596de"]`
- Created: `2026-09-12T22:38:34.376451459+08:00` | Size: 4.27 GiB | 层数: 13
- 入口: `["bash", "docker/entrypoint.sh"]`  Cmd: `["single"]`
- WorkingDir: `/app`
- 关键环境变量: `["PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576", "NV_CUDA_CUDART_VERSION=13.0.88-1", "CUDA_VERSION=13.0.1", "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64", "NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "PYTHONUNBUFFERED=1", "VLLM_NO_USAGE_STATS=1", "HF_HUB_ENABLE_HF_TRANSFER=1"]`
- Labels: `{"com.docker.compose.project": "qwen38-27b-rtx3090", "com.docker.compose.service": "single", "com.docker.compose.version": "2.40.3", "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>", "org.opencontainers.image.ref.name": "ubuntu", "org.opencontainers.image.version": "24.04"}`
- RootFS 层 digest（13 个，末尾 5 个）:
```
sha256:d806e26feceaf4f9cc81158040c89bdeb63942b2c7368a046b2d27aafa41ca38
sha256:9b750ed6999c9031ef8149ea226702f888241bfc582dd8104b7496c801d2e658
sha256:67ea67a4c7af7bf8177959f1b879a01c68b490d81dc7d2c759f230f1c2fcf601
sha256:0b0386d27004164f0c0f292e1883adf26ccaaafc481ca789d4191139423fd27d
sha256:ffa65d0f88b954acbfa82d8cd6518a88e601d002d80d2b707d62b13645bf4540
```

### `qwen38-27b-3090:latest`

- Image ID: `sha256:61a1291720d4e117352e83264b7c483d433eadc3445935498bf978bc0e78e7e6`
- RepoDigests: `["qwen38-27b-3090@sha256:61a1291720d4e117352e83264b7c483d433eadc3445935498bf978bc0e78e7e6"]`
- Created: `2026-08-26T03:14:01.394741478+08:00` | Size: 4.26 GiB | 层数: 13
- 入口: `["bash", "docker/entrypoint.sh"]`  Cmd: `["single"]`
- WorkingDir: `/app`
- 关键环境变量: `["PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 brand=nvidia,driver>=535,driver<536 brand=quadro,driver>=535,driver<536 brand=quadrortx,driver>=535,driver<536 brand=nvidiartx,driver>=535,driver<536 brand=vapps,driver>=535,driver<536 brand=vpc,driver>=535,driver<536 brand=vcs,driver>=535,driver<536 brand=vws,driver>=535,driver<536 brand=cloudgaming,driver>=535,driver<536 brand=unknown,driver>=550,driver<551 brand=grid,driver>=550,driver<551 brand=tesla,driver>=550,driver<551 brand=nvidia,driver>=550,driver<551 brand=quadro,driver>=550,driver<551 brand=quadrortx,driver>=550,driver<551 brand=nvidiartx,driver>=550,driver<551 brand=vapps,driver>=550,driver<551 brand=vpc,driver>=550,driver<551 brand=vcs,driver>=550,driver<551 brand=vws,driver>=550,driver<551 brand=cloudgaming,driver>=550,driver<551 brand=unknown,driver>=565,driver<566 brand=grid,driver>=565,driver<566 brand=tesla,driver>=565,driver<566 brand=nvidia,driver>=565,driver<566 brand=quadro,driver>=565,driver<566 brand=quadrortx,driver>=565,driver<566 brand=nvidiartx,driver>=565,driver<566 brand=vapps,driver>=565,driver<566 brand=vpc,driver>=565,driver<566 brand=vcs,driver>=565,driver<566 brand=vws,driver>=565,driver<566 brand=cloudgaming,driver>=565,driver<566 brand=unknown,driver>=570,driver<571 brand=grid,driver>=570,driver<571 brand=tesla,driver>=570,driver<571 brand=nvidia,driver>=570,driver<571 brand=quadro,driver>=570,driver<571 brand=quadrortx,driver>=570,driver<571 brand=nvidiartx,driver>=570,driver<571 brand=vapps,driver>=570,driver<571 brand=vpc,driver>=570,driver<571 brand=vcs,driver>=570,driver<571 brand=vws,driver>=570,driver<571 brand=cloudgaming,driver>=570,driver<571 brand=unknown,driver>=575,driver<576 brand=grid,driver>=575,driver<576 brand=tesla,driver>=575,driver<576 brand=nvidia,driver>=575,driver<576 brand=quadro,driver>=575,driver<576 brand=quadrortx,driver>=575,driver<576 brand=nvidiartx,driver>=575,driver<576 brand=vapps,driver>=575,driver<576 brand=vpc,driver>=575,driver<576 brand=vcs,driver>=575,driver<576 brand=vws,driver>=575,driver<576 brand=cloudgaming,driver>=575,driver<576", "NV_CUDA_CUDART_VERSION=13.0.88-1", "CUDA_VERSION=13.0.1", "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64", "NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "PYTHONUNBUFFERED=1", "VLLM_NO_USAGE_STATS=1", "HF_HUB_ENABLE_HF_TRANSFER=1"]`
- Labels: `{"com.docker.compose.project": "qwen38-27b-rtx3090", "com.docker.compose.service": "single", "com.docker.compose.version": "2.40.3", "maintainer": "NVIDIA CORPORATION <cudatools@nvidia.com>", "org.opencontainers.image.ref.name": "ubuntu", "org.opencontainers.image.version": "24.04"}`
- RootFS 层 digest（13 个，末尾 5 个）:
```
sha256:9ddecbd55cf13e943a5c525531946f4422a9562466f8d98fe8d274681ddcdab2
sha256:f5d97f572f420cda3bce67acacd1be11cd1b2f32a13eff1b09903cfc53a223d9
sha256:d7d49141a5640d133206e29ddb2af5c46d1d81dbd7906761a8e6d83357e61371
sha256:dd921d2e59f17c342aa48c95f3e6d4534ef35ff2d8a884a14484e96cd99fce36
sha256:61f83643c535e66f610176856c13f49f3c8b664a43e61658a98e714299a65c8b
```

## 重建线索

- 两个镜像都是 Ubuntu 24.04 基座（`LABEL org.opencontainers.image.version=24.04`）；
  `qwen38-27b-3090`（2026-08-26）比 ghcr 那份（2026-09-12）早两周，)。
- 完整构建步骤（`docker history --no-trunc`，从新到旧）见下：

### `ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` 的 history（前 25 条）
```
CMD ["single"]
ENTRYPOINT ["bash" "docker/entrypoint.sh"]
EXPOSE [18020/tcp]
VOLUME [/cache /app/models]
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
RUN /bin/sh -c mkdir -p /cache /app/models && chmod 1777 /cache # buildkit
RUN /bin/sh -c set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1);     for p in patches/*.patch; do       case "$p" in         patches/dflash2-backport.patch) echo "== skip $p (DFlash2 is native in vLLM 0.28.0)"; continue ;;       esac;       echo "== $p"; patch -p1 -d "$SP" < "$p";     done;     bash kvarn/install.sh;     bash verify.sh --install # buildkit
COPY . . # buildkit
RUN /bin/sh -c venv/bin/pip install -r docker/requirements.txt # buildkit
COPY docker/requirements.txt docker/requirements.txt # buildkit
RUN /bin/sh -c python3.12 -m venv venv && venv/bin/pip install --upgrade pip # buildkit
WORKDIR /app
RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends       python3.12 python3.12-venv python3.12-dev       cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0       build-essential patch curl ca-certificates     && rm -rf /var/lib/apt/lists/* # buildkit
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
COPY NGC-DL-CONTAINER-LICENSE / # buildkit
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN |1 TARGETARCH=amd64 /bin/sh -c echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-13-0=${NV_CUDA_CUDART_VERSION}     cuda-compat-13-0     && rm -rf /var/lib/apt/lists/* # buildkit
ENV CUDA_VERSION=13.0.1
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     gnupg2 curl ca-certificates &&     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub | apt-key add - &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list &&     apt-get purge --autoremove -y curl     && rm -rf /var/lib/apt/lists/* # buildkit
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
ARG TARGETARCH
```

### `qwen38-27b-3090:latest` 的 history（前 25 条）
```
CMD ["single"]
ENTRYPOINT ["bash" "docker/entrypoint.sh"]
EXPOSE [18020/tcp]
VOLUME [/cache /app/models]
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
RUN /bin/sh -c mkdir -p /cache /app/models && chmod 1777 /cache # buildkit
RUN /bin/sh -c set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1);     for p in patches/*.patch; do echo "== $p"; patch -p1 -d "$SP" < "$p"; done;     bash kvarn/install.sh;     bash verify.sh --install # buildkit
COPY . . # buildkit
RUN /bin/sh -c venv/bin/pip install -r docker/requirements.txt # buildkit
COPY docker/requirements.txt docker/requirements.txt # buildkit
RUN /bin/sh -c python3.12 -m venv venv && venv/bin/pip install --upgrade pip # buildkit
WORKDIR /app
RUN /bin/sh -c apt-get update && apt-get install -y --no-install-recommends       python3.12 python3.12-venv python3.12-dev       cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0       build-essential patch curl ca-certificates     && rm -rf /var/lib/apt/lists/* # buildkit
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
COPY NGC-DL-CONTAINER-LICENSE / # buildkit
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN |1 TARGETARCH=amd64 /bin/sh -c echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-13-0=${NV_CUDA_CUDART_VERSION}     cuda-compat-13-0     && rm -rf /var/lib/apt/lists/* # buildkit
ENV CUDA_VERSION=13.0.1
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     gnupg2 curl ca-certificates &&     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub | apt-key add - &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list &&     apt-get purge --autoremove -y curl     && rm -rf /var/lib/apt/lists/* # buildkit
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
ARG TARGETARCH
```
