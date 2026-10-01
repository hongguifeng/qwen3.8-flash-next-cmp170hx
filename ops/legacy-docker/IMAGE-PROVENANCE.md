# 预构建镜像的还原凭据（镜像本体已于 2026-09-29 删除以释放磁盘）

本文件是 `18gogogo/170hx1-qwen38nextf:sm80` 的**可校验快照**：
原生部署（`vllm-native/`）就是从这个镜像里拷出来的，删掉镜像不影响运行；
万一将来需要回退 Docker（`ops/legacy-docker/run_container.sh`），按下面命令重新拉取即可。

## 重新拉取（需要网络；镜像仍在 Docker Hub）

```bash
# 按 digest 拉，可保证与当初用过的完全同一份：
docker pull 18gogogo/170hx1-qwen38nextf@sha256:9d8f3babf23360d46009f9c17a6747ce4c953e67169ffc4f9071a7ae5c338bd3
# 拉完校验（应输出同一个 ID）：
docker image inspect 18gogogo/170hx1-qwen38nextf:sm80 --format '{{.Id}}'
# 然后： ops/legacy-docker/run_container.sh
```

## 快照

| 项 | 值 |
|---|---|
| Tag | `18gogogo/170hx1-qwen38nextf:sm80` |
| Image ID | `sha256:9d8f3babf23360d46009f9c17a6747ce4c953e67169ffc4f9071a7ae5c338bd3` |
| RepoDigest | `18gogogo/170hx1-qwen38nextf@sha256:9d8f3babf23360d46009f9c17a6747ce4c953e67169ffc4f9071a7ae5c338bd3` |
| 创建时间 | 2026-09-26T05:06:30.010109221Z |
| 架构 | amd64 |
| 压缩大小 | 8.02 GiB |
| 层数 | 21 |
| 入口 | `["/opt/entrypoint.sh"]` |
| 工作目录 | `None` |

## 层 digest（可逐层比对）

```
sha256:f103cd120fdd6cbc7f3d0e1d5bfb3e16290c32ead7b822d15c34460ecdf64b29
sha256:a9cc1c90f1c1812263f7395267f109af8b70082d274df55ee16ec8ca9b50c3cb
sha256:70d00315500b1f9a6f12573c859ffaba655f84524b9669278c5c366fce2e9b46
sha256:f8a73e269513e40b106af860a8f43f68ad63c5864e898ebadd89eb9ac0bc1659
sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef
sha256:660c624ed9d916a585112fea809842b456d68f691bef3ba236d70ef001eef293
sha256:16cbecec273981d0fdd8614381a6137c08f560715d09fdfc8e7b7d2ed4e2491c
sha256:988348b2a2cc7aeef3c551ea2b082caca16615a381161e8f382b85e3bef01e10
sha256:23bc2e99562ef8e5e8bf3b8877532aa80facd75f091e985c26405ab5eeaec027
sha256:5c52383fbf034c3da1a5931fed626ad088a13b41b85e20aba525da6abc3c5502
sha256:cb09ec0dd1633f27b9e6dacad4425eadcddb71cb07e96361cf594caf7f9e29fa
sha256:81c0acf1115b7f04b1c227ba830d00cf59572cf549335b23bcbca72b31df01df
sha256:711fbfc92ac13bf13e5056dcdd56c7d86b87c09f1fc51cc3f2e643848e6c1e17
sha256:5345b745348700c57891ca521981d0a66091b6bb6e6235bae85386a37ce206cd
sha256:dd295fceac0d14df39d0c1470881583a05269f81a4d0e88e4ce3b81c1ba4fae7
sha256:d5b81965d01365fca21f19a741b6a3c34787d955a9c5ea88cdd8f8a02aa2461d
sha256:3a177511077949beaea6405c46be41ee34784d0151ab93e10ada77afcc9f8b2a
sha256:a90602c3397850fef554a9d560a34c402e76fc28600e1c8143269be2c146d6d3
sha256:487cdeca9da29215728af1a9de254a86790d1b9143c4124f938c33b9df733269
sha256:9ac662f460e81971e34cc163a589c3027c062d2dfee95b907d27614836da358f
sha256:2e3e652fa7f0750eba43069c0a0183d7dcaa03fd52763f1c7f867ff32ba12d98
```

## 校验提示

`docker pull` 后 `docker image inspect` 的 ID 若等于上表 Image ID，
则内容与当初完全一致（Docker 按内容寻址，ID 由配置+层 digest 决定）。
