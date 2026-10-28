# ONNX Runtime ARM64 distribution

- Version: 1.23.2, Linux AArch64, CPU inference.
- Copied unchanged from the source repository directory:
  `policy_deployment_ros2/thirdparty/onnxruntime-linux-aarch64-1.23.2`.
- Upstream: https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2
- Bundled `GIT_COMMIT_ID`: `a83fc4d58cb48eb68890dd689f94f28288cf2278`.
- Microsoft MIT `LICENSE`, `ThirdPartyNotices.txt`, headers, libraries and
  original relative library symlinks are preserved in the distribution.
- `lib/libonnxruntime.so.1.23.2` SHA256:
  `648ffa64fbe027ae27139109410900cf776a030dec2dbbac51053318cc44c286`.
- `lib/libonnxruntime_providers_shared.so` SHA256:
  `3b6be288fbfb7dff8770d08a23defdde18e8f7e0f5a2b344a0e5e238c999ea88`.

No runtime dependency on the original local directory is introduced. CMake
selects the bundled ARM64 distribution for `aarch64`/`arm64` targets. x86-64
continues to use the existing Unitree ONNX Runtime 1.22.0. This does not install
Python wheels or enable CUDA for camera inference.
