<div align="center">

# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Edge-LLM/)
[![version](https://img.shields.io/badge/release-0.5.0-green)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/tensorrt_edgellm/version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/LICENSE)

[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)

---
<div align="left">

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/quick-start-guide.html).

### Internal Packaging for Bazel `http_archive`

This repository includes `package_http_archive.sh` to package runtime libraries and headers into an archive for Bazel `http_archive` consumption.

Default behavior:
- Output layout: `include/tensorrt-edge-llm/**` and `lib/**`
- Includes all `.so/.so.*` and `.a` from `build/`
- Excludes `libexampleUtils.a` by default
- Preserves `.so` symlink chain and verifies links after extraction
- Prints archive path and `sha256`
- Uses bundle prefix `tensorrt-edge-llm-<arch>` where `arch` defaults to `x86_64`

Run in Docker (recommended in this workspace):

```bash
docker exec -u qcraft qcraft_dev_qcraft bash -lc \
  'bash /hosthome/Dev/TensorRT-Edge-LLM/package_http_archive.sh \
    --project-dir /hosthome/Dev/TensorRT-Edge-LLM \
    --output-dir /hosthome/Dev/TensorRT-Edge-LLM'
```

Include `libexampleUtils.a` only when needed:

```bash
docker exec -u qcraft qcraft_dev_qcraft bash -lc \
  'bash /hosthome/Dev/TensorRT-Edge-LLM/package_http_archive.sh \
    --project-dir /hosthome/Dev/TensorRT-Edge-LLM \
    --output-dir /hosthome/Dev/TensorRT-Edge-LLM \
    --include-example-utils'
```

Package for another architecture tag (for example `aarch64`):

```bash
docker exec -u qcraft qcraft_dev_qcraft bash -lc \
  'bash /hosthome/Dev/TensorRT-Edge-LLM/package_http_archive.sh \
    --project-dir /hosthome/Dev/TensorRT-Edge-LLM \
    --output-dir /hosthome/Dev/TensorRT-Edge-LLM \
    --arch aarch64'
```

`WORKSPACE` snippet (same style as existing third-party archives):

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

tensorrt_edge_llm_arch = "x86_64"
tensorrt_edge_llm_package_version = "x86_64_<timestamp-from-bundle-name>"

http_archive(
    name = "tensorrt_edge_llm_{}".format(tensorrt_edge_llm_arch),
    build_file = clean_dep("//third_party/tensorrt_edge_llm:tensorrt_edge_llm.{}.BUILD".format(tensorrt_edge_llm_arch)),
    sha256 = "<sha256-from-package-script>",
    urls = [
        "https://<your-oss-url>/tensorrt-edge-llm-{}.tar.gz".format(tensorrt_edge_llm_package_version),
    ],
    strip_prefix = "tensorrt-edge-llm-{}".format(tensorrt_edge_llm_package_version),
)
```

To match this naming, package with:
`--arch <x86_64|aarch64|...>`

If your workspace does not define `clean_dep`, use a plain label string for `build_file` instead.

Then in your Bazel target:

```python
deps = [
    "@tensorrt_edge_llm_<arch>//:headers",
    "@tensorrt_edge_llm_<arch>//:edgellm_plugin_so",
]
```

---

## Documentation

### Introduction

- **[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/overview.html)** - What is TensorRT Edge-LLM and key features
- **[Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/supported-models.html)** - Complete model compatibility matrix

### User Guide

- **[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/installation.html)** - Set up Python export pipeline and C++ runtime
- **[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/quick-start-guide.html)** - Run your first inference in ~15 minutes
- **[Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)** - End-to-end LLM, VLM, EAGLE, and LoRA workflows
- **[Input Format Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/input-format.html)** - Request format and specifications
- **[Chat Template Format](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/chat-template-format.html)** - Chat template configuration

### Developer Guide

#### Software Design

- **[Python Export Pipeline](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/python-export-pipeline.html)** - Model export and quantization
- **[Engine Builder](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/engine-builder.html)** - Building TensorRT engines
- **[C++ Runtime Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/cpp-runtime-overview.html)** - Runtime system architecture
  - [LLM Inference Runtime](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/llm-inference-runtime.html)
  - [LLM SpecDecode Runtime](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/llm-inference-specdecode-runtime.html)

#### Advanced Topics

- **[Customization Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html)** - Customizing TensorRT Edge-LLM for your needs
- **[TensorRT Plugins](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/tensorrt-plugins.html)** - Custom plugin development
- **[Tests](tests/)** - Comprehensive test suite for contributors

---

## Use Cases

**🚗 Automotive**
- In-vehicle AI assistants
- Voice-controlled interfaces
- Scene understanding
- Driver assistance systems

**🤖 Robotics**
- Natural language interaction
- Task planning and reasoning
- Visual question answering
- Human-robot collaboration

**🏭 Industrial IoT**
- Equipment monitoring with NLP
- Automated inspection
- Predictive maintenance
- Voice-controlled machinery

**📱 Edge Devices**
- On-device chatbots
- Offline language processing
- Privacy-preserving AI
- Low-latency inference

---

## Tech Blogs

*Coming soon*

Stay tuned for technical deep-dives, optimization guides, and deployment best practices.

---

## Latest News

* [01/05] 🚀 Accelerate AI Inference for Edge and Robotics with NVIDIA Jetson T4000 and NVIDIA JetPack 7.1 ✨ [➡️ link](https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1/)
* [01/05] 🚀 Accelerating LLM and VLM Inference for Automotive and Robotics with NVIDIA TensorRT Edge-LLM ✨ [➡️ link](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Full Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)
- **Examples**: [Code Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/getting-started/examples.html)
- **Roadmap**: [Developer Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---
