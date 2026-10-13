
# VulkanSlangStarter V 1.0

A minimal starting point for a Vulkan renderer with ShaderToHuman and Slang shader hot reload by Martin Mittring.

<img width="75%" alt="image" src="https://github.com/user-attachments/assets/fa9d0d22-9f62-4605-b0c4-8bf1f2c29e86" />

Features:
* Slang (HLSL like but allows for differential, compiles slower)
* Shader cache for faster startup (supporting includes, cache key is file content based, not date)
* Shader hot reload (press F5)
* ImGui (2D Immediate Mode GUI, Docking Branch)
* Texture support (non bindless)
* Uses ShaderToHuman to print text from inside the shader, see below
* Window and App icon (red triangle)
* Screenshot function

Platform: Windows11 x64
Tested on "NVIDIA GeForce RTX 2080", driver "610.188.0"

Command line parameters: --validate=0 / --validate=1 to turn the Vulkan validation layers off/on
(default: on in Debug, off in Release)

### todo (next ones on top):
* ShaderToHuman more 2D examples
* ShaderToHuman 3D example
* ShaderToHuman Quad with VS and PS example
* Support Linux

### done (add at top, some element have been added at bottom):
* 10/12/2026 V1.0

===============================================

### extern (part of the repo unless stated)
* VulkanSDK 1.4.357.0 (not part of repo)
* glfw3.lib 3.3.4 (zlib license)
* ImGUI (docking branch) v1.92.8 (MIT)
* stb_image 2.30, stb_image_write 1.16 (public domain / MIT)
* Volk (MIT)
* GLM (MIT, comes with the VulkanSDK, not part of repo)
* VMA (Vulkan Memory Allocator) (MIT)
* slang-2026.13.1-windows-x86_64 (Apache-2.0 WITH LLVM-exception)
* [ShaderToHuman (S2H)](https://github.com/electronicarts/ShaderToHuman) in shaders/s2h (BSD-3-Clause, Electronic Arts)
* hash-library (sha256, zlib-like license, see http://create.stephan-brumme.com/disclaimer.html)
* vk_framework.h, debug_util.h, logger.h from [vk_minimal_latest](https://github.com/nvpro-samples/vk_minimal_latest) (Apache-2.0, NVIDIA), modified

### License
Apache-2.0, see [LICENSE](LICENSE). Copyright (c) 2026 Martin Mittring.

Based on NVIDIA's [vk_minimal_latest](https://github.com/nvpro-samples/vk_minimal_latest) sample
(Apache-2.0, Copyright (c) 2023-2026 NVIDIA CORPORATION), heavily modified. Each source file lists its
copyright holders in its SPDX header and says if it is derived from vk_minimal_latest.
resources/image1.jpg comes from vk_minimal_latest as well.
Third-party code in extern/ and shaders/s2h/ keeps its own license (see above).

