
# VulkanSlangStarter V 1.0

A starting point for a Vulkan renderer by Martin Mittring

<img width="75%" alt="image" src="https://github.com/user-attachments/assets/fa9d0d22-9f62-4605-b0c4-8bf1f2c29e86" />

Features:
* Slang (HLSL like but allows for differential, compiles slower)
* Shader cache for faster startup (supporting includes, cache key is file content based, not date)
* Shader hot reload (press F5)
* ImGui (2D Immediate Mode GUI, Docking Branch)
* Texture support (non bindless)
* Uses ShaderToHuman to print text from inside the shader, see below
* Window and App icon (red triangle)

Platform: Windows11 x64
Tested on "NVIDIA GeForce RTX 2080", driver "610.188.0"

Command line parameters: --validate=0 / --validate=1 to turn the Vulkan validation layers off/on
(default: on in Debug, off in Release)

### todo (next ones on top):
* support Linux

### done (add at top, some element have been added at bottom):
* 10/12/2026 V1.0

===============================================

### extern (part of the repo unless stated)
* VulkanSDK 1.4.357.0 (not part of repo)
* glfw3.lib 3.3.4
* ImGUI (docking branch) v1.92.8
* stb_image 2.30
* Volk
* GLM
* VMA (Vulkan Memory Allocator)
* slang-2026.13.1-windows-x86_64
* [ShaderToHuman (S2H)](https://github.com/electronicarts/ShaderToHuman)
* hash-library (sha256 http://create.stephan-brumme.com/disclaimer.html)
 
