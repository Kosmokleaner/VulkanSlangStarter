
# VulkanSlangStarter V 0.9

A starting point for a Vulkan renderer by Martin Mittring

Features:
* Shader cache for faster startup (supporting includes, cache key is file content based, date)
* Slang (HLSL like but allows for differential, compiles slower)
* ImGui (2D Immediate Mode GUI)

Platform: Windows11 x64, later Linux

Run with commandline parameter: --test for running the headless (no GPU) tests, --selftest for the
scripted real-GPU/screenshot-compare tests (see TestMode.h)

### todo (next ones on top):
*

### done (add at top, some element have been added at bottom):
*

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
 
### random notes
* filesystem requires c++ 17+
* span requires c++ 20+
* RenderDoc does not support descriptorheap yet so cannot use it

### useful references
*
