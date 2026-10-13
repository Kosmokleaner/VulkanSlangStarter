// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Derived from src/minimal_latest.cpp in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#pragma once

#include "vulkanBase.h"
#include "Buffer.h"  // for the shaderio namespace (SceneInfo, GraphicsPushData)

//--- App ------------------------------------------------------------------------------------------------------------
// Main class for the sample. The application is the main class that is used to create the window,
// the Vulkan context, the swapchain, and the resources. See App.cpp for the member
// function bodies; only the class definition (declarations + member variables) lives here.
class App
{
public:
	App() = default;
	// @param size initial window size (client area)
	App(VkExtent2D size = { 800, 600 });
	~App();

	void run();

private:
	// Main initialization sequence -- see the body's "Resource model" comment for why each
	// resource class uses the binding mechanism it does.
	void init();
	void destroy();

	// Creates a command pool (long life) and buffer for each frame in flight. Unlike the
	// temporary command pool, these pools persist between frames and don't use
	// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT. Each frame gets its own command buffer which records
	// all rendering commands for that frame.
	void createFrameSubmission(uint32 numFrames);

	// Main frame rendering function: records the ImGui UI and the offscreen graphics pass (via
	// recordGraphicsCommands), then renders the RenderTarget into the swapchain image through ImGui.
	void drawFrame(VkCommandBuffer cmd);

	// End the frame by submitting the command buffer to the GPU and presenting the image. Adds
	// binary semaphores to wait for the image to be available and signal when rendering is done,
	// plus the timeline semaphore to signal when the frame is completed. Moves to the next frame.
	void endFrame(VkCommandBuffer cmd);

	// Call this function if the viewport size changes. This happens when the window is resized,
	// or when the ImGui viewport window is resized.
	void onViewportSizeChange(VkExtent2D size);

	// We are using dynamic rendering, which is a more flexible way to render to the swapchain
	// image. Only ImGui is rendered to the swapchain image. The scene is rendered to the
	// RenderTarget, and the rendered image is displayed using ImGui.
	void beginDynamicRenderingToSwapchain(VkCommandBuffer cmd) const;
	// End of dynamic rendering. The image is transitioned back to the present layout, and the
	// rendering is ended.
	void endDynamicRenderingToSwapchain(VkCommandBuffer cmd);

	// Update the scene information buffer (GPU-side) once per frame. The buffer is accessed in
	// the shader via its buffer device address (BDA). This keeps per-frame data in GPU memory,
	// avoiding redundant push data transfers.
	void updateSceneBuffer(VkCommandBuffer cmd) const;

	// Recording the commands to render the scene into the RenderTarget: the post shader's
	// fullscreen triangle (gradient + g_hndTex0 on top).
	void recordGraphicsCommands(VkCommandBuffer cmd);

	// Initialize Dear ImGui: creates the ImGui context and its GLFW/Vulkan backends, and sets up
	// docking.
	void initImGui();

	// Create descriptor heap buffers for bindless resource access (VK_EXT_descriptor_heap).
	// Instead of traditional descriptor sets and pools, the descriptor heap stores sampler and
	// image descriptors in GPU buffers. Bindless shaders access them by index. Also acquires
	// m_linearSampler for non-bindless shaders.
	void createDescriptorHeap(VkCommandBuffer cmd);

	// Create a descriptor pool for ImGui. ImGui still uses traditional descriptor sets for its
	// font textures and RenderTarget display textures. The application's own textures/samplers
	// use the descriptor heap (bindless) or each Shader's own descriptor pool (non-bindless).
	void createUIDescriptorPool();

	// Prepare frame resources - the first step in the rendering process. It looks if the
	// swapchain requires rebuild, which happens when the window is resized. It acquires the image
	// from the swapchain to render into. Returns true if we can proceed with rendering, false
	// otherwise.
	bool prepareFrameResources();

	// Begin command buffer recording for the frame. It resets the command pool to reuse the
	// command buffer for recording new rendering commands for the current frame. Returns the
	// command buffer for the frame.
	VkCommandBuffer beginCommandRecording();

	//--------------------------------------------------------------------------------------------------
	// The window
	GLFWwindow* m_window{};

	// The swapchain
	utils::Swapchain m_swapchain;
	// Scene info GPU buffer, updated once per frame, accessed via BDA
	utils::Buffer m_sceneInfoBuffer;
	// Loaded textures are Texture objects (Texture.h) tracked in g_TextureManager, not members
	// here -- see setupScene1() and the g_TextureManager cleanup loop in destroy().
	// The sampler pool, used to create a sampler for the texture
	utils::SamplerPool m_samplerPool;

	// linear filtering, repeat addressing; for non-bindless textures (see createDescriptorHeap)
	VkSampler m_linearSampler{};

	// Offscreen color + depth target rendered into and shown via ImGui::Image
	utils::RenderTarget m_renderTarget;
	// The ImGui image wrapper for the render target's color view
	utils::ImTexture m_imTexture;

	// The window surface
	VkSurfaceKHR m_surface{};
	// The window size
	VkExtent2D m_windowSize{ 800, 600 };
	// The viewport area in the window
	VkExtent2D m_viewportSize{ 800, 600 };

	// ImGui descriptor pool (ImGui still uses traditional descriptor sets)
	VkDescriptorPool m_uiDescriptorPool{};

	// Descriptor heap (VK_EXT_descriptor_heap): textures/samplers for bindless shaders.
	// GPU buffers holding sampler and image descriptors, bound per-frame via vkCmdBind*HeapEXT.
	// The resource heap lives in g_renderer.m_descriptorHeap; this is the sampler heap.
	// GPU buffer holding sampler descriptors
	utils::Buffer m_samplerHeapBuffer{};
	// Heap metadata needed for VkBindHeapInfoEXT at draw time
	// Total aligned size of the sampler heap buffer
	VkDeviceSize m_samplerHeapSize{};
	//  VkDeviceSize m_resourceHeapSize{};        // Total aligned size of the resource heap buffer
	// Byte offset where the sampler reserved range starts
	VkDeviceSize m_samplerReservedOffset{};
	// Size of the mandatory sampler reserved range
	VkDeviceSize m_samplerReservedSize{};

	// Frame resources and synchronization
	struct FrameData
	{
		// Command pool for recording commands for this frame
		VkCommandPool cmdPool;
		// Command buffer containing the frame's rendering commands
		VkCommandBuffer cmdBuffer;
		// Timeline value last signaled for this frame's resources
		uint64 lastSignalValue;
	};
	// Collection of per-frame resources to support multiple frames in flight
	std::vector<FrameData> m_frameData;
	// Timeline semaphore used to synchronize CPU submission with GPU completion
	VkSemaphore m_frameTimelineSemaphore{};
	// Monotonic timeline counter (increments each frame)
	uint64 m_frameCounter{ 1 };
	// The swapchain owns the in-flight slot index; query it via m_swapchain.getFrameResourceIndex().
	// Utility to pace the frame rate
	utils::FramePacer m_framePacer;

	EVSyncMode m_vSyncMode = EVSyncMode::TearFree;
};
