#pragma once

#include "vulkanBase.h"
#include "Buffer.h"  // for the shaderio namespace (VisualizeMode, used by m_visualizeMode)

//--- App ------------------------------------------------------------------------------------------------------------
// Main class for the sample. The application is the main class that is used to create the window,
// the Vulkan context, the swapchain, and the resources. See App.cpp for the member
// function bodies; only the class definition (declarations + member variables) lives here.
class App
{
public:
	App() = default;
	// @param hidden: skips ever showing the GLFW window (see main()'s own tests::WantsSelfTestMode
	// check) -- a real Vulkan device/swapchain/render loop still exists and runs exactly the same,
	// there's just nothing displayed on screen for it. Used by --selftest so a scripted run doesn't
	// pop up and steal focus; left false for the normal interactive app.
	App(VkExtent2D size = { 800, 600 }, bool hidden = false);
	~App();

	void run();

	// Deterministic GPU self-test (VVerse.exe --selftest, see tests::WantsSelfTestMode, TestMode.h):
	// this App is already fully init()'d by the time a caller can reach this (init() runs
	// at the end of the constructor, same as the normal interactive path) -- drives the SAME run()
	// loop through a fixed script (setup1/setup2/save/load/screenshot/compare, each feature used
	// twice -- see selfTestTick()'s own comment) via a hook called once per rendered frame, instead
	// of a separate render path.
	// @return process exit code (0 = every check passed).
	int runSelfTest();

private:
	// Called once per frame from run()'s own loop, only when g_selfTest.active -- advances the
	// self-test script by exactly one step every kSelfTestSettleFrames-th call (see its own
	// definition, App.cpp), closing the window (glfwSetWindowShouldClose) once the
	// script reaches its last step.
	void selfTestTick();

	// Main initialization sequence -- see the body's "Resource model" comment for why each
	// resource class uses the binding mechanism it does.
	void init();
	void destroy();

	// Creates a command pool (long life) and buffer for each frame in flight. Unlike the
	// temporary command pool, these pools persist between frames and don't use
	// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT. Each frame gets its own command buffer which records
	// all rendering commands for that frame.
	void createFrameSubmission(uint32 numFrames);

	// Main frame rendering function: records the ImGui UI, the ray tracing dispatch (if enabled)
	// and the offscreen graphics pass (via recordGraphicsCommands), then renders the RenderTarget
	// into the swapchain image through ImGui.
	void drawFrame(VkCommandBuffer cmd);

	// End the frame by submitting the command buffer to the GPU and presenting the image. Adds
	// binary semaphores to wait for the image to be available and signal when rendering is done,
	// plus the timeline semaphore to signal when the frame is completed. Moves to the next frame.
	void endFrame(VkCommandBuffer cmd);

#if USE_RAY_TRACING
	// (Re)allocates both g_outputBuffer slots if pixelCount exceeds the current capacity
	// (grow-only -- see onViewportSizeChange's call site for why), zero-filling each one via cmd
	// so an immediately-following frame with "Temporal Fade" turned up blends against defined
	// data instead of whatever was in the freshly (re)allocated GPU memory. No-op otherwise.
	// @param cmd: an already-recording command buffer; the caller submits it.
	void EnsureOutputBufferCapacity(VkCommandBuffer cmd, VkDeviceSize pixelCount);

	// (Re)creates g_restirDIContext and (re)allocates its buffers (grow-only, same philosophy as
	// EnsureOutputBufferCapacity) if width/height exceed the current RTXDI capacity. Manages its
	// own single-time command buffer internally (unlike EnsureOutputBufferCapacity) -- callers must
	// ensure no in-flight GPU work still references the old g_rtxdiReservoirs/g_rtxdiNormals before
	// calling this (see onViewportSizeChange's vkDeviceWaitIdle -- this must NOT be called mid-frame
	// from updateSceneBuffer() the way it originally was, which raced a still-executing previous
	// frame's ray dispatch against these buffers being freed/reallocated underneath it and reliably
	// caused VK_ERROR_DEVICE_LOST). No-op if width/height are already within capacity.
	void EnsureRtxdiCapacity(uint32 width, uint32 height);

	// (Re)creates ReSTIR GI's own reservoir buffer (g_giReservoirs) if width/height exceed current
	// capacity -- same grow-only philosophy and in-flight-safety requirement as EnsureRtxdiCapacity
	// (only call from init()/onViewportSizeChange(), never mid-frame). No-op if width/height are
	// already within capacity. Independent of EnsureRtxdiCapacity (own buffer, own capacity
	// tracking) since ReSTIR GI and RTXDI DI are otherwise unrelated features.
	void EnsureGIReservoirCapacity(uint32 width, uint32 height);

	// (Re)creates the G-buffer render targets (albedo, normal+roughness, view-space depth, motion
	// vector -- see the globals declared near g_gbufferAlbedo in App.cpp) if width/height
	// exceed current capacity, same grow-only philosophy and in-flight-safety requirement as
	// EnsureRtxdiCapacity (only call from init()/onViewportSizeChange(), never mid-frame). No-op if
	// width/height are already within capacity.
	void EnsureGBufferCapacity(uint32 width, uint32 height);

	// What interaction.slang's mouse-pick ray found, already resolved to the current frame-in-flight
	// slot -- see g_ReadbackMapped's own comment (App.cpp) for the readback mechanism this
	// hides. index == -1 means nothing is under the cursor (or Ctrl/LMB wasn't held); the caller
	// never needs to know about m_swapchain.getFrameResourceIndex() or the couple-frame read lag.
	struct PickResult
	{
		int32 index = -1;     // raw entity index (scene.m_sceneObjects) or material index (g_MaterialManager)
		bool isMaterial = false; // which of the two `index` is -- see interaction.slang's bMaterial
	};
	PickResult GetPickResult() const;
#endif

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

	// Recording the commands to render the scene (rasterizer pass + post pass). See
	// dispatchRayTracing() for the ray tracing dispatch, called from drawFrame() before this.
	void recordGraphicsCommands(VkCommandBuffer cmd);

#if USE_RAY_TRACING
	// Builds this frame's scene-level ray tracing state (the per-mesh {IB,VB,primitiveMaterials}
	// address buffer and the shared push data), then explicitly dispatches each ray tracing shader
	// by name (e.g. g_ray.dispatchRayTracing(...)), in the order written here -- no frame graph, no
	// automatic discovery. Adding another ray tracing shader means adding its own explicit call
	// here (see g_ray/g_gfx/g_post in App.cpp for the pattern); it can pass a different
	// pushData.outputAddress/prevOutputAddress to target a different output buffer while still
	// sharing pushData's sceneInfoAddress/meshBufferAddress/materialBufferAddress.
	void dispatchRayTracing(VkCommandBuffer cmd);

	// Explicitly (re)writes each ray tracing shader's descriptor set (TLAS, output image, scene
	// buffer, textures, sampler) for one frame-in-flight slot, by name (e.g.
	// g_ray.updateRayDescriptorSet(...)) -- they all read the same scene, per App's
	// single-scene assumption. Called every frame (the TLAS itself is rebuilt every frame, see
	// Renderer::createTopLevelAS) and, for every slot, right after a shader reload.
	// @param frameSlot: m_swapchain.getFrameResourceIndex(), same slot m_frameData is indexed by.
	void updateRayDescriptorSet(uint32 frameSlot);
#endif

	// Initialize Dear ImGui: creates the ImGui context and its GLFW/Vulkan backends, and sets up
	// docking.
	void initImGui();

	// Create descriptor heap buffers for bindless resource access (VK_EXT_descriptor_heap).
	// Instead of traditional descriptor sets and pools, the descriptor heap stores sampler and
	// image descriptors in GPU buffers. Shaders access them by index using layout(descriptor_heap).
	void createDescriptorHeap(VkCommandBuffer cmd);

	// Create a descriptor pool for ImGui. ImGui still uses traditional descriptor sets for its
	// font textures and RenderTarget display textures. The application's own textures/samplers
	// use the descriptor heap instead (see createDescriptorHeap).
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
	// here -- see the two Texture::loadFromFile calls in init() and the g_TextureManager cleanup
	// loop in destroy(). The two hand-authored test images always load first, so they claim
	// descriptor-heap slots 0/1 (== shaderio::SceneInfo::texId, the "Image 1"/"Image 2" debug
	// toggle below).
	// The sampler pool, used to create a sampler for the texture
	utils::SamplerPool m_samplerPool;

	VkSampler m_rtSampler{};

	// g_outputBuffer's allocated capacity in pixels; see onViewportSizeChange's grow-only comment.
	VkDeviceSize m_outputBufferCapacityPixels{};

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

	// Compute: traditional pipeline + layout (no descriptor heap access in compute).
	VkPipelineLayout m_computePipelineLayout{};
	VkPipeline       m_computePipeline{};
	// ImGui descriptor pool (ImGui still uses traditional descriptor sets)
	VkDescriptorPool m_uiDescriptorPool{};

	// Descriptor heap (VK_EXT_descriptor_heap): replaces traditional descriptor sets for textures/samplers.
	// GPU buffers holding sampler and image descriptors, bound per-frame via vkCmdBind*HeapEXT.
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
//	EVSyncMode m_vSyncMode = EVSyncMode::VSync;

	// The current image to display
	int m_imageID = 0;
	// Maximum textures allowed in the application
	uint32 m_maxTextures = 10000;

	friend void Reflection(CIniArchive& archive, WindowPersist& ref, App& app);
};
