#include "global.h"	// include first in any cpp

#include "RhiTests.h"
#include "TestFramework.h"
#include "Buffer.h"
#include "DeferredRelease.h"

#include <array>
#include <glm/glm.hpp>

namespace tests
{
	static const char* kSuiteContext = "RhiContext";
	static const char* kSuiteBuffer = "RhiBuffer";

	//---- Partial init: instance + device + allocator, no window/surface/swapchain --------
	// Mirrors the first part of App::init() (App.cpp) up to but not
	// including glfwCreateWindowSurface() -- everything a headless RHI test needs and
	// nothing more. No device extensions are requested: Buffer only needs buffer device
	// address, which is core Vulkan 1.2 and already asserted-on unconditionally inside
	// Context::init() (extern/vk_framework.h initLogicalDevice), so an empty
	// ContextCreateInfo is enough.
	struct HeadlessRhi
	{
		utils::Context context;
		utils::ResourceAllocator allocator;
		VkCommandPool cmdPool = {};

		void Init()
		{
			// glfwInit() is required even headless: Context::initInstance() calls
			// glfwGetRequiredInstanceExtensions() unconditionally to build the instance
			// extension list.
			ASSERT(glfwInit() == GLFW_TRUE, "Could not initialize GLFW!");
			VK_CHECK(volkInitialize());

			// Validation layers left at their default (on in Debug): Buffer's two construction paths
			// (initRaw with data, init() without) are both VMA-backed now, so there's no manual
			// vkAllocateMemory/vkFreeMemory left in Buffer.cpp to leak -- this fixture runs clean
			// with validation on, and would catch it again if that ever regresses.
			utils::ContextCreateInfo contextConfig;

			context.init(contextConfig);

			allocator.init(VmaAllocatorCreateInfo{
				.physicalDevice = context.getPhysicalDevice(),
				.device = context.getDevice(),
				.instance = context.getInstance(),
				.vulkanApiVersion = std::min(context.getDeviceApiVersion(), VK_API_VERSION_1_4),
				});

			cmdPool = utils::createCommandPool(context.getDevice(), context.getGraphicsQueue().familyIndex,
				VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		}

		void Deinit()
		{
			vkDestroyCommandPool(context.getDevice(), cmdPool, nullptr);
			cmdPool = {};

			// allocator must be torn down before the device it was created against
			allocator.deinit();
			context.deinit();

			glfwTerminate();
		}

		// records+submits+waits a one-off command buffer, matching the
		// beginSingleTimeCommands/endSingleTimeCommands pattern used throughout
		// App.cpp (e.g. around line 918)
		template <class Fn>
		void RunOnceAndWait(Fn&& recordCommands)
		{
			VkCommandBuffer cmd = utils::beginSingleTimeCommands(context.getDevice(), cmdPool);
			recordCommands(cmd);
			utils::endSingleTimeCommands(cmd, context.getDevice(), cmdPool, context.getGraphicsQueue().queue);
		}
	};

	static void Test_HeadlessContextHasValidHandles(TestReport& report, HeadlessRhi& rhi)
	{
		report.Check(kSuiteContext, "context_device_is_valid", rhi.context.getDevice() != VK_NULL_HANDLE,
			"Context::init() with a minimal (no extensions) ContextCreateInfo must still produce a real VkDevice");
		report.Check(kSuiteContext, "context_physical_device_is_valid", rhi.context.getPhysicalDevice() != VK_NULL_HANDLE,
			"Context::init() must select a VkPhysicalDevice (picks the first discrete GPU it finds)");
		report.Check(kSuiteContext, "context_instance_is_valid", rhi.context.getInstance() != VK_NULL_HANDLE,
			"Context::init() must produce a real VkInstance");
		report.Check(kSuiteContext, "context_graphics_queue_is_valid", rhi.context.getGraphicsQueue().queue != VK_NULL_HANDLE,
			"getGraphicsQueue() must return a real VkQueue after init()");
	}

	// a Buffer that has been constructed but never init()'d must look exactly like one
	// that has been properly deinit()'d -- both are "no GPU resource owned" states, and
	// Buffer::~Buffer()'s own asserts rely on this (Buffer.cpp:9-17)
	static void Test_FreshBufferStartsInDeinitState(TestReport& report)
	{
		Buffer buffer(EF_Type_Buffer);

		bool looksEmpty = buffer.m_data.buffer == VK_NULL_HANDLE && buffer.m_data.allocation == nullptr
			&& buffer.m_data.address == 0 && buffer.m_size == 0 && buffer.m_stride == 0;

		report.Check(kSuiteBuffer, "fresh_buffer_starts_empty", looksEmpty,
			"a just-constructed Buffer (before init()) must have a null VkBuffer/allocation/address and zero size/stride");
	}

	static void Test_VertexBufferInitRaw(TestReport& report, HeadlessRhi& rhi)
	{
		// EF_Type_Buffer: IAppInit's flag for index/vertex/shader-code buffers (EventManager.h);
		// this also self-registers the Buffer into the global g_EventManager for its lifetime
		// (unregisters in ~Buffer), which is harmless here since the test never calls
		// g_EventManager.callAppInit/callAppDeInit.
		Buffer buffer(EF_Type_Buffer);

		std::array<float3, 3> triangle = { float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0) };

		rhi.RunOnceAndWait([&](VkCommandBuffer cmd)
			{
				buffer.initRaw(rhi.allocator, cmd, triangle.data(), triangle.size(), (uint32)sizeof(float3), Buffer::eBT_VertexBuffer);
			});
		// safe to free now: RunOnceAndWait already waited on a fence for the upload to finish
		rhi.allocator.freeStagingBuffers();

		report.Check(kSuiteBuffer, "vertex_buffer_handle_valid", buffer.m_data.buffer != VK_NULL_HANDLE,
			"initRaw(..., eBT_VertexBuffer) must produce a real VkBuffer");
		report.Check(kSuiteBuffer, "vertex_buffer_has_device_address", buffer.m_data.address != 0,
			"vertex/index buffers are created with SHADER_DEVICE_ADDRESS_BIT (Buffer.cpp:23), so address must be non-zero");
		report.Check(kSuiteBuffer, "vertex_buffer_size_matches_stride_times_count", buffer.m_size == sizeof(float3) * triangle.size(),
			"m_size must equal stride * count, got " + std::to_string(buffer.m_size) + " expected " + std::to_string(sizeof(float3) * triangle.size()));
		report.Check(kSuiteBuffer, "vertex_buffer_stride_matches", buffer.m_stride == sizeof(float3),
			"m_stride must equal sizeof(float3), got " + std::to_string(buffer.m_stride));

		EventContext context{ rhi.context.getDevice(), rhi.allocator };
		buffer.deinit(context);

		bool looksEmpty = buffer.m_data.buffer == VK_NULL_HANDLE && buffer.m_data.allocation == nullptr
			&& buffer.m_data.address == 0 && buffer.m_size == 0 && buffer.m_stride == 0;
		report.Check(kSuiteBuffer, "vertex_buffer_deinit_resets_state", looksEmpty,
			"after deinit(), a Buffer must return to the same all-zero state as a freshly constructed one");
	}

	static void Test_IndexBufferInitRaw(TestReport& report, HeadlessRhi& rhi)
	{
		Buffer buffer(EF_Type_Buffer);

		std::array<uint32, 3> indices = { 0, 1, 2 };

		rhi.RunOnceAndWait([&](VkCommandBuffer cmd)
			{
				buffer.initRaw(rhi.allocator, cmd, indices.data(), indices.size(), (uint32)sizeof(uint32), Buffer::eBT_IndexBuffer);
			});
		rhi.allocator.freeStagingBuffers();

		report.Check(kSuiteBuffer, "index_buffer_handle_valid", buffer.m_data.buffer != VK_NULL_HANDLE,
			"initRaw(..., eBT_IndexBuffer) must produce a real VkBuffer");
		report.Check(kSuiteBuffer, "index_buffer_has_device_address", buffer.m_data.address != 0,
			"index buffers are created with SHADER_DEVICE_ADDRESS_BIT (Buffer.cpp:27-28), so address must be non-zero");
		report.Check(kSuiteBuffer, "index_buffer_size_matches_stride_times_count", buffer.m_size == sizeof(uint32) * indices.size(),
			"m_size must equal stride * count");

		EventContext context{ rhi.context.getDevice(), rhi.allocator };
		buffer.deinit(context);
		report.Check(kSuiteBuffer, "index_buffer_deinit_resets_state", buffer.m_data.buffer == VK_NULL_HANDLE && buffer.m_size == 0,
			"after deinit(), the index buffer's handle and size must be reset to zero");
	}

	static void Test_ShaderBindingTableInitRaw(TestReport& report, HeadlessRhi& rhi)
	{
		// eBT_ShaderBindingTable is the odd one out in initRaw: it OVERWRITES the usage
		// flags instead of OR-ing them in (Buffer.cpp:35-38), so unlike vertex/index it does
		// NOT get VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT or the acceleration-structure-build
		// usage bit -- only SBT | SHADER_DEVICE_ADDRESS | TRANSFER_DST. This test documents
		// that distinction, not just that "some buffer got created".
		Buffer buffer(EF_Type_Buffer);

		std::array<uint8, 64> sbtBytes = {};

		rhi.RunOnceAndWait([&](VkCommandBuffer cmd)
			{
				buffer.initRaw(rhi.allocator, cmd, sbtBytes.data(), sbtBytes.size(), (uint32)sizeof(uint8), Buffer::eBT_ShaderBindingTable);
			});
		rhi.allocator.freeStagingBuffers();

		report.Check(kSuiteBuffer, "sbt_buffer_handle_valid", buffer.m_data.buffer != VK_NULL_HANDLE,
			"initRaw(..., eBT_ShaderBindingTable) must produce a real VkBuffer");
		report.Check(kSuiteBuffer, "sbt_buffer_has_device_address", buffer.m_data.address != 0,
			"SBT buffers request SHADER_DEVICE_ADDRESS_BIT directly (Buffer.cpp:37), so address must be non-zero");
		report.Check(kSuiteBuffer, "sbt_buffer_size_matches_byte_count", buffer.m_size == sbtBytes.size(),
			"m_size must equal stride(1) * count(64)");

		EventContext context{ rhi.context.getDevice(), rhi.allocator };
		buffer.deinit(context);
		report.Check(kSuiteBuffer, "sbt_buffer_deinit_resets_state", buffer.m_data.buffer == VK_NULL_HANDLE,
			"after deinit(), the SBT buffer's handle must be reset to VK_NULL_HANDLE");
	}

	static void Test_NoDataBufferInit(TestReport& report, HeadlessRhi& rhi)
	{
		// Buffer::init(allocator, size, usage, minAlignment) is the no-CPU-data-to-upload
		// construction path (pure GPU-side allocation; used by Renderer.cpp for
		// tlasBuffer/scratch and Mesh.cpp for BLAS/its scratch buffer) -- VMA-backed, same as
		// initRaw(). Notably it never sets m_size or m_stride (only initRaw does) -- documented
		// here so a future caller doesn't assume those fields are populated on this path.
		Buffer buffer(EF_Type_Buffer);

		const VkDeviceSize size = 256;
		buffer.init(rhi.allocator, size, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT);

		report.Check(kSuiteBuffer, "no_data_init_handle_valid", buffer.m_data.buffer != VK_NULL_HANDLE,
			"init(allocator, size, usage) must produce a real VkBuffer");
		report.Check(kSuiteBuffer, "no_data_init_has_device_address", buffer.m_data.address != 0,
			"init() always adds SHADER_DEVICE_ADDRESS_BIT to usage (Buffer.cpp), so address must be non-zero");
		report.Check(kSuiteBuffer, "no_data_init_has_vma_allocation", buffer.m_data.allocation != nullptr,
			"unlike the old manual/non-VMA path this used to be, init() now goes through allocator.createBuffer() "
			"like initRaw() does, so m_data.allocation must be set");
		report.Check(kSuiteBuffer, "no_data_init_does_not_set_size_or_stride", buffer.m_size == 0 && buffer.m_stride == 0,
			"unlike initRaw(), this overload never touches m_size/m_stride -- they stay 0 even though the buffer is really "
			+ std::to_string(size) + " bytes; callers on this path must track size themselves");

		EventContext context{ rhi.context.getDevice(), rhi.allocator };
		buffer.deinit(context);

		report.Check(kSuiteBuffer, "no_data_init_deinit_resets_handle", buffer.m_data.buffer == VK_NULL_HANDLE,
			"deinit() must reset m_data.buffer to VK_NULL_HANDLE immediately, even though the underlying GPU buffer's "
			"actual destruction is deferred (see DeferredRelease.h / Test_DeferredReleaseRetiresOnCompletedFrame)");
	}

	// Exercises DeferredReleaseManager directly (DeferredRelease.h): Buffer::deinit() no longer
	// destroys its GPU resource immediately -- it hands it to g_deferredRelease, stamped with
	// the frame it was released during, and the resource only actually gets destroyed once
	// Update() is told the GPU has reached (or passed) that frame. This is what makes
	// deinit()+init() safe to call every frame (e.g. rebuilding a TLAS each frame) with no
	// manual bookkeeping or hitches at the call site.
	static void Test_DeferredReleaseRetiresOnCompletedFrame(TestReport& report, HeadlessRhi& rhi)
	{
		// flush anything earlier tests left pending (e.g. Test_VertexBufferInitRaw's deinit(),
		// enqueued with whatever retire frame was current at the time) so the counts below are
		// exact rather than relative to unrelated leftover entries
		g_deferredRelease.Update(rhi.allocator, std::numeric_limits<uint64>::max());
		const size_t pendingBefore = g_deferredRelease.GetPendingCount();

		Buffer buffer(EF_Type_Buffer);
		buffer.init(rhi.allocator, 256, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT);

		g_deferredRelease.BeginFrame(100);
		EventContext context{ rhi.context.getDevice(), rhi.allocator };
		buffer.deinit(context);

		report.Check(kSuiteBuffer, "deferred_release_enqueues_instead_of_destroying",
			g_deferredRelease.GetPendingCount() == pendingBefore + 1,
			"deinit() during frame 100 must enqueue the buffer in g_deferredRelease, not destroy it immediately");

		g_deferredRelease.Update(rhi.allocator, 99);
		report.Check(kSuiteBuffer, "deferred_release_keeps_entry_before_its_frame_completes",
			g_deferredRelease.GetPendingCount() == pendingBefore + 1,
			"Update() with completedValue=99 must NOT free an entry released during frame 100 -- the GPU hasn't reached it yet");

		g_deferredRelease.Update(rhi.allocator, 100);
		report.Check(kSuiteBuffer, "deferred_release_frees_entry_once_its_frame_completes",
			g_deferredRelease.GetPendingCount() == pendingBefore,
			"Update() with completedValue=100 must free the entry -- the GPU has now reached the frame it was released during");
	}

	void RunRhiTests(TestReport& report)
	{
		HeadlessRhi rhi;
		rhi.Init();

		Test_HeadlessContextHasValidHandles(report, rhi);
		Test_FreshBufferStartsInDeinitState(report);
		Test_VertexBufferInitRaw(report, rhi);
		Test_IndexBufferInitRaw(report, rhi);
		Test_ShaderBindingTableInitRaw(report, rhi);
		Test_NoDataBufferInit(report, rhi);
		Test_DeferredReleaseRetiresOnCompletedFrame(report, rhi);

		// mirrors the real app's shutdown (App.cpp destroy()): free anything still
		// pending unconditionally before tearing down the allocator/device that owns it
		g_deferredRelease.deinit(rhi.allocator);

		rhi.Deinit();

		report.Check(kSuiteContext, "context_deinit_resets_device_handle", rhi.context.getDevice() == VK_NULL_HANDLE,
			"Context::deinit() resets *this to {} (extern/vk_framework.h), so getDevice() must return VK_NULL_HANDLE afterwards");
	}
}
