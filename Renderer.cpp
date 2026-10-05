#include "global.h"	// include first in any cpp
#include "Renderer.h"
#include "Buffer.h"
#include "Scene.h"
#include "ShaderCache.h"
#include "RenderComponent.h"
#include "LightComponent.h"
#include "Texture.h"

#include "vulkanBase.h"
#include <glm/gtc/type_ptr.hpp> //  glm::value_ptr


#if USE_RAY_TRACING
void Renderer::createTopLevelAS(VulkanContext& vulkanContext, Scene& scene, uint32 frameSlot)
{
	if (scene.m_sceneObjects.empty())
		return;

	ASSERT(frameSlot < kMaxFramesInFlight);
	TlasSlot& slot = m_tlasSlots[frameSlot];

	// VkTransformMatrixKHR is row-major 3x4, glm::mat4 is column-major; transpose before memcpy.
	auto toTransformMatrixKHR = [](const glm::mat4& m) {
		VkTransformMatrixKHR t;
		memcpy(&t, glm::value_ptr(glm::transpose(m)), sizeof(t));
		return t;
		};

	// Build this frame's instance data for the TLAS (transforms may have changed since last frame)
	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
	tlasInstances.reserve(scene.m_sceneObjects.size());

//	for (const auto& instance : scene.m_sceneObjects)
	for (uint32 i = 0, count = scene.m_sceneObjects.size(); i < count; ++i)
	{
		const CEntityHandle entityHnd = scene.m_sceneObjects.GetHandleByIterator(i);
		const Entity& ref = scene.m_sceneObjects.GetElementByIterator(i);

		VkAccelerationStructureInstanceKHR asInstance{};
		VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };

		auto lightPtr = g_LightManager.GetElementPtr(ref.hndLight);
		
		// GetElementPtr() only compares Seed, not whether the slot is actually in use -- a stale
		// handle (e.g. a scene object referencing a procedural mesh like g_plane, which
		// saveMeshes() never serializes -- see Mesh::m_loadFromFile) whose Seed happens to match a
		// never-used slot's default Seed (0, see CObjManager::InitOnce) still "resolves" to a
		// non-null Mesh**, but the Mesh* it points to is the slot's default value, nullptr -- so
		// the pointee must be checked too, not just the pointer.
		auto meshPtr = g_MeshManager.GetElementPtr(ref.hndRenderComp);

		// bit 0: primary view, bit 1: secondary views
		asInstance.mask = 0xFF;

		if(!meshPtr)
		{
			extern CRenderCompHandle g_lightMesh;
			if(lightPtr)
			{
				meshPtr = g_MeshManager.GetElementPtr(g_lightMesh);
				// only render in primary view
				asInstance.mask = 0x1;
			}
		}

		if (!meshPtr)
			continue;

		addressInfo.accelerationStructure = meshPtr->m_BottomLevelAS;

		if (!addressInfo.accelerationStructure)
			continue;

		ASSERT(meshPtr->m_indices.m_data.address && meshPtr->m_vertices.m_data.address && meshPtr->m_primitiveMaterials.m_data.address);

		const VkDeviceAddress blasAddress = vkGetAccelerationStructureDeviceAddressKHR(vulkanContext.device, &addressInfo);

		asInstance.transform = toTransformMatrixKHR(ref.transform.GetObjectMatrix());  // Position of the instance
		// InstanceID() (gl_InstanceCustomIndexEXT) is the *entity's* raw index (scene.m_sceneObjects'
		// Elements[] position, what GetElementPtr()/GetHandleByIndex() index with) -- not the mesh's.
		// A ray tracing closest-hit shader that needs the mesh (ray.slang/laser.slang, for vertex/
		// material lookup) gets there with one indirection through App::dispatchRayTracing's
		// g_entityMeshIndexBuffer (entityIndex -> meshIndex), built the same "packed by iteration
		// order, correct only as long as nothing's ever been removed" way g_meshBuffer already is.
		// interaction.slang's pick ray needs no indirection at all: the entity index *is* the answer
		// its readback hands back to g_selection.
		asInstance.instanceCustomIndex = entityHnd.Index;
		asInstance.accelerationStructureReference = blasAddress;  // Address of the BLAS
		asInstance.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;  // No culling - double sided
		tlasInstances.emplace_back(asInstance);
	}

	const uint32 instanceCount = static_cast<uint32>(tlasInstances.size());
	const VkDeviceSize instanceDataSize = sizeof(VkAccelerationStructureInstanceKHR) * instanceCount;

	// First use of this slot (or the scene grew): (re)allocate its instances buffer, sized for the
	// current instance count -- grow-only, same convention as App::EnsureOutputBufferCapacity.
	if (slot.tlasInstancesBuffer.m_size < instanceDataSize)
	{
		slot.tlasInstancesBuffer.init(vulkanContext.allocator, instanceDataSize,
			VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);  // required by vkCmdUpdateBuffer below
	}

	// Update this slot's instance data every frame. vkCmdUpdateBuffer (not
	// createBufferAndUploadData) is used deliberately: the data is small (well under its 64KB
	// limit for any reasonable scene) and this way it needs no per-frame staging buffer
	// allocation/free -- it just becomes part of this frame's command buffer, which this slot's
	// own frame-in-flight wait already keeps safe to reuse (see TlasSlot's comment).
	ASSERT(instanceDataSize <= 65536);
	
	if(slot.tlasInstancesBuffer.m_data.buffer)
		vkCmdUpdateBuffer(vulkanContext.cmd, slot.tlasInstancesBuffer.m_data.buffer, 0, instanceDataSize, tlasInstances.data());

	// Make the instance data write visible to the acceleration structure build below. Access masks
	// passed explicitly: utils::inferAccessMaskFromStage (vk_framework.h) doesn't know about
	// VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR and asserts on any stage it can't
	// infer -- which fired every frame once this stopped being a one-time call.
	utils::cmdBufferMemoryBarrier(vulkanContext.cmd, slot.tlasInstancesBuffer.m_data.buffer,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);

	// Convert the instance information to acceleration structure geometry, similar to primitiveToGeometry()
	VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
																	  .data = {.deviceAddress = slot.tlasInstancesBuffer.m_data.address} };
	VkAccelerationStructureGeometryKHR asGeometry{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
						.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
						.geometry = {.instances = geometryInstances} };

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &asGeometry;

	// asInfo.size needs the actual required size, which is only known after querying the build
	// sizes -- done unconditionally (not just on first use), since the instance count can grow
	// across calls (e.g. "setup #1" clicked more than once, adding more scene objects each time)
	// and the existing slot.tlas/tlasBuffer may no longer be big enough for it.
	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(vulkanContext.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount, &sizeInfo);

	// First use of this slot, or the scene grew since it was last sized: (re)create its TLAS
	// storage buffer + acceleration structure object. Grow-only, same convention as
	// tlasInstancesBuffer above -- rebuilt in place (same handle, same buffer) every frame this
	// doesn't trigger. Leaving the old slot.tlas handle undestroyed here would leak it (caught by
	// the validation layer at vkDestroyDevice, same as every other such leak fixed this session);
	// slot.tlasBuffer.init() below already defers freeing the old buffer safely (Buffer::init()).
	if (slot.tlas == VK_NULL_HANDLE || slot.tlasBuffer.m_size < sizeInfo.accelerationStructureSize)
	{
		if (slot.tlas != VK_NULL_HANDLE)
		{
			vkDestroyAccelerationStructureKHR(vulkanContext.device, slot.tlas, nullptr);
			slot.tlas = VK_NULL_HANDLE;
		}

		// Create TLAS storage buffer, sized from the query above
		slot.tlasBuffer.init(vulkanContext.allocator, sizeInfo.accelerationStructureSize,
			VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

		VkAccelerationStructureCreateInfoKHR asInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		asInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		asInfo.buffer = slot.tlasBuffer.m_data.buffer;
		asInfo.size = sizeInfo.accelerationStructureSize;

		ASSERT(asInfo.buffer);
		VK_CHECK(vkCreateAccelerationStructureKHR(vulkanContext.device, &asInfo, nullptr, &slot.tlas));

		// Scratch buffer, only needed during the build below. vkCmdBuildAccelerationStructuresKHR
		// requires buildInfo.scratchData.deviceAddress to be a multiple of
		// minAccelerationStructureScratchOffsetAlignment (VkPhysicalDeviceAccelerationStructurePropertiesKHR);
		// pass it straight through to Buffer::init()'s minAlignment, which forwards it to VMA.
		VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
		{
			VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
			props2.pNext = &asProps;
			vkGetPhysicalDeviceProperties2(vulkanContext.physicalDevice, &props2);
		}

		slot.scratch.init(vulkanContext.allocator, sizeInfo.buildScratchSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
			asProps.minAccelerationStructureScratchOffsetAlignment);
	}

	buildInfo.dstAccelerationStructure = slot.tlas;
	buildInfo.scratchData.deviceAddress = slot.scratch.m_data.address;

	VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{ .primitiveCount = instanceCount };
	const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &asBuildRangeInfo;

	vkCmdBuildAccelerationStructuresKHR(vulkanContext.cmd, 1, &buildInfo, &rangePtr);

	VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
	barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
	barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

	vkCmdPipelineBarrier(
		vulkanContext.cmd,
		VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		0,
		1, &barrier,
		0, nullptr,
		0, nullptr);
}
/*
void Renderer::createTopLevelAS(VulkanContext& vulkanContext, Scene& scene)
{
	if (m_tlas != VK_NULL_HANDLE || tlasInstancesBuffer.m_data.buffer != VK_NULL_HANDLE
		|| tlasBuffer.m_data.buffer != VK_NULL_HANDLE || scratch.m_data.buffer != VK_NULL_HANDLE)
	{
		vkQueueWaitIdle(vulkanContext.queue);

		if (m_tlas != VK_NULL_HANDLE)
		{
			vkDestroyAccelerationStructureKHR(vulkanContext.device, m_tlas, nullptr);
			m_tlas = {};
		}

		if (tlasInstancesBuffer.m_data.buffer != VK_NULL_HANDLE)
			tlasInstancesBuffer.deinit(vulkanContext.device, vulkanContext.allocator);
		if (tlasBuffer.m_data.buffer != VK_NULL_HANDLE)
			tlasBuffer.deinit(vulkanContext.device, vulkanContext.allocator);
		if (scratch.m_data.buffer != VK_NULL_HANDLE)
			scratch.deinit(vulkanContext.device, vulkanContext.allocator);
	}

	const uint32 instanceCount = (uint32)scene.m_sceneObjects.size();

	// VkTransformMatrixKHR is row-major 3x4, glm::mat4 is column-major; transpose before memcpy.
	auto toTransformMatrixKHR = [](const glm::mat4& m)
	{
		auto tm = glm::transpose(m);

		VkTransformMatrixKHR t;
		memcpy(&t, &tm, sizeof(t));
		return t;
	};

	// Prepare instance data for TLAS
	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
	tlasInstances.reserve(instanceCount);					// todo: bad runtime allocs

	uint32 instanceCustomIndex = 0;
	for(auto it = scene.m_sceneObjects.begin(), end = scene.m_sceneObjects.end(); it != end; ++it, ++instanceCustomIndex)
	{
		// todo
		extern Mesh g_mesh;

		VkAccelerationStructureInstanceKHR asInstance{};
		VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
		addressInfo.accelerationStructure = g_mesh.m_BottomLevelAS;
		const VkDeviceAddress blasAddress = vkGetAccelerationStructureDeviceAddressKHR(vulkanContext.device, &addressInfo);

//		glm::mat4 transform = glm::transpose(it->transform.GetViewMatrix());

		glm::mat4 transform = glm::mat4(1.0f);
//		transform[3] = float4(it->transform.GetPos(), 1.0f);
		asInstance.transform = toTransformMatrixKHR(transform);

		asInstance.instanceCustomIndex = instanceCustomIndex; // gl_InstanceCustomIndexEXT
		asInstance.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // No culling - double sided
		asInstance.mask = 0xFF;
		asInstance.accelerationStructureReference = blasAddress;
		tlasInstances.emplace_back(asInstance);
	}


	// Then create the buffer with the instance data and upload it
	{
		const VkDeviceSize instanceDataSize = sizeof(VkAccelerationStructureInstanceKHR) * tlasInstances.size();

		tlasInstancesBuffer.m_data = vulkanContext.allocator.createBufferAndUploadData(
			vulkanContext.cmd,
			std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances),
			VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
			VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);
		tlasInstancesBuffer.m_size = instanceDataSize;
		tlasInstancesBuffer.m_stride = sizeof(VkAccelerationStructureInstanceKHR);

		VkAccelerationStructureGeometryInstancesDataKHR instancesData {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		instancesData.arrayOfPointers = VK_FALSE;
		instancesData.data.deviceAddress = tlasInstancesBuffer.m_data.address;

		VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.geometry.instances = instancesData;

		VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geometry;

		VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };

		vkGetAccelerationStructureBuildSizesKHR(vulkanContext.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount, &sizeInfo);

		// Create TLAS storage buffer
		tlasBuffer.init(
			vulkanContext.physicalDevice,
			vulkanContext.device,
			sizeInfo.accelerationStructureSize,
			VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
			VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

		// Create TLAS object
		VkAccelerationStructureCreateInfoKHR asInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		asInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		asInfo.buffer = tlasBuffer.m_data.buffer;
		asInfo.size = sizeInfo.accelerationStructureSize;

		VK_CHECK(vkCreateAccelerationStructureKHR(vulkanContext.device, &asInfo, nullptr, &m_tlas));


		// Scratch buffer
//		Buffer scratch;
		scratch.init(vulkanContext.physicalDevice, vulkanContext.device, sizeInfo.buildScratchSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT);

		// Final build info
		buildInfo.dstAccelerationStructure = m_tlas;
		buildInfo.scratchData.deviceAddress = scratch.m_data.address;

		VkAccelerationStructureBuildRangeInfoKHR range{};
		range.primitiveCount = instanceCount;

		const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &range;

		// Build
		vkCmdBuildAccelerationStructuresKHR(vulkanContext.cmd, 1, &buildInfo, &rangePtr);

		VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

		vkCmdPipelineBarrier(
			vulkanContext.cmd,
			VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
			VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
			0,
			1,
			&barrier,
			0,
			nullptr,
			0,
			nullptr);

//		NVVK_CHECK(m_stagingUploader.appendBuffer(tlasInstancesBuffer, 0,
//			std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances)));
//		NVVK_DBG_NAME(tlasInstancesBuffer.buffer);
//		m_stagingUploader.cmdUploadAppended(vulkanContext.cmd);
//todo		m_app->submitAndWaitTempCmdBuffer(cmd);
	}

//	m_allocator.destroyBuffer(tlasInstancesBuffer);  // Cleanup
}
*/
#endif

Renderer::Renderer()
{

}

void Renderer::init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd)
{
	g_ShaderCache.Load();
	m_descriptorHeap.init(physicalDevice, cmd);

#if USE_RAY_TRACING
	m_rtProps = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
	VkPhysicalDeviceProperties2 props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProps };
	vkGetPhysicalDeviceProperties2(physicalDevice, &props);
#endif

	{
		VulkanContext vulkanContext(m_context.getDevice(), m_context.getPhysicalDevice(), m_allocator, cmd, m_context.getGraphicsQueue().queue, m_transientCmdPool);

		// Procedural, not loaded from disk: this used to be resources/ReplaceMe.png, but that file
		// doesn't exist in the repo -- Texture::loadFromFile() fails silent when stbi_load() returns
		// null (see its own "return; // failed" comment), so uploadPixels()/addTexture() never ran
		// and this texture never actually claimed descriptor-heap index 0 at all. The assert below
		// didn't catch it either -- hdn.Index is g_TextureManager's own handle index, unrelated to
		// m_heapIndex (the descriptor-heap slot addTexture() returns), so it stayed trivially true.
		// Whatever the next addTexture() call happened to be (e.g. a G-buffer/render-target/TAA
		// history registration -- none of which exist yet when this runs, but do the moment any of
		// them touches the heap before a real scene texture does) silently claimed slot 0 instead,
		// permanently breaking Shader::updateRayDescriptorSet's textureViews[0] guard (confirmed live
		// as vkCmdTraceRaysKHR's VUID-vkCmdTraceRaysKHR-None-08114, "g_tlas ... has never been
		// updated" -- that guard returns early forever once slot 0 stops being a real Texture2D). A
		// tiny in-memory checkerboard can't go missing, so this no longer depends on any file at all.
		const uint32 kMagenta = 0xFFFF00FFu; // little-endian RGBA8 (see Texture::loadFromMemory's own byte-order comment): R=FF G=00 B=FF A=FF
		const uint32 kBlack = 0xFF000000u;   // R=00 G=00 B=00 A=FF
		const uint32 pixels[4] = { kMagenta, kBlack, kBlack, kMagenta };

		Texture tex0;
		tex0.loadFromMemory(vulkanContext, pixels, 2, 2, "ReplaceMe (procedural fallback)");
		tex0.isEngineTexture = true;
		CTexHandle hdn = g_TextureManager.push_back(tex0);
		ASSERT(hdn.Index == 0);	// replaceme texture is fallback with index 0
	}
}

void Renderer::deinit(VkDevice device, utils::ResourceAllocator& allocator)
{
	ASSERT(device);

	EventContext context{ device, allocator };
	m_descriptorHeap.deinit(context);

	for (TlasSlot& slot : m_tlasSlots)
	{
		if (slot.tlas != VK_NULL_HANDLE)
		{
			vkDestroyAccelerationStructureKHR(device, slot.tlas, nullptr);
			slot.tlas = {};
		}

		slot.tlasInstancesBuffer.deinit(context);
		slot.tlasBuffer.deinit(context);
		slot.scratch.deinit(context);
	}

	g_ShaderCache.Save();
}

void Renderer::reloadShaders()
{
	Log("\n");

	VkDevice device = m_context.getDevice();
	VK_CHECK(vkDeviceWaitIdle(device));

	EventContext context{ m_context.getDevice(), m_allocator };
	g_EventManager.callReloadShaders(context);

}

void Renderer::createTransientCommandPool()
{
	m_transientCmdPool = utils::createCommandPool(m_context.getDevice(), m_context.getGraphicsQueue().familyIndex,
		VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
	DBG_VK_NAME(m_transientCmdPool);
}
