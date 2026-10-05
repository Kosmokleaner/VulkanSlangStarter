#include "global.h"
#include "DescriptorHeap.h"
#include "DeferredRelease.h"

uint32 DescriptorHeap::addTexture(VkImage image, VkFormat format, VkImageLayout layout, VkImageAspectFlags aspectMask, uint32 mipCount)
{
	ResourceDesc r{};
	r.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	r.image = image;
	r.format = format;
	r.layout = layout;
	r.aspectMask = aspectMask;
	r.mipCount = mipCount;

	m_resources.push_back(r);
	return uint32(m_resources.size() - 1);
}
uint32 DescriptorHeap::addCubeTexture(VkImage image, VkFormat format, VkImageLayout layout, uint32 mipCount)
{
	ResourceDesc r{};
	r.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	r.image = image;
	r.format = format;
	r.layout = layout;
	r.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	r.layerCount = 6;
	r.mipCount = mipCount;

	m_resources.push_back(r);
	return uint32(m_resources.size() - 1);
}
uint32 DescriptorHeap::addStorageImage(VkImage image, VkFormat format)
{
	ResourceDesc r{};
	r.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	r.image = image;
	r.format = format;
	r.layout = VK_IMAGE_LAYOUT_GENERAL;

	m_resources.push_back(r);
	return uint32(m_resources.size() - 1);
}

void DescriptorHeap::updateTexture(uint32 index, VkImage image, VkFormat format, VkImageLayout layout, VkImageAspectFlags aspectMask, uint32 mipCount)
{
	ASSERT(index < m_resources.size());
	ResourceDesc& r = m_resources[index];
	r.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	r.image = image;
	r.format = format;
	r.layout = layout;
	r.viewType = VK_IMAGE_VIEW_TYPE_2D;
	r.layerCount = 1;
	r.mipCount = mipCount;
	r.aspectMask = aspectMask;
}

void DescriptorHeap::updateStorageImage(uint32 index, VkImage image, VkFormat format)
{
	ASSERT(index < m_resources.size());
	ResourceDesc& r = m_resources[index];
	r.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	r.image = image;
	r.format = format;
	r.layout = VK_IMAGE_LAYOUT_GENERAL;
	r.viewType = VK_IMAGE_VIEW_TYPE_2D;
	r.layerCount = 1;
	r.mipCount = 1;
}

void DescriptorHeap::init(VkPhysicalDevice physicalDevice, VkCommandBuffer cmd)
{
	// Query the descriptor heap properties: descriptor sizes, alignment, and maximum heap capacities
	VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT };
	VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &heapProps };
	vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

	// All heap buffers need these usage flags: device address for binding, transfer dst for upload,
	// and the descriptor heap bit to mark them as heap storage
	VkBufferUsageFlags2 heapUsage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;

	// Compute how many descriptors the hardware can hold, accounting for the mandatory reserved range
//	uint32 maxSamplerCapacity = static_cast<uint32>(
//		(heapProps.maxSamplerHeapSize - heapProps.minSamplerHeapReservedRange) / heapProps.samplerDescriptorSize);
	uint32 maxImageCapacity = static_cast<uint32>(
		(heapProps.maxResourceHeapSize - heapProps.minResourceHeapReservedRange) / heapProps.imageDescriptorSize);

	// ---- Sampler Heap ----
	// We only need one sampler in this sample (linear filtering), but the heap can hold more
//	uint32 maxSamplers = std::min(1U, maxSamplerCapacity);
//	VkDeviceSize samplerHeapSize = heapProps.samplerDescriptorSize * maxSamplers + heapProps.minSamplerHeapReservedRange;
//	samplerHeapSize = utils::alignUp(samplerHeapSize, heapProps.samplerHeapAlignment);
//	std::vector<uint8> samplerHeapData(samplerHeapSize, 0);  // CPU staging buffer, zero-initialized

	m_heapAlignment = heapProps.resourceHeapAlignment;

	uint32 maxResources = std::min(m_maxTextures, maxImageCapacity);
	m_resourceReservedOffset = heapProps.imageDescriptorSize * maxResources;
	m_resourceReservedSize = heapProps.minResourceHeapReservedRange;
	m_imageDescriptorSize = heapProps.imageDescriptorSize;
}

void DescriptorHeap::build(VkDevice device, utils::ResourceAllocator& allocator, VkCommandBuffer cmd)
{
	// No early-out for an empty m_resources: vkCmdBindResourceHeapEXT's mandatory reserved range
	// (m_resourceReservedSize, from init()'s heapProps.minResourceHeapReservedRange) must still be
	// present in whatever's bound -- skipping the buffer entirely when there are 0 resources leaves
	// heapRange.size at 0 while reservedRangeSize stays whatever init() set it to, and the very next
	// bind call fails validation (VUID-vkCmdBindResourceHeapEXT-pBindInfo-11232: reservedRangeOffset
	// + reservedRangeSize > heapRange.size). resourceCount==0 below already degrades correctly on its
	// own -- a heap sized to just the reserved range, with 0 descriptors written.
	const uint32 resourceCount = uint32(m_resources.size());
	m_resourceReservedOffset = m_imageDescriptorSize * resourceCount;
	m_resourceHeapSize = utils::alignUp(m_resourceReservedOffset + m_resourceReservedSize, m_heapAlignment);

	std::vector<uint8> heapData(m_resourceHeapSize, 0);

	std::vector<VkImageViewCreateInfo> viewInfos(resourceCount);
	std::vector<VkImageDescriptorInfoEXT> imageInfos(resourceCount);
	std::vector<VkResourceDescriptorInfoEXT> resourceInfos(resourceCount);
	std::vector<VkHostAddressRangeEXT> dsts(resourceCount);

	for (uint32 i = 0; i < resourceCount; i++)
	{
		const ResourceDesc& r = m_resources[i];

		dsts[i] = {
			.address = heapData.data() + static_cast<size_t>(i) * static_cast<size_t>(m_imageDescriptorSize),
			.size = m_imageDescriptorSize,
		};

		switch (r.type)
		{
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		{
			viewInfos[i] = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = r.image,
				.viewType = r.viewType,
				.format = r.format,
				.subresourceRange = {r.aspectMask, 0, r.mipCount, 0, r.layerCount},
			};

			imageInfos[i] = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT,
				.pView = &viewInfos[i],
				.layout = r.layout,
			};

			resourceInfos[i] = {
				.sType = VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT,
				.type = r.type,
				.data = {.pImage = &imageInfos[i]},
			};
			break;
		}
		default:
			ASSERT(false, "Unsupported descriptor type");
		}
	}

	// vkWriteResourceDescriptorsEXT requires resourceCount > 0 -- nothing to write when there are
	// no resources yet, but the heap buffer below (sized to at least the mandatory reserved range)
	// must still be created so a later vkCmdBindResourceHeapEXT has something valid to bind.
	if (resourceCount)
		vkWriteResourceDescriptorsEXT(device, resourceCount, resourceInfos.data(), dsts.data());

	// All heap buffers need these usage flags: device address for binding, transfer dst for upload,
	// and the descriptor heap bit to mark them as heap storage
	VkBufferUsageFlags2 heapUsage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;

	// build() can run more than once now (e.g. after a scene reload re-registers textures). The
	// previous heap buffer may still be referenced by GPU work in flight (it's bound once per frame
	// via vkCmdBindResourceHeapEXT), so it can't be destroyed immediately -- destroying a buffer
	// still in use is invalid Vulkan usage (VUID-vkDestroyBuffer-buffer-00922), caught by the
	// validation layer. Defer it the same way Buffer::deinit() does, or its VMA allocation leaks
	// instead (caught at shutdown by vk_mem_alloc.h's "Some allocations were not freed" assert).
	g_deferredRelease.Release(m_resourceHeapBuffer);

	m_resourceHeapBuffer = allocator.createBufferAndUploadData(cmd, std::span(heapData), heapUsage, {}, m_heapAlignment);
	DBG_VK_NAME(m_resourceHeapBuffer.buffer);
}

void DescriptorHeap::deinit(EventContext& context)
{
	context.allocator.destroyBuffer(m_resourceHeapBuffer);  // Descriptor heap: resource (image) GPU buffer
	m_resourceHeapBuffer = {};
}
