#include "global.h"	// include first in any cpp
#include "vulkanBase.h"
#include "NrdIntegration.h"
#include "Shader.h"
#include <cstdio>

namespace shaderio {  // Shader IO namespace -- shared layout between C++ and shaders
	using namespace glm;  // GLSL-style types without the glm:: prefix inside the namespace
#include "shaders/shader_io.h"
}  // namespace shaderio

extern Renderer g_renderer;

NrdDenoiser g_nrdDenoiser;

// nrd::Format -> VkFormat. Covers every entry NRD's public enum defines (not just the ones
// REBLUR_DIFFUSE's pools happen to use today), so this keeps working if a future NRD version or a
// different denoiser combination asks for a pool texture format not seen during this integration's
// own diagnostic dump (see NrdIntegration.h's class comment).
static VkFormat ToVkFormat(nrd::Format format)
{
	switch (format)
	{
	case nrd::Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
	case nrd::Format::R8_SNORM: return VK_FORMAT_R8_SNORM;
	case nrd::Format::R8_UINT: return VK_FORMAT_R8_UINT;
	case nrd::Format::R8_SINT: return VK_FORMAT_R8_SINT;
	case nrd::Format::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
	case nrd::Format::RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
	case nrd::Format::RG8_UINT: return VK_FORMAT_R8G8_UINT;
	case nrd::Format::RG8_SINT: return VK_FORMAT_R8G8_SINT;
	case nrd::Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
	case nrd::Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
	case nrd::Format::RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
	case nrd::Format::RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
	case nrd::Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
	case nrd::Format::R16_UNORM: return VK_FORMAT_R16_UNORM;
	case nrd::Format::R16_SNORM: return VK_FORMAT_R16_SNORM;
	case nrd::Format::R16_UINT: return VK_FORMAT_R16_UINT;
	case nrd::Format::R16_SINT: return VK_FORMAT_R16_SINT;
	case nrd::Format::R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
	case nrd::Format::RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
	case nrd::Format::RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
	case nrd::Format::RG16_UINT: return VK_FORMAT_R16G16_UINT;
	case nrd::Format::RG16_SINT: return VK_FORMAT_R16G16_SINT;
	case nrd::Format::RG16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
	case nrd::Format::RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
	case nrd::Format::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
	case nrd::Format::RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
	case nrd::Format::RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
	case nrd::Format::RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
	case nrd::Format::R32_UINT: return VK_FORMAT_R32_UINT;
	case nrd::Format::R32_SINT: return VK_FORMAT_R32_SINT;
	case nrd::Format::R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
	case nrd::Format::RG32_UINT: return VK_FORMAT_R32G32_UINT;
	case nrd::Format::RG32_SINT: return VK_FORMAT_R32G32_SINT;
	case nrd::Format::RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
	case nrd::Format::RGB32_UINT: return VK_FORMAT_R32G32B32_UINT;
	case nrd::Format::RGB32_SINT: return VK_FORMAT_R32G32B32_SINT;
	case nrd::Format::RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
	case nrd::Format::RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
	case nrd::Format::RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
	case nrd::Format::RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
	case nrd::Format::R10_G10_B10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	case nrd::Format::R10_G10_B10_A2_UINT: return VK_FORMAT_A2B10G10R10_UINT_PACK32;
	case nrd::Format::R11_G11_B10_UFLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	case nrd::Format::R9_G9_B9_E5_UFLOAT: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
	default:
		ASSERT(false && "NRD: unhandled nrd::Format");
		return VK_FORMAT_R8G8B8A8_UNORM;
	}
}

// Coarse-grained "wait for everything, block everything" compute<->compute/transfer/ray-tracing
// barrier. NRD's own dispatch sequence has real fine-grained per-resource dependencies (and so
// does the ray tracing pass that runs just before this), but tracking those precisely buys
// little here (this is one small denoiser pass among many GPU-bound stages) at real risk of a
// subtle missing-barrier bug -- a single conservative barrier between every step is simpler and
// provably correct, matching this integration's "keep it clean and simple" brief.
static void cmdFullBarrier(VkCommandBuffer cmd)
{
	const VkMemoryBarrier2 barrier{
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
	};
	const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier };
	vkCmdPipelineBarrier2(cmd, &depInfo);
}

void NrdDenoiser::init()
{
	VkDevice device = g_renderer.m_context.getDevice();

	{
		nrd::DenoiserDesc denoiserDesc{};
		denoiserDesc.identifier = 0;
		denoiserDesc.denoiser = nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;

		nrd::InstanceCreationDesc instanceCreationDesc{};
		instanceCreationDesc.denoisers = &denoiserDesc;
		instanceCreationDesc.denoisersNum = 1;

		nrd::Result result = nrd::CreateInstance(instanceCreationDesc, m_instance);
		if (result != nrd::Result::SUCCESS)
		{
			LOGE("NRD: CreateInstance failed (nrd::Result = %u)", (uint32)result);
			return;
		}
	}

	const nrd::LibraryDesc* libraryDesc = nrd::GetLibraryDesc();
	const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*m_instance);

	m_resourcesSetIndex = instanceDesc->resourcesSpaceIndex;
	m_constantsSetIndex = instanceDesc->constantBufferAndSamplersSpaceIndex;
	ASSERT(m_resourcesSetIndex < 2 && m_constantsSetIndex < 2 && m_resourcesSetIndex != m_constantsSetIndex);

	m_textureBinding = libraryDesc->spirvBindingOffsets.textureOffset;
	m_storageBinding = libraryDesc->spirvBindingOffsets.storageTextureAndBufferOffset;
	m_constantBufferBinding = libraryDesc->spirvBindingOffsets.constantBufferOffset;
	m_samplerBinding = libraryDesc->spirvBindingOffsets.samplerOffset;

	m_perSetTexturesMaxNum = instanceDesc->descriptorPoolDesc.perSetTexturesMaxNum;
	m_perSetStorageTexturesMaxNum = instanceDesc->descriptorPoolDesc.perSetStorageTexturesMaxNum;
	m_maxDispatchesPerFrame = instanceDesc->descriptorPoolDesc.setsMaxNum;

	// Two immutable samplers (NEAREST_CLAMP, LINEAR_CLAMP -- nrd::Sampler's only two values, in
	// that order), baked directly into m_constantsSetLayout below. Plain vkCreateSampler rather
	// than App's SamplerPool: that pool is a member of the app class, not reachable from
	// this free-standing global.
	{
		VkSamplerCreateInfo samplerCI{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};
		VK_CHECK(vkCreateSampler(device, &samplerCI, nullptr, &m_samplers[(uint32)nrd::Sampler::NEAREST_CLAMP]));

		samplerCI.magFilter = VK_FILTER_LINEAR;
		samplerCI.minFilter = VK_FILTER_LINEAR;
		samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		VK_CHECK(vkCreateSampler(device, &samplerCI, nullptr, &m_samplers[(uint32)nrd::Sampler::LINEAR_CLAMP]));
	}

	// --- NRD's own 2-descriptor-set layout + shared pipeline layout -------------------------
	{
		// NRD's SPIRV shaders bind each resource at its OWN sequential binding number -- e.g. the
		// 2nd texture input of some pipeline is "gIn_Normal_Roughness" at binding textureOffset+1 --
		// not one bindless array at a single binding (confirmed by a validation error naming an
		// exact binding/variable when this was first tried as one arrayed binding per type).
		// perSetTexturesMaxNum/perSetStorageTexturesMaxNum is the widest count any single pipeline
		// needs, so the shared layout declares that many individual bindings of each type; any one
		// dispatch's pipeline only statically references as many of them as it actually needs.
		std::vector<VkDescriptorSetLayoutBinding> resourceBindings;
		resourceBindings.reserve(m_perSetTexturesMaxNum + m_perSetStorageTexturesMaxNum);
		for (uint32 i = 0; i < m_perSetTexturesMaxNum; ++i)
			resourceBindings.push_back({ .binding = m_textureBinding + i, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
				.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT });
		for (uint32 i = 0; i < m_perSetStorageTexturesMaxNum; ++i)
			resourceBindings.push_back({ .binding = m_storageBinding + i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT });
		const VkDescriptorSetLayoutCreateInfo resourcesLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32)resourceBindings.size(), .pBindings = resourceBindings.data(),
		};
		VK_CHECK(vkCreateDescriptorSetLayout(device, &resourcesLayoutInfo, nullptr, &m_resourcesSetLayout));

		// UNIFORM_BUFFER_DYNAMIC, not plain UNIFORM_BUFFER: each of this frame's ~14 dispatches
		// needs different constant data, and rewriting one shared descriptor mid-frame is unsafe
		// (see the class comment) -- a dynamic offset into a per-frame ring buffer sidesteps that
		// without needing a fresh "constants" descriptor set per dispatch too.
		//
		// Each sampler gets its OWN binding (samplerOffset+i), not one binding with an array of
		// samplers -- same "individual sequential bindings, not one bindless array" scheme the
		// resources set uses (see its own comment); confirmed the same way, via a validation error
		// naming an exact binding/variable ("gLinearClamp" at binding samplerOffset+1) when this
		// was first tried as one arrayed sampler binding.
		std::vector<VkDescriptorSetLayoutBinding> constantsBindings;
		constantsBindings.push_back({ .binding = m_constantBufferBinding, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT });
		for (uint32 i = 0; i < (uint32)std::size(m_samplers); ++i)
			constantsBindings.push_back({ .binding = m_samplerBinding + i, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
				.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .pImmutableSamplers = &m_samplers[i] });
		const VkDescriptorSetLayoutCreateInfo constantsLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32)constantsBindings.size(), .pBindings = constantsBindings.data(),
		};
		VK_CHECK(vkCreateDescriptorSetLayout(device, &constantsLayoutInfo, nullptr, &m_constantsSetLayout));

		VkDescriptorSetLayout setLayouts[2]{};
		setLayouts[m_resourcesSetIndex] = m_resourcesSetLayout;
		setLayouts[m_constantsSetIndex] = m_constantsSetLayout;
		const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2, .pSetLayouts = setLayouts,
		};
		VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
	}

	// --- One VkPipeline per InstanceDesc::pipelines[] entry, from NRD's pre-embedded SPIR-V ---
	{
		m_pipelines.resize(instanceDesc->pipelinesNum);
		for (uint32 i = 0; i < instanceDesc->pipelinesNum; ++i)
		{
			const nrd::PipelineDesc& pipelineDesc = instanceDesc->pipelines[i];

			const VkShaderModuleCreateInfo moduleInfo{
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = pipelineDesc.computeShaderSPIRV.size,
				.pCode = (const uint32*)pipelineDesc.computeShaderSPIRV.bytecode,
			};
			VkShaderModule module{};
			VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));

			const VkPipelineShaderStageCreateInfo stageInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = instanceDesc->shaderEntryPoint,
			};
			const VkComputePipelineCreateInfo pipelineInfo{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stageInfo, .layout = m_pipelineLayout,
			};
			VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipelines[i]));

			vkDestroyShaderModule(device, module, nullptr);
		}
	}

	// --- Per-frame-slot constant buffer ring (one region per dispatch, dynamic-offset bound) ---
	{
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(g_renderer.m_context.getPhysicalDevice(), &props);
		const VkDeviceSize alignment = props.limits.minUniformBufferOffsetAlignment;
		m_constantBufferStride = (uint32)(((instanceDesc->constantBufferMaxDataSize + alignment - 1) / alignment) * alignment);

		const VkDescriptorPoolSize constantsPoolSizes[]{
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, Renderer::kMaxFramesInFlight },
			{ VK_DESCRIPTOR_TYPE_SAMPLER, (uint32)std::size(m_samplers) * Renderer::kMaxFramesInFlight },
		};
		const VkDescriptorPoolCreateInfo constantsPoolInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = Renderer::kMaxFramesInFlight,
			.poolSizeCount = (uint32)std::size(constantsPoolSizes), .pPoolSizes = constantsPoolSizes,
		};
		VK_CHECK(vkCreateDescriptorPool(device, &constantsPoolInfo, nullptr, &m_constantsPool));

		VkDescriptorSetLayout constantsLayouts[Renderer::kMaxFramesInFlight];
		for (uint32 i = 0; i < Renderer::kMaxFramesInFlight; ++i)
			constantsLayouts[i] = m_constantsSetLayout;
		const VkDescriptorSetAllocateInfo constantsAI{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = m_constantsPool,
			.descriptorSetCount = Renderer::kMaxFramesInFlight, .pSetLayouts = constantsLayouts,
		};
		VK_CHECK(vkAllocateDescriptorSets(device, &constantsAI, m_constantsSet));

		for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
		{
			m_constantBuffer[slot] = g_renderer.m_allocator.createBuffer(
				VkDeviceSize(m_constantBufferStride) * m_maxDispatchesPerFrame,
				VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

			// Immutable samplers (baked into the layout above) never need a descriptor write.
			const VkDescriptorBufferInfo constantBufferInfo{
				.buffer = m_constantBuffer[slot].buffer, .offset = 0, .range = m_constantBufferStride,
			};
			const VkWriteDescriptorSet write{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_constantsSet[slot],
				.dstBinding = m_constantBufferBinding, .descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &constantBufferInfo,
			};
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
		}
	}

	// --- Per-frame-slot "resources" descriptor pool -- fixed size, independent of viewport ---
	{
		for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
		{
			const VkDescriptorPoolSize resourcesPoolSizes[]{
				{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_perSetTexturesMaxNum * m_maxDispatchesPerFrame },
				{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_perSetStorageTexturesMaxNum * m_maxDispatchesPerFrame },
			};
			const VkDescriptorPoolCreateInfo resourcesPoolInfo{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = m_maxDispatchesPerFrame,
				.poolSizeCount = (uint32)std::size(resourcesPoolSizes), .pPoolSizes = resourcesPoolSizes,
			};
			VK_CHECK(vkCreateDescriptorPool(device, &resourcesPoolInfo, nullptr, &m_resourcesPool[slot]));
		}
	}

	// --- Bridge shaders (nrd_prepare/nrd_resolve): own tiny descriptor set layout/pipeline layout ---
	{
		// Binding 3 (SAMPLED_IMAGE): NRD's OUT_VALIDATION debug overlay -- only read by
		// nrd_resolve.slang (see its comment), unused/dummy for nrd_prepare.slang.
		// Binding 4 (SAMPLED_IMAGE): G-buffer albedo, read by both bridge shaders for albedo
		// demodulation (see nrd_prepare.slang/nrd_resolve.slang's header comments).
		// Binding 5 (STORAGE_IMAGE): specular counterpart of binding 2 -- prepare's write target for
		// IN_SPEC_RADIANCE_HITDIST, dummy (reuses the same view) for resolve.
		// Binding 6 (SAMPLED_IMAGE): specular counterpart of binding 0's "read the denoised result"
		// role -- OUT_SPEC_RADIANCE_HITDIST, read by resolve; dummy for prepare.
		const VkDescriptorSetLayoutBinding bridgeBindings[7]{
			{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
			{ .binding = 6, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		};
		const VkDescriptorSetLayoutCreateInfo bridgeLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32)std::size(bridgeBindings), .pBindings = bridgeBindings,
		};
		VK_CHECK(vkCreateDescriptorSetLayout(device, &bridgeLayoutInfo, nullptr, &m_bridgeSetLayout));

		const VkPushConstantRange bridgePushRange{
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::NrdBridgePushData),
		};
		const VkPipelineLayoutCreateInfo bridgePipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &m_bridgeSetLayout,
			.pushConstantRangeCount = 1, .pPushConstantRanges = &bridgePushRange,
		};
		VK_CHECK(vkCreatePipelineLayout(device, &bridgePipelineLayoutInfo, nullptr, &m_bridgePipelineLayout));

		// Reuses Shader::compile() purely as a ".slang -> SPIRV" utility (see NrdIntegration.h's
		// class comment) -- this integration writes its own vkCreateShaderModule/
		// vkCreateComputePipelines below instead of Shader's ray/graphics-pipeline machinery, which
		// has no compute-dispatch path at all (see readme.md's NRD section).
		auto createBridgePipeline = [&](const char* filePath, const char* entryName) -> VkPipeline
		{
			Shader shader(filePath, entryName);
			shader.compile();
			if (shader.code[0].empty())
			{
				LOGE("NRD: failed to compile bridge shader '%s'", filePath);
				return VkPipeline{};
			}

			const VkShaderModuleCreateInfo moduleInfo{
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = shader.code[0].size() * sizeof(uint32), .pCode = shader.code[0].data(),
			};
			VkShaderModule module{};
			VK_CHECK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module));

			const VkPipelineShaderStageCreateInfo stageInfo{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = entryName,
			};
			const VkComputePipelineCreateInfo pipelineInfo{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = stageInfo, .layout = m_bridgePipelineLayout,
			};
			VkPipeline pipeline{};
			VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

			vkDestroyShaderModule(device, module, nullptr);
			return pipeline;
		};
		m_preparePipeline = createBridgePipeline("shaders/nrd_prepare.slang", "main");
		m_resolvePipeline = createBridgePipeline("shaders/nrd_resolve.slang", "main");

		const VkDescriptorPoolSize bridgePoolSizes[]{
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 5 * 2 * Renderer::kMaxFramesInFlight },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * 2 * Renderer::kMaxFramesInFlight },
		};
		const VkDescriptorPoolCreateInfo bridgePoolInfo{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2 * Renderer::kMaxFramesInFlight,
			.poolSizeCount = (uint32)std::size(bridgePoolSizes), .pPoolSizes = bridgePoolSizes,
		};
		VK_CHECK(vkCreateDescriptorPool(device, &bridgePoolInfo, nullptr, &m_bridgePool));

		VkDescriptorSetLayout bridgeLayouts[2 * Renderer::kMaxFramesInFlight];
		for (uint32 i = 0; i < 2 * Renderer::kMaxFramesInFlight; ++i)
			bridgeLayouts[i] = m_bridgeSetLayout;
		VkDescriptorSet allBridgeSets[2 * Renderer::kMaxFramesInFlight]{};
		const VkDescriptorSetAllocateInfo bridgeAI{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = m_bridgePool,
			.descriptorSetCount = 2 * Renderer::kMaxFramesInFlight, .pSetLayouts = bridgeLayouts,
		};
		VK_CHECK(vkAllocateDescriptorSets(device, &bridgeAI, allBridgeSets));
		for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
		{
			m_prepareSet[slot] = allBridgeSets[slot * 2 + 0];
			m_resolveSet[slot] = allBridgeSets[slot * 2 + 1];
		}
		// Contents (which G-buffer/bridge-texture views these sets point at) are written by
		// ensureCapacity(), once viewport-sized textures exist -- not here.
	}

	LOGI("NRD: instance created (REBLUR_DIFFUSE_SPECULAR), %u pipelines, %u permanent + %u transient pooled textures",
		instanceDesc->pipelinesNum, instanceDesc->permanentPoolSize, instanceDesc->transientPoolSize);
}

void NrdDenoiser::deinit()
{
	VkDevice device = g_renderer.m_context.getDevice();
	if (!device)
		return;

	destroyPool(m_permanentPool);
	destroyPool(m_transientPool);
	{
		std::vector<PooledTexture> bridgeTextures{ m_inDiffRadianceHitDist, m_outDiffRadianceHitDist, m_validationTexture, m_inSpecRadianceHitDist, m_outSpecRadianceHitDist };
		destroyPool(bridgeTextures);
		m_inDiffRadianceHitDist = {};
		m_outDiffRadianceHitDist = {};
		m_validationTexture = {};
		m_inSpecRadianceHitDist = {};
		m_outSpecRadianceHitDist = {};
	}

	if (m_bridgePool) vkDestroyDescriptorPool(device, m_bridgePool, nullptr);
	if (m_bridgePipelineLayout) vkDestroyPipelineLayout(device, m_bridgePipelineLayout, nullptr);
	if (m_bridgeSetLayout) vkDestroyDescriptorSetLayout(device, m_bridgeSetLayout, nullptr);
	if (m_preparePipeline) vkDestroyPipeline(device, m_preparePipeline, nullptr);
	if (m_resolvePipeline) vkDestroyPipeline(device, m_resolvePipeline, nullptr);

	for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
		if (m_resourcesPool[slot]) vkDestroyDescriptorPool(device, m_resourcesPool[slot], nullptr);

	if (m_constantsPool) vkDestroyDescriptorPool(device, m_constantsPool, nullptr);
	for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
		if (m_constantBuffer[slot].buffer) g_renderer.m_allocator.destroyBuffer(m_constantBuffer[slot]);

	for (VkPipeline pipeline : m_pipelines)
		if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
	m_pipelines.clear();

	if (m_pipelineLayout) vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
	if (m_resourcesSetLayout) vkDestroyDescriptorSetLayout(device, m_resourcesSetLayout, nullptr);
	if (m_constantsSetLayout) vkDestroyDescriptorSetLayout(device, m_constantsSetLayout, nullptr);

	for (VkSampler sampler : m_samplers)
		if (sampler) vkDestroySampler(device, sampler, nullptr);

	if (m_instance)
	{
		nrd::DestroyInstance(*m_instance);
		m_instance = nullptr;
	}

	*this = NrdDenoiser();
}

void NrdDenoiser::createPooledTextures(VkCommandBuffer cmd, std::vector<PooledTexture>& pool, const nrd::TextureDesc* descs, uint32 count, VkExtent2D viewportSize, const char* debugPrefix)
{
	pool.resize(count);
	for (uint32 i = 0; i < count; ++i)
	{
		const nrd::TextureDesc& desc = descs[i];
		const VkExtent2D size{
			(viewportSize.width + desc.downsampleFactor - 1) / desc.downsampleFactor,
			(viewportSize.height + desc.downsampleFactor - 1) / desc.downsampleFactor,
		};
		const VkFormat format = ToVkFormat(desc.format);

		PooledTexture& target = pool[i];
		const VkImageCreateInfo imageInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = format,
			.extent = { size.width, size.height, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
			.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		};
		target.image = g_renderer.m_allocator.createImage(imageInfo);

		char name[128];
		snprintf(name, sizeof(name), "%s-%u", debugPrefix, i);
		utils::DebugUtil::getInstance().setObjectName(target.image.image, name);

		const VkImageViewCreateInfo viewInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = target.image.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format, .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
		};
		VK_CHECK(vkCreateImageView(g_renderer.m_context.getDevice(), &viewInfo, nullptr, &target.view));

		utils::cmdInitImageLayout(cmd, target.image.image);
		const VkClearColorValue clearValue{ { 0.0f, 0.0f, 0.0f, 0.0f } };
		const VkImageSubresourceRange range{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 };
		vkCmdClearColorImage(cmd, target.image.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
	}
}

void NrdDenoiser::destroyPool(std::vector<PooledTexture>& pool)
{
	VkDevice device = g_renderer.m_context.getDevice();
	for (PooledTexture& tex : pool)
	{
		if (tex.view != VK_NULL_HANDLE)
			vkDestroyImageView(device, tex.view, nullptr);
		if (tex.image.image != VK_NULL_HANDLE)
			g_renderer.m_allocator.destroyImage(tex.image);
	}
	pool.clear();
}

void NrdDenoiser::ensureCapacity(uint32 width, uint32 height, VkImageView gbufferNormalRoughnessView, VkImageView gbufferViewZView, VkImageView gbufferAlbedoView)
{
	if (!m_instance)
		return;

	if (m_inDiffRadianceHitDist.image.image != VK_NULL_HANDLE && width <= m_capacityWidth && height <= m_capacityHeight)
		return;

	m_capacityWidth = glm::max(width, m_capacityWidth);
	m_capacityHeight = glm::max(height, m_capacityHeight);
	const VkExtent2D size{ m_capacityWidth, m_capacityHeight };

	destroyPool(m_permanentPool);
	destroyPool(m_transientPool);
	{
		std::vector<PooledTexture> bridgeTextures{ m_inDiffRadianceHitDist, m_outDiffRadianceHitDist, m_validationTexture, m_inSpecRadianceHitDist, m_outSpecRadianceHitDist };
		destroyPool(bridgeTextures);
		m_inDiffRadianceHitDist = {};
		m_outDiffRadianceHitDist = {};
		m_validationTexture = {};
		m_inSpecRadianceHitDist = {};
		m_outSpecRadianceHitDist = {};
	}

	const nrd::InstanceDesc* instanceDesc = nrd::GetInstanceDesc(*m_instance);

	VkDevice device = g_renderer.m_context.getDevice();
	VkCommandBuffer cmd = utils::beginSingleTimeCommands(device, g_renderer.m_transientCmdPool);

	createPooledTextures(cmd, m_permanentPool, instanceDesc->permanentPool, instanceDesc->permanentPoolSize, size, "NRD-Permanent");
	createPooledTextures(cmd, m_transientPool, instanceDesc->transientPool, instanceDesc->transientPoolSize, size, "NRD-Transient");
	{
		// Full-resolution (downsampleFactor=1): reuses createPooledTextures() for this integration's
		// own bridge textures (see NrdIntegration.h) instead of duplicating its image-creation code.
		// RGBA8_UNORM for the validation overlay matches NRD's documented "RGBA8+" requirement for
		// OUT_VALIDATION. The two specular entries mirror the diffuse pair exactly (see ray.slang's
		// hdrSpecular / REBLUR_DIFFUSE_SPECULAR).
		const nrd::TextureDesc bridgeDescs[5]{
			{ nrd::Format::RGBA16_SFLOAT, 1 }, { nrd::Format::RGBA16_SFLOAT, 1 }, { nrd::Format::RGBA8_UNORM, 1 },
			{ nrd::Format::RGBA16_SFLOAT, 1 }, { nrd::Format::RGBA16_SFLOAT, 1 },
		};
		std::vector<PooledTexture> bridgeTextures;
		createPooledTextures(cmd, bridgeTextures, bridgeDescs, 5, size, "NRD-Bridge");
		m_inDiffRadianceHitDist = bridgeTextures[0];
		m_outDiffRadianceHitDist = bridgeTextures[1];
		m_validationTexture = bridgeTextures[2];
		m_inSpecRadianceHitDist = bridgeTextures[3];
		m_outSpecRadianceHitDist = bridgeTextures[4];
	}

	utils::endSingleTimeCommands(cmd, device, g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);

	// (Re)write the bridge shaders' fixed per-frame-slot descriptor sets now that the views they
	// point at have changed -- see the class comment for why this only needs to happen here, not
	// every dispatch() call.
	for (uint32 slot = 0; slot < Renderer::kMaxFramesInFlight; ++slot)
	{
		const VkDescriptorImageInfo normalRoughnessInfo{ .imageView = gbufferNormalRoughnessView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo viewZInfo{ .imageView = gbufferViewZView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo inDiffInfo{ .imageView = m_inDiffRadianceHitDist.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo outDiffInfo{ .imageView = m_outDiffRadianceHitDist.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo validationInfo{ .imageView = m_validationTexture.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo albedoInfo{ .imageView = gbufferAlbedoView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo inSpecInfo{ .imageView = m_inSpecRadianceHitDist.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		const VkDescriptorImageInfo outSpecInfo{ .imageView = m_outSpecRadianceHitDist.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

		const VkWriteDescriptorSet prepareWrites[7]{
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 0,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &normalRoughnessInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 1,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &viewZInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 2,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &inDiffInfo },
			// binding 3 unused by nrd_prepare.slang -- dummy, see nrd_resolve.slang's binding 3 comment.
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 3,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &normalRoughnessInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 4,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &albedoInfo },
			// binding 5: specular counterpart of binding 2 -- prepare's write target for IN_SPEC_RADIANCE_HITDIST.
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 5,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &inSpecInfo },
			// binding 6 unused by nrd_prepare.slang -- dummy, see nrd_resolve.slang's binding 6 comment.
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_prepareSet[slot], .dstBinding = 6,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &normalRoughnessInfo },
		};
		vkUpdateDescriptorSets(device, (uint32)std::size(prepareWrites), prepareWrites, 0, nullptr);

		// nrd_resolve.slang declares/reads bindings 0 (denoised diffuse output), 1 (viewZ, to skip
		// sky/miss pixels NRD itself never denoises), 3 (NRD's OUT_VALIDATION debug overlay), 4
		// (albedo, for re-modulation) and 6 (denoised specular output); bindings 2 and 5 are still
		// required to hold a valid descriptor of their declared type (this set layout has no
		// PARTIALLY_BOUND flag) even though resolve.slang doesn't declare/use them -- reuse
		// outDiff/outSpecRadianceHitDist's own views, which were created with both SAMPLED_IMAGE and
		// STORAGE_IMAGE usage, so they're valid for either descriptor type here.
		const VkWriteDescriptorSet resolveWrites[7]{
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 0,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &outDiffInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 1,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &viewZInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 2,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &outDiffInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 3,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &validationInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 4,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &albedoInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 5,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &outSpecInfo },
			{ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = m_resolveSet[slot], .dstBinding = 6,
			  .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &outSpecInfo },
		};
		vkUpdateDescriptorSets(device, (uint32)std::size(resolveWrites), resolveWrites, 0, nullptr);
	}
}

VkImageView NrdDenoiser::resolveResourceView(const nrd::ResourceDesc& resource, const DispatchParams& params) const
{
	switch (resource.type)
	{
	case nrd::ResourceType::IN_MV: return params.gbufferMotionVectorView;
	case nrd::ResourceType::IN_NORMAL_ROUGHNESS: return params.gbufferNormalRoughnessView;
	case nrd::ResourceType::IN_VIEWZ: return params.gbufferViewZView;
	case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST: return m_inDiffRadianceHitDist.view;
	case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST: return m_outDiffRadianceHitDist.view;
	case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST: return m_inSpecRadianceHitDist.view;
	case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST: return m_outSpecRadianceHitDist.view;
	case nrd::ResourceType::TRANSIENT_POOL: return m_transientPool[resource.indexInPool].view;
	case nrd::ResourceType::PERMANENT_POOL: return m_permanentPool[resource.indexInPool].view;
	case nrd::ResourceType::OUT_VALIDATION: return m_validationTexture.view;
	default:
		LOGE("NRD: unexpected resource type %u for REBLUR_DIFFUSE_SPECULAR", (uint32)resource.type);
		return VK_NULL_HANDLE;
	}
}

void NrdDenoiser::dispatch(VkCommandBuffer cmd, const DispatchParams& params)
{
	if (!m_instance)
		return;

	VkDevice device = g_renderer.m_context.getDevice();
	ASSERT(params.frameSlot < Renderer::kMaxFramesInFlight);

	shaderio::NrdBridgePushData pushData{};
	pushData.outputBufferAddress = params.outputBufferAddress;
	pushData.outputSpecularBufferAddress = params.outputSpecularBufferAddress;
	pushData.hitDistParamsA = m_hitDistParamsA;
	pushData.hitDistParamsB = m_hitDistParamsB;
	pushData.hitDistParamsC = m_hitDistParamsC;
	pushData.viewportWidth = params.viewportSize.width;
	pushData.viewportHeight = params.viewportSize.height;
	pushData.denoisingRange = m_denoisingRange;
	pushData.showValidation = params.showValidation ? 1 : 0;

	const uint32 groupsX = (params.viewportSize.width + 7) / 8;
	const uint32 groupsY = (params.viewportSize.height + 7) / 8;

	// This frame's ray tracing pass (G-buffer + g_outputBuffer writes) must be visible before
	// nrd_prepare reads them.
	cmdFullBarrier(cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preparePipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bridgePipelineLayout, 0, 1, &m_prepareSet[params.frameSlot], 0, nullptr);
	vkCmdPushConstants(cmd, m_bridgePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushData), &pushData);
	vkCmdDispatch(cmd, groupsX, groupsY, 1);

	cmdFullBarrier(cmd);

	// --- NRD itself ---
	nrd::CommonSettings commonSettings{};
	memcpy(commonSettings.viewToClipMatrix, params.viewToClip, sizeof(commonSettings.viewToClipMatrix));
	memcpy(commonSettings.viewToClipMatrixPrev, params.viewToClipPrev, sizeof(commonSettings.viewToClipMatrixPrev));
	memcpy(commonSettings.worldToViewMatrix, params.worldToView, sizeof(commonSettings.worldToViewMatrix));
	memcpy(commonSettings.worldToViewMatrixPrev, params.worldToViewPrev, sizeof(commonSettings.worldToViewMatrixPrev));
	commonSettings.cameraJitter[0] = params.cameraJitterX;
	commonSettings.cameraJitter[1] = params.cameraJitterY;
	commonSettings.cameraJitterPrev[0] = params.cameraJitterPrevX;
	commonSettings.cameraJitterPrev[1] = params.cameraJitterPrevY;
	// "Flexible dynamic resolution scaling support" (NRDSettings.h): resourceSize is the actual
	// allocated dimensions of every texture we bind (G-buffer, pooled, bridge -- all sized to
	// m_capacityWidth/Height, this class's grow-only capacity), while rectSize is the currently-
	// active sub-rectangle within them (the real viewport, usually smaller than capacity thanks to
	// the grow-only policy -- e.g. ImGui's docking layout settling to a smaller size than its first
	// startup estimate, per onViewportSizeChange()'s own log). Passing viewportSize for BOTH (as
	// this used to) told NRD every texture was exactly viewport-sized when they were actually
	// larger, corrupting every internal UV computation that scales by resourceSize -- confirmed
	// live as a periodic vertical-banding artifact (a wrong texture/rect size ratio aliasing NRD's
	// own bilinear resampling) before this fix.
	commonSettings.resourceSize[0] = (uint16_t)m_capacityWidth;
	commonSettings.resourceSize[1] = (uint16_t)m_capacityHeight;
	commonSettings.resourceSizePrev[0] = (uint16_t)m_capacityWidth;
	commonSettings.resourceSizePrev[1] = (uint16_t)m_capacityHeight;
	commonSettings.rectSize[0] = (uint16_t)params.viewportSize.width;
	commonSettings.rectSize[1] = (uint16_t)params.viewportSize.height;
	commonSettings.rectSizePrev[0] = (uint16_t)params.viewportSize.width;
	commonSettings.rectSizePrev[1] = (uint16_t)params.viewportSize.height;
	// Comfortably above real scene geometry but below this scene's zFar=100000 sky sentinel (see
	// ray.slang's viewZ write on a miss) -- sky pixels must read as "invalid" (viewZ > this) to NRD.
	commonSettings.denoisingRange = m_denoisingRange;
	// Slightly more lenient than NRD's own 0.01 default. REBLUR_DIFFUSE only (no specular split,
	// see readme's todo) means grazing-angle/fine-detail surfaces (e.g. tufted upholstery) carry
	// more per-pixel normal variation than a would-be specular lobe would otherwise absorb, making
	// them more prone to disocclusion false positives under camera jitter. Confirmed live (same
	// region, consecutive frames, before/after): with the default 0.01 the tufted cushion's shading
	// pattern visibly reshuffled every frame even with a static camera (validation overlay's "DIFF
	// FRAMES" panel showed permanent sub-1-frame history there); at 0.02 the same region holds a
	// stable pattern frame-to-frame.
	commonSettings.disocclusionThreshold = 0.02f;
	commonSettings.frameIndex = params.frameIndex;
	commonSettings.accumulationMode = nrd::AccumulationMode::CONTINUE;
	// Always on (trivial extra cost: one more small RGBA8 texture + overlay compute) so
	// DispatchParams::showValidation can toggle the *display* of it without needing to also
	// recreate NRD's resource list -- see nrd_resolve.slang's own use of showValidation.
	commonSettings.enableValidation = true;

	nrd::Result setCommonResult = nrd::SetCommonSettings(*m_instance, commonSettings);
	if (setCommonResult != nrd::Result::SUCCESS)
		LOGE("NRD: SetCommonSettings failed (nrd::Result = %u)", (uint32)setCommonResult);

	nrd::ReblurSettings reblurSettings{};
	reblurSettings.hitDistanceParameters.A = m_hitDistParamsA;
	reblurSettings.hitDistanceParameters.B = m_hitDistParamsB;
	reblurSettings.hitDistanceParameters.C = m_hitDistParamsC;
	reblurSettings.maxBlurRadius = maxBlurRadius;
	reblurSettings.minBlurRadius = minBlurRadius;
	reblurSettings.maxAccumulatedFrameNum = maxAccumulatedFrameNum;
	nrd::SetDenoiserSettings(*m_instance, 0, &reblurSettings);

	const nrd::Identifier identifier = 0;
	const nrd::DispatchDesc* dispatchDescs = nullptr;
	uint32 dispatchDescsNum = 0;
	nrd::Result getDispatchesResult = nrd::GetComputeDispatches(*m_instance, &identifier, 1, dispatchDescs, dispatchDescsNum);
	if (getDispatchesResult != nrd::Result::SUCCESS)
	{
		LOGE("NRD: GetComputeDispatches failed (nrd::Result = %u)", (uint32)getDispatchesResult);
		return;
	}

	ASSERT(dispatchDescsNum <= m_maxDispatchesPerFrame);
	VK_CHECK(vkResetDescriptorPool(device, m_resourcesPool[params.frameSlot], 0));

	std::vector<VkDescriptorImageInfo> imageInfos;
	std::vector<VkWriteDescriptorSet> writes;

	for (uint32 d = 0; d < dispatchDescsNum; ++d)
	{
		const nrd::DispatchDesc& dispatchDesc = dispatchDescs[d];

		VkDescriptorSet resourcesSet{};
		const VkDescriptorSetAllocateInfo resourcesAI{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = m_resourcesPool[params.frameSlot],
			.descriptorSetCount = 1, .pSetLayouts = &m_resourcesSetLayout,
		};
		VK_CHECK(vkAllocateDescriptorSets(device, &resourcesAI, &resourcesSet));

		// Each resource goes to its own sequential binding (textureBinding+i / storageBinding+i,
		// in the order NRD provides them for the two descriptor types) -- see the resources set
		// layout's own comment for why this isn't one arrayed binding per type.
		imageInfos.resize(dispatchDesc.resourcesNum);
		writes.clear();
		writes.reserve(dispatchDesc.resourcesNum);
		uint32 textureIndex = 0, storageIndex = 0;
		for (uint32 r = 0; r < dispatchDesc.resourcesNum; ++r)
		{
			const nrd::ResourceDesc& resource = dispatchDesc.resources[r];
			imageInfos[r] = { .imageView = resolveResourceView(resource, params), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

			const bool isTexture = (resource.descriptorType == nrd::DescriptorType::TEXTURE);
			const uint32 binding = isTexture ? (m_textureBinding + textureIndex++) : (m_storageBinding + storageIndex++);
			const VkDescriptorType descriptorType = isTexture ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writes.push_back({ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = resourcesSet,
				.dstBinding = binding, .descriptorCount = 1, .descriptorType = descriptorType, .pImageInfo = &imageInfos[r] });
		}
		vkUpdateDescriptorSets(device, (uint32)writes.size(), writes.data(), 0, nullptr);

		const uint32 constantOffset = d * m_constantBufferStride;
		if (dispatchDesc.constantBufferDataSize > 0)
		{
			vkCmdUpdateBuffer(cmd, m_constantBuffer[params.frameSlot].buffer, constantOffset,
				dispatchDesc.constantBufferDataSize, dispatchDesc.constantBufferData);
		}

		// Transfer-write (the constant buffer update above) and the previous dispatch's
		// compute-write (pooled/bridge textures) both need to be visible before this dispatch reads
		// them -- see cmdFullBarrier's comment for why one conservative barrier is used everywhere.
		cmdFullBarrier(cmd);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[dispatchDesc.pipelineIndex]);

		VkDescriptorSet sets[2]{};
		sets[m_resourcesSetIndex] = resourcesSet;
		sets[m_constantsSetIndex] = m_constantsSet[params.frameSlot];
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 2, sets, 1, &constantOffset);

		vkCmdDispatch(cmd, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
	}

	cmdFullBarrier(cmd);

	// --- nrd_resolve: unpack NRD's denoised OUT_DIFF_RADIANCE_HITDIST back into g_outputBuffer ---
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bridgePipelineLayout, 0, 1, &m_resolveSet[params.frameSlot], 0, nullptr);
	vkCmdPushConstants(cmd, m_bridgePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushData), &pushData);
	vkCmdDispatch(cmd, groupsX, groupsY, 1);

	// Make the resolved g_outputBuffer visible to the later graphics pass (post.slang reads it).
	cmdFullBarrier(cmd);
}
