// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <vector>
#include <string>

#include "vulkanBase.h"
#include "EventManager.h"
#include "Renderer.h"  // Renderer::kMaxFramesInFlight -- see m_descSet's comment


class Shader : public IAppInit
{
public: // todo
	// e.g. "shaders/rast.slang"
	std::string filePath;

	// compute: [0]:CS
	// vsps:	[0]:VS, [1]:PS
	// ray:		[0]:rayGen, [1]:rayMiss, [2]:closestHit
	std::string functionName[3];
	std::vector<uint32> code[3];
	// not used for ray
	VkShaderEXT m_shader[3] = {};

	// vsps only. 0: bindless -- shader objects are created with VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT,
	// textures come from the descriptor heap, no descriptor sets at all. >0: non-bindless -- classic
	// descriptor set 0 with one combined image sampler per texture at bindings 0..m_textureCount-1
	// (e.g. [[vk::binding(0, 0)]] Sampler2D in Slang), see cmdBindTextures().
	uint32 m_textureCount = 0;

	// ray tracing pipeline -- ray shaders aren't created yet (see create()), so always 0 for now
	VkPipeline m_pipeline = {};
	// non-bindless only (m_textureCount > 0)
	VkDescriptorSetLayout m_descSetLayout = {};
	VkDescriptorPool m_descPool = {};
	// One descriptor set per frame-in-flight slot (Renderer::kMaxFramesInFlight), so that
	// rewriting it every frame (cmdBindTextures) never has to wait for a previous frame still
	// reading the descriptor set it's bound to -- same double-buffering idea as App::m_frameData,
	// reusing its existing per-slot GPU-completion wait instead of adding any new fence/barrier.
	VkDescriptorSet m_descSet[Renderer::kMaxFramesInFlight] = {};
	// m_descSetLayout + the GraphicsPushData push constant range (non-bindless only)
	VkPipelineLayout m_pipelineLayout = {};
	// Shader binding table for this shader's own ray tracing pipeline (m_pipeline) -- SBT contents
	// are per-pipeline (built from its group handles), so each ray tracing Shader owns its own.
	// Not built yet (see m_pipeline).
	Buffer m_sbt;

public:
	// @param inFilePath e.g. "shaders/comp.slang", must not be 0
	Shader(const char* inFilePath, const char* inCS);
	// @param inFilePath e.g. "shaders/vsps.slang", must not be 0
	// @param inTextureCount 0: bindless, >0: non-bindless, see m_textureCount
	Shader(const char* inFilePath, const char* inVS, const char* inPS, uint32 inTextureCount = 0);
	// @param inFilePath e.g. "shaders/ray.slang", must not be 0
	Shader(const char* inFilePath, const char* inRayGen, const char* inRayMiss, const char* inRayClosestHit);

	//
	void compile();
	//
	void create(EventContext& context);

	// Sets the GraphicsPushData for the next draw: vkCmdPushDataEXT when bindless, vkCmdPushConstants2
	// when non-bindless (a descriptor heap shader only sees push data, a classic one only push constants).
	void cmdPushData(VkCommandBuffer cmd, const shaderio::GraphicsPushData& data) const;
	// Non-bindless only: writes this frame slot's descriptor set and binds it as set 0.
	// @param frameSlot swapchain frame-in-flight index, its set is safe to rewrite because
	//   App::prepareFrameResources() already waited for that slot's previous frame
	// @param textures one per texture slot (m_textureCount), in binding order
	void cmdBindTextures(VkDevice device, VkCommandBuffer cmd, uint32 frameSlot, std::span<const VkDescriptorImageInfo> textures);

	// interface IAppInit ====================================
	virtual void init(EventContext& context);
	virtual void deinit(EventContext& context);
	virtual void reload(EventContext& context);

private:
	void createGfx(EventContext& context);
	// non-bindless only: m_descSetLayout, m_descPool, m_descSet, m_pipelineLayout
	void createDescriptorSets(EventContext& context);
};