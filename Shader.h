#pragma once
#include <vector>
#include <string>

#include "vulkanBase.h"
#include "EventManager.h"
#include "Renderer.h"  // Renderer::kMaxFramesInFlight -- see m_rtDescSet's comment


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

	// 0 if shader error
	VkPipeline m_pipeline = {};
	VkDescriptorSetLayout m_rtDescSetLayout = {};
	VkDescriptorPool m_rtPool = {};
	// One descriptor set per frame-in-flight slot (Renderer::kMaxFramesInFlight), so that
	// rebuilding the TLAS every frame (Renderer::createTopLevelAS) never has to wait for a
	// previous frame's ray tracing dispatch to finish reading the descriptor set it's bound to --
	// same double-buffering idea as App::m_frameData, reusing its existing per-slot
	// GPU-completion wait instead of adding any new fence/barrier.
	VkDescriptorSet m_rtDescSet[Renderer::kMaxFramesInFlight] = {};
	VkPipelineLayout m_rtPipelineLayout = {};
	// Shader binding table for this shader's own ray tracing pipeline -- SBT contents are
	// per-pipeline (built from this shader's own group handles), so each ray tracing Shader
	// instance owns its own, unlike the scene-level buffers (mesh/material/TLAS) that are shared
	// by every ray tracing shader. Built lazily on the first dispatchRayTracing() call, once.
	Buffer m_sbt;

public:
	// @param inFilePath e.g. "shaders/comp.slang", must not be 0
	Shader(const char* inFilePath, const char* inCS);
	// @param inFilePath e.g. "shaders/vsps.slang", must not be 0
	Shader(const char* inFilePath, const char* inVS, const char* inPS);
	// @param inFilePath e.g. "shaders/ray.slang", must not be 0
	Shader(const char* inFilePath, const char* inRayGen, const char* inRayMiss, const char* inRayClosestHit);

	//
	void compile();
	//
	void create(EventContext& context);

	// interface IAppInit ====================================
	virtual void init(EventContext& context);
	virtual void deinit(EventContext& context);
	virtual void reload(EventContext& context);

	// Ray tracing only, called explicitly by name (e.g. g_ray.updateRayDescriptorSet(...)) from
	// App.cpp, the same way g_gfx/g_post are bound explicitly in recordGraphicsCommands.
	// Traditional (non-heap) binding for the closest-hit shader's texture lookup --
	// VK_EXT_descriptor_heap can't be used here because a descriptor-heap pipeline requires
	// layout == VK_NULL_HANDLE (VUID-VkRayTracingPipelineCreateInfoKHR-flags-11311), which is
	// incompatible with the TLAS's traditional descriptor set binding 0.
	// gbufferAlbedoView/gbufferNormalRoughnessView/gbufferViewZView/gbufferMotionVectorView: bindings
	// 5-8, the G-buffer render targets ray.slang's raygen shader writes via RWTexture2D (see
	// App.cpp's g_gbufferAlbedo etc.) -- unused by any shader but g_ray, but every ray
	// tracing Shader still needs a valid view bound here (no PARTIALLY_BOUND flag on these bindings).
	// iblSpecularCubeView: binding 9, the IBL environment cubemap (see IBLEnvironment.h), sampled by
	// rayMissMain (getSkyColor(), ray.slang) for the skybox and every bounce/shadow ray that escapes
	// to open sky.
	void updateRayDescriptorSet(VkDevice device, VkAccelerationStructureKHR tlas, VkImageView outputView,
		VkBuffer sceneBuffer, const VkImageView* textureViews, uint32 textureViewCount, VkSampler sampler, uint32 frameSlot,
		VkImageView gbufferAlbedoView, VkImageView gbufferNormalRoughnessView, VkImageView gbufferViewZView, VkImageView gbufferMotionVectorView,
		VkImageView iblSpecularCubeView);
	// Lazily builds this shader's own SBT (once), binds its pipeline and this frame-slot's
	// descriptor set (see updateRayDescriptorSet), and issues vkCmdTraceRaysKHR. Called explicitly
	// by name (e.g. g_ray.dispatchRayTracing(...)) from App::dispatchRayTracing().
	// pushData is caller-built per call, so a second ray tracing shader can target a different
	// output buffer (pushData.outputAddress/prevOutputAddress) while sharing the rest of the scene.
	// @return false if not ready to dispatch (e.g. failed to compile); caller surfaces that.
	bool dispatchRayTracing(VkCommandBuffer cmd, uint32 frameSlot, VkCommandPool transientCmdPool,
		VkExtent2D size, const shaderio::RayPushData& pushData);

private:
	void createGfx(EventContext& context);
	void createRay(EventContext& context);
};