#pragma once
#include "vulkanBase.h"
#include "Renderer.h"
#include "NRD.h"
#include <vector>

// Wires NVIDIA NRD (one REBLUR_DIFFUSE_SPECULAR denoiser) into the render loop as an alternative to
// the existing freeze-and-accumulate "Reference" scheme -- see g_nrdRealtimeMode (App.cpp)
// for the Realtime/Reference toggle. Denoises two separate signals ray.slang produces: g_outputBuffer
// (everything except the direct/NEE specular lobe -- emissive, fog, GI/indirect, diffuse NEE) through
// the diffuse channel, and g_outputBufferSpecular (the direct/NEE specular lobe only -- see
// ray.slang's hdrSpecular comment for the exact split and why indirect specular bounces aren't
// included) through the specular channel, using this engine's own G-buffer (Stages 1-3 of this
// integration, see readme.md) as NRD's non-noisy inputs. NRD's own compute
// shaders are pre-embedded SPIR-V inside extern/NRD/lib/NRD.lib (see that folder's header
// comment) -- this file only creates the Vulkan objects (pipelines/descriptor sets/pooled
// textures) NRD tells us, at runtime, that it needs; it doesn't compile or vendor any NRD shader
// source itself.
//
// Descriptor/sync design (see NrdIntegration.cpp for the full reasoning):
//  - NRD's own "resources" set (SAMPLED_IMAGE[]/STORAGE_IMAGE[] arrays) gets a brand new
//    VkDescriptorSet per GetComputeDispatches() entry, allocated from a per-frame-slot pool reset
//    once at the start of dispatch() -- descriptor sets can't be safely rewritten and reused
//    within the same not-yet-submitted command buffer (the GPU would see only the LAST write for
//    every use of that set, not the write in effect when each dispatch was recorded).
//  - NRD's "constants+samplers" set uses ONE static VkDescriptorSet per frame slot with a
//    UNIFORM_BUFFER_DYNAMIC binding -- each dispatch gets its own region of a per-frame-slot ring
//    buffer, selected via a dynamic offset at bind time (no descriptor rewriting needed).
//  - This integration's own two bridge shaders (nrd_prepare/nrd_resolve) share one descriptor set
//    layout/pipeline; their two per-frame-slot sets are written once in ensureCapacity() (only
//    the G-buffer/bridge-texture views change, on resize) and simply rebound every frame.
class NrdDenoiser
{
public:
	void init();
	void deinit();

	// Grow-only, same philosophy/safety requirement as App::EnsureRtxdiCapacity (only
	// ever call from init()/onViewportSizeChange(), never mid-frame -- see that function's comment
	// for why: freeing a resource a still-in-flight dispatch might read is undefined).
	// gbufferNormalRoughnessView/gbufferViewZView/gbufferAlbedoView: only needed here (not just in
	// dispatch()'s DispatchParams) because the bridge shaders' own fixed descriptor sets
	// (m_prepareSet/m_resolveSet) are written once here, not every dispatch() call -- see the class
	// comment.
	void ensureCapacity(uint32 width, uint32 height, VkImageView gbufferNormalRoughnessView, VkImageView gbufferViewZView, VkImageView gbufferAlbedoView);

	// REBLUR_DIFFUSE tuning, live-tunable (Settings window) and persisted (Settings.ini) -- applied
	// every dispatch() call (see its own nrd::ReblurSettings construction), not just at init(), so a
	// slider takes effect immediately. Defaults match nrd::ReblurSettings' own struct defaults
	// (NRDSettings.h) -- this project's starting point before any tuning.
	//
	// maxBlurRadius/minBlurRadius (pixels): the denoiser's spatial blur kernel size -- maxBlurRadius
	// is the radius used while history is still short (right after a disocclusion or a moving
	// object), decaying toward minBlurRadius as more frames accumulate for that pixel. Too much
	// blur with little noise left to justify it usually means these can come down.
	// maxAccumulatedFrameNum: how many frames of temporal history REBLUR keeps blending into a
	// stable pixel -- higher smooths noise more but reacts more slowly to real scene changes
	// (ghosting on disocclusion/fast motion); lower is more responsive but noisier per frame.
	float maxBlurRadius = 30.0f;
	float minBlurRadius = 1.0f;
	uint32 maxAccumulatedFrameNum = 30;

	struct DispatchParams
	{
		VkExtent2D viewportSize;
		uint32 frameSlot;   // m_swapchain.getFrameResourceIndex() -- which of the 2 frame-in-flight copies to use
		uint32 frameIndex;  // matches shaderio::SceneInfo::frameIndex for this same frame

		// This engine's own G-buffer (already exactly NRD's expected format -- see
		// NrdIntegration.cpp's header comment -- bound directly, no bridging shader for these 3).
		VkImageView gbufferMotionVectorView;
		VkImageView gbufferNormalRoughnessView;
		VkImageView gbufferViewZView;
		// Albedo demodulation (see nrd_prepare.slang/nrd_resolve.slang's header comments) -- bound
		// directly like the other 3 G-buffer views above, no bridging needed.
		VkImageView gbufferAlbedoView;

		// Raw GLM column-major float[16] (glm::value_ptr), current and previous frame -- NRD's own
		// convention directly (CommonSettings: "usage - vector is a column, layout - column-major"),
		// NOT the shader-side row_major-with-C++-transpose convention shader_io.h's SceneInfo uses.
		// See the call site (App.cpp) for exactly which existing matrices are already in
		// this raw form vs. need an extra transpose to get there.
		const float* worldToView;
		const float* worldToViewPrev;
		const float* viewToClip;
		const float* viewToClipPrev;

		// [-0.5; 0.5], matching ray.slang's own GetTemporalJitter() (mathlib.slang), re-centered.
		float cameraJitterX, cameraJitterY;
		float cameraJitterPrevX, cameraJitterPrevY;

		// g_outputBuffer[currentIndex].m_data.address -- the same buffer ray.slang wrote this
		// frame's (RGB radiance, signed hit distance) into; nrd_prepare/nrd_resolve.slang read and
		// (resolve only) rewrite it in place via this same buffer-device-address idiom.
		VkDeviceAddress outputBufferAddress;
		// g_outputBufferSpecular.m_data.address -- same idiom as outputBufferAddress above, for
		// ray.slang's direct/NEE-specular-only channel (see its own hdrSpecular comment). Not
		// ping-ponged (no "prev" counterpart) -- see g_outputBufferSpecular's own comment
		// (App.cpp) for why.
		VkDeviceAddress outputSpecularBufferAddress;

		// Diagnostic-only (see g_nrdShowValidation, App.cpp): when true, enables NRD's
		// own REBLUR_VALIDATION debug overlay and has nrd_resolve.slang display it instead of the
		// denoised radiance -- only meaningful with VisualizeMode left at Default.
		bool showValidation = false;
	};

	// Runs nrd_prepare -> NRD's own GetComputeDispatches() sequence -> nrd_resolve for one frame.
	void dispatch(VkCommandBuffer cmd, const DispatchParams& params);

private:
	struct PooledTexture
	{
		utils::Image image{};
		VkImageView view{};
	};

	void createPooledTextures(VkCommandBuffer cmd, std::vector<PooledTexture>& pool, const nrd::TextureDesc* descs, uint32 count, VkExtent2D viewportSize, const char* debugPrefix);
	void destroyPool(std::vector<PooledTexture>& pool);
	VkImageView resolveResourceView(const nrd::ResourceDesc& resource, const DispatchParams& params) const;
	void updateBridgeDescriptorSets(const DispatchParams& params);

	nrd::Instance* m_instance = nullptr;

	VkSampler m_samplers[2] = {}; // nrd::Sampler::MAX_NUM == 2: NEAREST_CLAMP then LINEAR_CLAMP, immutable

	// NRD's own 2-descriptor-set layout, per its InstanceDesc (space indices/binding offsets
	// queried at init() -- see NrdIntegration.cpp, not hardcoded, in case a future NRD version
	// changes them). One shared pipeline layout for every InstanceDesc::pipelines[] entry.
	VkDescriptorSetLayout m_resourcesSetLayout = {};  // "resourcesSpaceIndex" set: textures + storage textures
	VkDescriptorSetLayout m_constantsSetLayout = {};  // "constantBufferAndSamplersSpaceIndex" set: 1 dynamic UBO + samplers
	VkPipelineLayout m_pipelineLayout = {};
	uint32 m_resourcesSetIndex = 0;
	uint32 m_constantsSetIndex = 1;
	uint32 m_textureBinding = 0, m_storageBinding = 0, m_constantBufferBinding = 0, m_samplerBinding = 0;
	uint32 m_perSetTexturesMaxNum = 0;
	uint32 m_perSetStorageTexturesMaxNum = 0;

	std::vector<VkPipeline> m_pipelines; // one per InstanceDesc::pipelines[], indexed by DispatchDesc::pipelineIndex

	// "Resources" set: reset once per frame slot at the start of dispatch(), then one fresh
	// VkDescriptorSet allocated+written per GetComputeDispatches() entry (see class comment).
	VkDescriptorPool m_resourcesPool[Renderer::kMaxFramesInFlight] = {};

	// "Constants+samplers" set: one static set per frame slot, bound with a per-dispatch dynamic
	// offset into that slot's ring buffer (see class comment) -- both created once in init(),
	// independent of viewport size.
	VkDescriptorPool m_constantsPool = {};
	VkDescriptorSet m_constantsSet[Renderer::kMaxFramesInFlight] = {};
	utils::Buffer m_constantBuffer[Renderer::kMaxFramesInFlight] = {};
	uint32 m_constantBufferStride = 0;   // constantBufferMaxDataSize rounded up to minUniformBufferOffsetAlignment
	uint32 m_maxDispatchesPerFrame = 0;  // = instanceDesc.descriptorPoolDesc.setsMaxNum; ring buffer slot count

	// This integration's own small bridge shaders (not from NRD) -- one shared descriptor set
	// layout/pipeline layout/pool for both (see class comment); one set per frame slot each,
	// (re)written in ensureCapacity() only (the underlying views only change on resize).
	VkDescriptorSetLayout m_bridgeSetLayout = {};
	VkPipelineLayout m_bridgePipelineLayout = {};
	VkPipeline m_preparePipeline = {};
	VkPipeline m_resolvePipeline = {};
	VkDescriptorPool m_bridgePool = {};
	VkDescriptorSet m_prepareSet[Renderer::kMaxFramesInFlight] = {};
	VkDescriptorSet m_resolveSet[Renderer::kMaxFramesInFlight] = {};

	// NRD's own pooled textures (its temporal history etc.) and this integration's two bridge
	// textures -- all single-instance, NOT double-buffered: NRD manages its own frame-to-frame
	// temporal state internally across these same textures (that's the point of PERMANENT_POOL),
	// and this project submits every frame's work to one queue in frame order, so consecutive
	// frames' compute dispatches touching the same texture are naturally serialized by Vulkan's
	// in-order queue execution -- no extra sync needed beyond the barriers already inserted between
	// consecutive dispatches within a single frame.
	std::vector<PooledTexture> m_permanentPool;
	std::vector<PooledTexture> m_transientPool;
	PooledTexture m_inDiffRadianceHitDist;
	PooledTexture m_outDiffRadianceHitDist;
	// Specular counterparts (see ray.slang's hdrSpecular / REBLUR_DIFFUSE_SPECULAR, NrdIntegration.cpp's
	// init()) -- same role/lifetime as the diffuse pair above.
	PooledTexture m_inSpecRadianceHitDist;
	PooledTexture m_outSpecRadianceHitDist;
	// OUT_VALIDATION -- NRD's own debug overlay, requested every frame regardless of whether it's
	// actually being displayed (see DispatchParams::showValidation); trivial extra cost, and always
	// having it bound keeps the bridge descriptor sets' layout unconditional. Single-instance like
	// the other pooled/bridge textures above -- see their shared comment for why.
	PooledTexture m_validationTexture;

	uint32 m_capacityWidth = 0;
	uint32 m_capacityHeight = 0;

	// REBLUR_FrontEnd_GetNormHitDist's hitDistParams (nrd_pack.slang) -- kept in lockstep with
	// nrd::ReblurSettings::hitDistanceParameters, set once in init().
	float m_hitDistParamsA = 3.0f, m_hitDistParamsB = 0.1f, m_hitDistParamsC = 20.0f;

	// Comfortably above real scene geometry but below this scene's zFar=100000 sky sentinel (see
	// ray.slang's viewZ write on a miss) -- kept in lockstep between nrd::CommonSettings and
	// nrd_resolve.slang's own copy (NrdBridgePushData), which needs it to skip pixels NRD itself
	// never denoises (see that shader's comment).
	float m_denoisingRange = 10000.0f;
};

extern NrdDenoiser g_nrdDenoiser;
