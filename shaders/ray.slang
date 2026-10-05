
// 0: off, 1:laser, 2:line light (legacy)
#define LIGHT_TYPE 0
// 0/1 -- one TraceRay per light sampled below was the main cost tanking framerate; turning this
// off skips the trace entirely (every light unoccluded) if kMaxShadowRaysPerPoint isn't enough.
#define ENABLE_LIGHT_SHADOWS 1
// g_lights (App.cpp) can hold dozens/hundreds of lights (e.g. a BSP level's worth) --
// tracing one shadow ray per light per shading point doesn't scale, so at most this many lights,
// chosen at random each time, are sampled per point; the result is scaled up by
// lightCount/sampleCount (see the light loops in rayGenMain) so the expected brightness matches
// summing every light, just noisier (fewer, randomly rotating lights each frame -- smooths out
// under temporal accumulation).
static const uint kMaxShadowRaysPerPoint = 10;


#include "shader_io.h"

#include "s2h/s2h_glsl.hlsl"
#include "s2h/s2h.hlsl"
#include "s2h/s2h_3d.hlsl"

#include "mathlib.slang"

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Push data (vkCmdPushDataEXT): carries only the scene buffer address and per-draw color.
// With descriptor heap (VK_EXT_descriptor_heap), the pipeline layout is VK_NULL_HANDLE,
// so traditional push constants and push descriptors (UBOs) cannot be used.
[[vk::push_constant]]
ConstantBuffer<RayPushData> pushData;

// Descriptor heap: textures and samplers live in GPU heap buffers (VK_EXT_descriptor_heap).
// We access them by heap index using Texture2D.Handle(uint2(index, 0)) and
// SamplerState.Handle(uint2(index, 0)). This replaces the old [[vk::binding(...)]] Sampler2D array.

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure g_tlas;

// Traditional (non-heap) texture binding for the closest-hit shader's texture lookup.
// VK_EXT_descriptor_heap can't be used on this pipeline: a descriptor-heap pipeline requires
// layout == VK_NULL_HANDLE (VUID-VkRayTracingPipelineCreateInfoKHR-flags-11311), which conflicts
// with g_tlas's traditional descriptor set binding above. Indexed either with a per-material
// glTF texture id (MaterialData::textureId, same index the graphics pass's descriptor heap uses)
// or, if the material has none, with scene.texId (the debug "Image 1"/"Image 2" toggle) -- see
// resolveTextureId() in mathlib.slang. Capped at kMaxRayTracingTextures entries; ids beyond the
// cap are clamped there.
[[vk::binding(3, 0)]]
SamplerState g_sampler;
[[vk::binding(4, 0)]]
Texture2D g_textures[kMaxRayTracingTextures];

// G-buffer render targets, written unconditionally by rayGenMain every pixel/frame (see the block
// near the end of rayGenMain) -- infrastructure for a future GPU-side denoiser (NRD); nothing reads
// these yet. Bindings 5-8, see Shader::updateRayDescriptorSet (Shader.cpp) and
// App::EnsureGBufferCapacity (App.cpp) for how they're created/bound.
[[vk::binding(5, 0)]]
RWTexture2D<float4> g_gbufferAlbedo;
[[vk::binding(6, 0)]]
RWTexture2D<float4> g_gbufferNormalRoughness;
[[vk::binding(7, 0)]]
RWTexture2D<float> g_gbufferViewZ;
[[vk::binding(8, 0)]]
RWTexture2D<float2> g_gbufferMotionVector;

// Image-based lighting environment (see IBLEnvironment.h) -- only the specular cubemap, only ever
// sampled at mip 0 (see getSkyColor() below): unlike the rasterizer, ray tracing gets its
// roughness response from the existing GGX-importance-sampled bounce direction (rayGenMain's
// path-tracing loop) instead of a pre-filtered mip, so no mip selection or diffuse irradiance map
// is needed here.
[[vk::binding(9, 0)]]
TextureCube g_iblSpecular;

// @return worldToView aka viewFromWorld
float4x4 GetViewMatrix()
{
	SceneInfo *scene = (SceneInfo *)pushData.sceneInfoAddress;

	return scene.viewFromWorld; //  pos is at bottom of matrix

//	return lookAt(float3(0, 0, 15), float3(0, 0, 0), float3(0, 1, 0));	// this moves the VS position, pos is at bottom of matrix
	// test
//	return float4x4( // Note: S2H prints out in this form
//		1.0, 2.0, 3.0, 0.1,
//		0.0, 1.0, 0.0, 0.2,
//		0.0, 0.0, 1.0, 0.3,
//		0.4, 0.5, 0.6, 1.0);
}

// @return viewToClip aka ClipFromView
float4x4 GetPerspMatrix()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

//	return scene.clipFromView;// bad

//	return ComputePerspectiveMatrix(45.0f * PI / 180.0f, scene.resolution.x / (float)scene.resolution.y, 0.1, 1000.0f);
//	return transpose(ComputePerspectiveMatrix(45.0f * PI / 180.0f, 16.0f / 9.0f, 0.1, 1000.0f));
	return transpose(scene.clipFromView);

	// test
//	return float4x4( // Note: S2H prints out in this form
//		2.41f, 0.0,   3.0, 0.0,
//		0.0,   2.41f, 0.0, 0.0,
//		0.0,   0.0,   0.0, 1.0,
//		0.0,   0.0,   0.1, 0.0);
}

float3 GetCameraPos()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;
	return scene.wsCameraPos;
}


#define S2S_CAMERA_POS() GetCameraPos()
#define S2S_NEAR() 0.1f

//------------------------------------------------------------------------------

struct Payload
{
    //
	float3 materialColor;
    //
	float3 emissiveColor;
    // 0..1
	float metallic;
    // 0..1
	float roughness;
    // 0 = disabled, >0 = darker albedo suppresses specular too -- see GetF0's own
    // specularOcclusionFromAlbedo parameter (mathlib.slang) and MaterialData::
    // specularOcclusionFromAlbedo's comment (shader_io.h).
	float specularOcclusionFromAlbedo;
    // in world space, not normalized during intersection but afterwards
	float3 interpolatedNormal;
    // -1 if not set
	int primitiveIndex;
    // -1 if not set, InstanceIndex() (system-generated TLAS-array position -- not comparable to
    // interaction.slang's pick result, which reports the entity index; see instanceId below)
	int instanceIndex;
	// -1 if not set. InstanceID() (gl_InstanceCustomIndexEXT) -- the entity's own raw index (see
	// Renderer::createTopLevelAS's comment on asInstance.instanceCustomIndex), the same value
	// interaction.slang's pick ray reports -- compare a selected-object index against *this*
	// field, not instanceIndex above.
	int instanceId;
	// used like z buffer, the smallest valus is used
	float minT;
};



// @param worldRayDirection normalized, e.g. WorldRayDirection()
float3 getSkyColor(float3 worldRayDirection)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

	// IBL environment cubemap (see IBLEnvironment.h/g_iblSpecular's own comment): every ray that
	// escapes to open sky samples it, whether that's the camera's own primary ray (the visible
	// skybox) or a bounce/shadow ray deep in rayGenMain's path-tracing loop -- the latter is what
	// makes IBL lighting on surfaces naturally shadowed (a bounce blocked by geometry never reaches
	// here at all) and gives roughness its correct response for free (the bounce direction was
	// already GGX-importance-sampled by roughness before tracing) with no separate diffuse map or
	// mip selection needed, unlike the rasterizer's equivalent (shader.rast.slang).
	if (scene.enableIBL != 0)
		return g_iblSpecular.SampleLevel(g_sampler, RotateCubemapLookupDir(worldRayDirection), 0.0f).rgb;

	// IBL off: the flat, UI-controlled sky (Environment tab's skyColor/skyIntensity, see
	// updateSceneBuffer(), App.cpp) -- same value every miss used before this function
	// had a caller, direction-independent.
	return s2h_accurateSRGBToLinear(scene.clearColor);
}

// Homogeneous participating media (Beer-Lambert extinction, no in-scattering/emission). Every ray
// segment that contributes to the final radiance -- primary, shadow, bounce -- routes its
// attenuation through here, so this is the one place to touch when upgrading to non-homogeneous
// fog (e.g. height-based or noise-driven density integrated along the segment instead of a single
// sigma_t * distance product).
// @param distance the ray segment's traveled length
// @return per-channel transmittance (1 = no attenuation, 0 = fully extinguished)
float3 EvalFogTransmittance(float distance)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;
	if (scene.fogDensity <= 0.0f)
		return 1.0f;
	float3 sigma_t = scene.fogDensity * scene.fogColor;
	return exp(-sigma_t * distance);
}

// Henyey & Greenstein, "Diffuse Radiation in the Galaxy" (1941), via the phase-function writeup in
// "Physically Based Rendering: From Theory to Implementation" -- copied (scalar form only; the
// Monte-Carlo direction-sampling variant isn't needed here, see EvalInscattering) from
// https://alextardif.com/downloads/ParticipatingMediaHLSL.html.
// @param cosineTheta angle between the direction toward the viewer and the direction toward the light
// @param g asymmetry, -1..1: >0 forward scattering, <0 backscattering, 0 isotropic
float CalculatePhaseHenyeyGreenstein(float cosineTheta, float g)
{
	float g2 = g * g;
	float denom = 1.0f + g2 + 2.0f * g * cosineTheta;
	return (1.0f - g2) / (4.0f * PI * denom * sqrt(max(denom, 1e-6f)));
}

// Single-scattering in-scattering toward a point light, sampled once at samplePos (no ray
// marching -- same single-sample scope as EvalFogTransmittance above; swap in a ray-marched
// integral here for non-homogeneous fog later). Reuses fogDensity * fogColor as the scattering
// coefficient too (single-scattering albedo of 1: no separate absorption term), and the phase
// function biases how much of it reaches the viewer for this ray direction.
// @param viewDir the ray direction arriving at samplePos (camera->point, or a bounce's direction)
// @param lightVisibility caller's already-computed shadow-ray result (occlusion combined with its
// own fog transmittance to the light, see perSample) -- avoids a second shadow ray here.
float3 EvalInscattering(float3 viewDir, float3 samplePos, float3 lightPos, float3 lightIntensity, float3 lightVisibility)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;
	if (scene.fogDensity <= 0.0f)
		return 0.0f;

	float3 L = normalize(lightPos - samplePos);
	float cosTheta = dot(-viewDir, L);
	float phase = CalculatePhaseHenyeyGreenstein(cosTheta, scene.fogAsymmetry);
//	float phase = lerp(CalculatePhaseHenyeyGreenstein(cosTheta, -0.9f), CalculatePhaseHenyeyGreenstein(cosTheta, 0.9f), 0.5f);

	float3 sigma_s = scene.fogDensity * scene.fogColor;
	return sigma_s * phase * lightIntensity * lightVisibility;
}

// Picks a light index uniformly at random -- used only for the secondary-bounce and sky-miss
// in-scattering light terms below, which don't go through RTXDI (see the primary-hit direct
// lighting branch in rayGenMain for that). Light picking for the primary hit used to go through a
// CPU-built power CDF (PickImportanceSampledLight); that's now RTXDI's job (spatiotemporal
// resampling gets a much better result from the same ray budget than a static per-frame CDF ever
// could), so the CDF and this function's old weighted-picking logic were removed together.
uint PickUniformRandomLight(uint lightCount, inout uint rndState, out float invPdf)
{
	invPdf = (float)lightCount;
	return min((uint)(nextRand(rndState) * lightCount), lightCount - 1);
}

// Perturbs a point light's position within a disc of radius sourceRadius facing surfacePos, to
// approximate a small area light source for soft shadows -- one random sample per call; the noise
// this adds smooths out under temporal accumulation as it changes frame to frame. sourceRadius is
// the light's own physical size (Light::sourceRadius), NOT its falloff distance (Light::radius,
// see EvalLightFalloff) -- shrinking that would just make the light dimmer sooner, not soften its
// shadow.
float3 JitterLightPosition(float3 lightPos, float3 surfacePos, float sourceRadius, inout uint rndState)
{
	if (sourceRadius <= 0.0f)
		return lightPos;

	float3 toLight = normalize(lightPos - surfacePos);
	float3 u, v;
	getOtherBaseVec(u, v, toLight);

	float2 diskSample = nextRand2Disc(rndState);
	return lightPos + sourceRadius * (diskSample.x * u + diskSample.y * v);
}

// @param pointLightPos in world space
// @return shadow visibility (0 = occluded) combined with the shadow ray's fog transmittance
float3 perSample(float3 surfacePos, float3 surfaceNormal, float3 pointLightPos, inout uint rndState)
{
#if !ENABLE_LIGHT_SHADOWS
	// Shadows disabled for now (see ENABLE_LIGHT_SHADOWS) -- every light is treated as unoccluded,
	// same as shader.rast.slang (which has no acceleration structure to trace against at all).
	return EvalFogTransmittance(length(pointLightPos - surfacePos));
#else
	// Distance to the light -- the shadow ray must stop there, not fly on for TMax=10000 past it.
	// A fixed long TMax used to mean the ray kept going well beyond a nearby light and (almost
	// certainly, inside an enclosed BSP corridor) hit the next wall -- registering as "occluded"
	// even when nothing actually stood between surfacePos and the light itself.
	float distToLight = length(pointLightPos - surfacePos);

	RayDesc rayDesc;
	rayDesc.Origin = surfacePos;
    // Set TMin to a non-zero small value to avoid aliasing issues due to floating - point errors.
    // TMin should be kept small to prevent missing geometry at close contact areas.
	rayDesc.TMin = 0.001f;
	rayDesc.TMax = distToLight;

	Payload payload2 = (Payload) 0;
	payload2.primitiveIndex = -1;
	payload2.instanceId = -1;
	payload2.instanceIndex = -1;

	RAY_FLAG flags = RAY_FLAG_NONE;
	uint instanceMask = ~0;
	uint RayContributionToHitGroupIndex = 0;
	uint MultiplierForGeometryContributionToHitGroupIndex = 1;	// ???????
	uint MissShaderIndex = 0;

	float weight = 1.0f;
	{
        // reference
//		rayDesc.Direction = getCosHemisphereSample(rndState, surfaceNormal);
//		rayDesc.Direction = normalize(float3(1, 2, 2));	// todo
		rayDesc.Direction = (pointLightPos - surfacePos) / distToLight; // already have the length above

		// TraceRay2 (not plain TraceRay) so shadow rays also render the light/laser debug lines --
		// wanted so they show up in reflections and other secondary views, not just the primary ray.
		// Note: a shadow ray that happens to pass right through its own target light's debug-line
		// marker can misread that as an occluding hit (TraceRay2 has no notion of this ray only
		// being valid up to TMax=distToLight) -- a real but narrow edge case, not the cause of the
		// widespread black-BSP-lighting bug (that was init()'s BSP mirror scale feeding a negative
		// radius into EvalLightFalloff, fixed in updateLights()'s bspScale).
		TraceRay2(g_tlas, flags, instanceMask, RayContributionToHitGroupIndex, MultiplierForGeometryContributionToHitGroupIndex, MissShaderIndex, rayDesc, payload2);
	}

	float3 addLight = 0;

	addLight = payload2.emissiveColor;
	addLight *= weight;

	if (payload2.primitiveIndex != -1)
		return 0.0f; // occluded -- no light reaches surfacePos, so fog along the way doesn't matter

	// Unoccluded: the medium it travels through is surfacePos..pointLightPos, i.e. distToLight.
	return EvalFogTransmittance(distToLight);
#endif
}


uint getLaserCornerCount()
{
#if LIGHT_TYPE == 0
	return 0;
#elif LIGHT_TYPE == 2
	return 2;
#else
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;
	uint2 resolution = (uint2)scene.resolution;
	// 16MB extra data to maintain state over multiple frames
	float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

	uint count = asuint(extraDataOut[0].x);
	return count;
#endif
}
float3 computeLaserCorner(uint cornerIndex)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

#if LIGHT_TYPE == 0
	return 0;
#elif LIGHT_TYPE == 2
	float r = scene.lightHalfLength;

	if(cornerIndex == 0)
		return scene.wsLightPos + float3(0, 0, -r);
	else
		return scene.wsLightPos + float3(0, 0, r);
#else
	uint2 resolution = (uint2)scene.resolution;
	// 16MB extra data to maintain state over multiple frames
	float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

	return extraDataOut[1 + cornerIndex].xyz;
#endif
}

// @return world space light pos
float3 computeSingleLight(inout uint rndState, out float3 outLightColor)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

	// 0..2.999
	float laserPos = (getLaserCornerCount() - 1.001f) * nextRand(rndState);

#if LIGHT_TYPE == 0
	outLightColor = 1.0f;
#elif LIGHT_TYPE == 2
	outLightColor = scene.lightColor;
#else
	outLightColor = float4(15, 0.2f, 0.1f, 0.4f) * scene.laserIntensity;		// deep red
#endif

	uint index = (uint)laserPos;
	// 0..2
	float3 A = computeLaserCorner(index);
	// 1..3
	float3 B = computeLaserCorner(index + 1);

//	return lerp(A, B, frac(laserPos));
	return lerp(A, B, laserPos - index);
}

void TraceRay2(
	RaytracingAccelerationStructure topLevel,
	uint rayFlags,
	uint cullMask,
	uint sbtRecordOffset,
	uint sbtRecordStride,
	uint missIndex,
	RayDesc ray,
	inout Payload payload)
{
	TraceRay(topLevel, rayFlags, cullMask, sbtRecordOffset, sbtRecordStride, missIndex, ray, payload);

	{
		SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

		Context3D context;

		s2h_init(context, ray.Origin, ray.Direction);
		context.depth = payload.minT;

		{
			uint2 resolution = (uint2)scene.resolution;
			// 16MB extra data to maintain state over multiple frames
			float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

			uint count = asuint(extraDataOut[0].x);

			for (uint i = 1; i < count; ++i)
			{
				float3 p0 = extraDataOut[1 + i - 1].xyz;
				float3 p1 = extraDataOut[1 + i].xyz;

				s2h_drawLineWS(context, p0, p1, float4(15, 0.2f, 0.1f, 0.4f) / 15 * scene.laserIntensity, 0.01f);
			}
		}

#if LIGHT_TYPE == 0
#elif LIGHT_TYPE == 2
//		if(scene.drawLights != 0)
//			s2h_drawLineWS(context, scene.wsLightPos - float3(0, 0, scene.lightHalfLength), scene.wsLightPos + float3(0, 0, scene.lightHalfLength), float4(scene.lightColor, 1), 0.05f);
#endif
		// laser physical
//		s2h_drawSphereWS(context, float3(0, 1.5f, -2), float4(1, 0, 0, 1), 0.02f);
//		s2h_drawLineWS(context, float3(0, 0.5f, -2), float3(0, 0.5f, -2) - float3(0, 0, 0.5f), float4(0.01f, 0.01f, 0.01f, 1.0f), 0.1f);

		if(context.dstColor.a > 0)
		{
			payload.materialColor = 0;
			payload.emissiveColor = context.dstColor.rgb;
			payload.metallic = 0;
			payload.roughness = 0;
			payload.interpolatedNormal = 0;
			payload.primitiveIndex = 0;
			payload.instanceId = -1;
			payload.minT = context.depth;
		}
//		newColor.rgb = lerp(newColor.rgb, context.dstColor.rgb, context.dstColor.a);
	}
}


// RTXDI (ReSTIR DI) -- replaces the old CDF-based light picking for the primary-hit direct
// lighting term (see the pathTracingBounces == 0 branch below and shaders/rtxdi/RtxdiBridge.slang).
// Included down here, not up top with mathlib.slang, because the bridge needs Payload/TraceRay2/
// pushData/g_tlas (all above) already declared, and RTXDI's own DI headers need the bridge's RAB_*
// functions declared before them (same order the reference RTXDI sample uses).
#define RTXDI_ENABLE_PRESAMPLING 0
//#define RTXDI_REGIR_DISABLED 0
#define RTXDI_REGIR_MODE RTXDI_REGIR_DISABLED

#include "rtxdi_io.h"
#include "Rtxdi/Utils/Math.hlsli"
#include "Rtxdi/Utils/RandomSamplerState.hlsli"
#include "Rtxdi/Utils/TypePacking.hlsli"
#include "rtxdi/RtxdiBridge.slang"

// Reservoir/neighbor-offset buffer access, following the same buffer-device-address raw-pointer
// idiom as g_lights/scene.lightsAddress elsewhere in this file. Assigned in rayGenMain below,
// before any RTXDI resampling call.
#define RTXDI_LIGHT_RESERVOIR_BUFFER g_rtxdiReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER g_rtxdiNeighborOffsets
static RTXDI_PackedDIReservoir* g_rtxdiReservoirs;
static float2* g_rtxdiNeighborOffsets;

#include "Rtxdi/DI/InitialSampling.hlsli"
#include "Rtxdi/DI/ReservoirStorage.hlsli"
#include "Rtxdi/DI/SpatioTemporalResampling.hlsli"

// ReSTIR GI (temporal-only indirect-bounce resampling) -- see gi_reservoir.slang's own header
// comment for exactly what this is (and is not: no spatial reuse, no shift-mapping). Not part of
// the vendored RTXDI SDK (only DI + ReGIR are vendored), so this is a small, self-contained
// addition rather than an SDK include -- no include-order dependency on the RTXDI headers above.
#include "gi_io.h"
static GIReservoir* g_giReservoirs;
#include "gi_reservoir.slang"

// Re-derives a hit's material index from its entity + primitive index -- the same lookup
// rayClosestHitMain's own materialID uses -- instead of carrying it in Payload: Payload is copied
// through every TraceRay call in the bounce loop and should stay small (register pressure), and
// this is only ever needed once per pixel, for the Ctrl+material-pick highlight check below.
uint GetMaterialID(int entityId, int primitiveIndex)
{
	uint meshID = ((uint*)pushData.entityMeshIndexAddress)[entityId];
	uint64_t* meshBuffer = (uint64_t*)pushData.meshBufferAddress + 3 * meshID;
	uint64_t primitiveMaterialsAddress = meshBuffer[2];
	return ((uint*)primitiveMaterialsAddress)[primitiveIndex];
}

[shader("raygeneration")]
void rayGenMain()
{
	uint2 pixel = DispatchRaysIndex().xy;

	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	{
		RtxdiConstants* rc = (RtxdiConstants*)pushData.rtxdiConstantsAddress;
		g_rtxdiReservoirs = (RTXDI_PackedDIReservoir*)rc.reservoirBufferAddress;
		g_rtxdiNeighborOffsets = (float2*)rc.neighborOffsetsAddress;
	}

	GIConstants* gc = (GIConstants*)pushData.giConstantsAddress;
	g_giReservoirs = (GIReservoir*)gc.reservoirBufferAddress;

	// 16MB extra data to maintain state over multiple frames
	float4* extraDataIn = (float4*)scene.outputAddress + DispatchRaysDimensions().x * DispatchRaysDimensions().y;
	float4* extraDataOut = (float4*)scene.prevOutputAddress + DispatchRaysDimensions().x * DispatchRaysDimensions().y;

	float4* output = (float4*)scene.outputAddress;

	uint index = pixel.y * DispatchRaysDimensions().x + pixel.x;

	float4 prevColor;
	{
		float4* prevOutput = (float4*)scene.prevOutputAddress;
		prevColor = prevOutput[index];
		// to clear old state out
		if(scene.temporalAccumulator == 0)
			prevColor = 0.0f;
	}

	// See g_outputBufferSpecular's own comment (App.cpp) -- same ping-pong/clear-on-reset
	// pattern as prevColor above, for ray.slang's direct/NEE-specular-only channel (hdrSpecular).
	float4* outputSpecular = (float4*)scene.outputSpecularAddress;
	float4 prevColorSpecular;
	{
		float4* prevOutputSpecular = (float4*)scene.prevOutputSpecularAddress;
		prevColorSpecular = prevOutputSpecular[index];
		if (scene.temporalAccumulator == 0)
			prevColorSpecular = 0.0f;
	}

	if(scene.temporalMixIn <= 0.0f)
	{
		output[index] = prevColor;
		outputSpecular[index] = prevColorSpecular;
		// Rendering is fully paused this frame (no TraceRay2 below, so no payload/surfacePos to
		// derive real values from) -- write safe, harmless defaults so the G-buffer never holds
		// uninitialized/stale data on a frozen frame. See the matching unconditional write at the
		// end of this function for the normal path.
		g_gbufferAlbedo[pixel] = float4(0, 0, 0, 1);
		g_gbufferNormalRoughness[pixel] = float4(0, 1, 0, 1);
		g_gbufferViewZ[pixel] = scene.zFar;
		g_gbufferMotionVector[pixel] = float2(0, 0);
		return;	// could be optimzied
	}

	float2 temporalJitter = scene.temporalJitter;

	float2 fragPos = ((pixel + temporalJitter) / scene.resolution) * 2.0f - 1.0f;
	
	RayDesc ray;
	ray.Origin = S2S_CAMERA_POS();
	ray.TMin = 0.001; // Minimum distance to avoid self-intersection
	ray.TMax = INFINITE; // Maximum distance (infinite for primary rays)
	{
		float4x4 worldToClip = MultiplyMatrices(GetPerspMatrix(), GetViewMatrix());
		float4x4 INV_VIEW_PROJECTION = inverse(worldToClip);

		float3 worldPos;
	    {
			float2 screenPos = fragPos.xy;
			screenPos.y = -screenPos.y;
			float4 worldPosHom = mul(INV_VIEW_PROJECTION, float4(screenPos, S2S_NEAR(), 1));
			worldPos = worldPosHom.xyz / worldPosHom.w;
		}
		ray.Direction = normalize(worldPos - S2S_CAMERA_POS());
	}
	
	Payload payload = (Payload) 0;
	payload.primitiveIndex = -1;
	payload.instanceId = -1;
	
	TraceRay2(
		g_tlas,
        RAY_FLAG_NONE,
        0x3,	// primary and secondary
        0, // hit group offset
        0, // multiplier
        0, // miss shader
        ray,
        payload);

	// Only normalize on an actual hit: on a miss, rayMissMain zeroes the whole payload (including
	// interpolatedNormal), and normalize((0,0,0)) is a 0/0 division -- NaN, which then poisons every
	// downstream use of this value for that pixel.
	if (payload.instanceId != -1)
		payload.interpolatedNormal = normalize(payload.interpolatedNormal);
	
	// scene.frameNoiseSeed is 0 (same pattern every frame) or frame-varying (a different pattern
	// each frame), depending on the "Frame Noise Randomization" checkbox -- see shader_io.h.
//	uint rndState = initRand(dot(DispatchRaysIndex(), uint3(82927, 21313, 1)), 0x12345678 + scene.frameNoiseSeed);
	// tiling noise to get better denoising
	uint rndState = initRand(dot(DispatchRaysIndex() % 32, uint3(82927, 21313, 1)), 0x12345678 + scene.frameNoiseSeed % 32);

	float3 surfacePos = ray.Origin + ray.Direction * payload.minT;

	// Screen-space motion, current -> previous, normalized to a fraction of the screen -- hoisted
	// to function scope (not just the G-buffer block below) since ReSTIR GI's own temporal
	// reprojection (the bounce loop, further down) reuses this SAME pixel's motion vector instead
	// of recomputing an identical reprojection a second time.
	float2 motion = 0.0f;

	// G-buffer targets (g_gbufferAlbedo/NormalRoughness/ViewZ/MotionVector, declared above) --
	// infrastructure for a future denoiser; also read back further down by the EVM_* visualize
	// modes (search "(GBuffer)"), so this runs before any of that rather than at the very end of
	// the function. Unconditional (every pixel, every frame) so nothing ever sees garbage.
	{
		bool hit = payload.instanceId != -1;

		g_gbufferAlbedo[pixel] = float4(hit ? payload.materialColor : float3(0, 0, 0), 1.0f);
		g_gbufferNormalRoughness[pixel] = float4(hit ? payload.interpolatedNormal : float3(0, 1, 0), hit ? payload.roughness : 1.0f);

		float3 posForViewZ = hit ? surfacePos : (ray.Origin + ray.Direction * scene.zFar);
		g_gbufferViewZ[pixel] = mul(GetViewMatrix(), float4(posForViewZ, 1.0f)).z;

		// Screen-space motion, current -> previous, normalized to a fraction of the screen
		// (resolution-independent, typically a small float for ordinary camera motion) rather than
		// raw pixel counts (which range up to a full screen's worth of pixels and don't mean
		// anything without also knowing the resolution). Generalizes the RTXDI branch's own
		// transient screenSpaceMotion computation (further down, RTXDI-only) to run for every pixel
		// regardless of whether that branch executes.
		if (hit)
		{
			RtxdiConstants* rc = (RtxdiConstants*)pushData.rtxdiConstantsAddress;
			// prevClipFromView needs transpose() before use, same as GetPerspMatrix() does for
			// scene.clipFromView -- both are cached/uploaded the same untransposed way on the C++
			// side (see g_rtxdiPrevClipFromView, App.cpp), unlike prevViewFromWorld/
			// scene.viewFromWorld, which already are pre-transposed (see GetViewMatrix()). Using it
			// raw here previously made this "motion" a smoothly-wrong function of world position
			// instead of an actual small frame-to-frame delta.
			float4x4 prevWorldToClip = MultiplyMatrices(transpose(rc.prevClipFromView), rc.prevViewFromWorld);
			float4 h = mul(prevWorldToClip, float4(surfacePos, 1.0f));
			float2 prevScreenPos = ((h.xy / h.w) * float2(0.5f, -0.5f) + 0.5f) * scene.resolution;
			// "+ 0.5f": prevScreenPos is a pixel-CENTER convention coordinate (NDC (0,0) maps to the
			// exact center of the screen, not a corner -- see the *0.5+0.5 remap above), so the
			// current pixel's own reference point must be its center too (pixel+0.5), not its
			// corner (bare pixel). NRD (IN_MV consumer) reconstructs the previous UV as exactly
			// "(pixel+0.5)/resolution + motion" -- confirmed against its own self-check in
			// REBLUR_Validation.cs.hlsl ("viewportPixelUv = (pixelPos+0.5)*rectSizeInv"). Leaving out
			// the "+0.5f" here introduced a constant half-pixel reprojection bias into every frame,
			// degrading temporal reuse -- worst on detailed/curved geometry (e.g. tufted upholstery)
			// where half a pixel can land on a different normal, tripping disocclusion rejection.
			motion = (prevScreenPos - (float2(pixel) + 0.5f)) / scene.resolution;
		}
		g_gbufferMotionVector[pixel] = motion;
	}

	// g_lights (App.cpp): [0] is always the single global light (g_lightPos/
	// g_lightColor/g_lightIntensity), followed by every point light the loaded BSP level's "light"
	// entities parsed into (C3DBSPLevel::m_Lights). Every vertex (primary hit and every indirect
	// bounce alike, see the bounce loop below) picks among these uniformly at random
	// (PickUniformRandomLight) and shadow-tests via perSample.
	Light* lights = (Light*)scene.lightsAddress;

    // start slightly above the surface to avoid hit with own surface
	surfacePos += payload.interpolatedNormal * 0.001f;
//	payload.minT = rayDesc.TMin;

	float3 hdr = payload.emissiveColor;

	// Specular NRD channel (see g_outputBufferSpecular's own comment, App.cpp): only the
	// direct/NEE specular BRDF lobe feeds this -- indirect/GI-bounced light stays folded into hdr
	// above regardless of which lobe it came from, a deliberate scope boundary (see readme.md's own
	// "NRD: split diffuse/specular" note) rather than a full per-lobe path split. hdrSpecular gets
	// NO temporal smoothing of its own (unlike hdr's optional GI/EMA reservoir path below) --
	// specular reflections move very differently under camera motion than diffuse light does, which
	// the simple EMA reservoir was never built to track; NRD's own specular denoiser (when enabled)
	// is squarely the right tool for that instead.
	float3 hdrSpecular = 0.0f;
	// Luminance-weighted average distance to the light(s) driving hdrSpecular, standing in for a
	// true specular-reflection-ray hit distance (NRD's REBLUR_SPECULAR wants "how far did the
	// reflection travel", but this is a shadow-ray NEE term, not a traced reflection) -- REBLUR's
	// hit-distance-based virtual-motion reprojection degrades gracefully with an approximate value,
	// and light distance is the closest available stand-in. Accumulated across every vertex/light
	// sample that contributes to hdrSpecular below, weighted by how much each one actually
	// contributed (so a bright nearby highlight dominates over a faint distant one).
	float specularHitDistNumerator = 0.0f;
	float specularHitDistDenominator = 0.0f;

	bool computeLighting = scene.visualizeMode == VisualizeMode::EEnum::EVM_Shaded;

	if(computeLighting)
	{
		if (payload.instanceId == -1)
		{
			// Miss: use rayMissMain's sky color directly. Running it through the same lighting path as
			// a hit used a zero-initialized normal/material (Payload isn't touched beyond payload.color
			// on miss), which made every background pixel evaluate to black instead of the sky color.
		}
		else
		{
			// Basic path tracing, inline in this loop (no shader-side recursion, no denoiser, one path
			// per pixel): walk the primary hit plus scene.pathTracingBounces further bounces, each one
			// TraceRay for the bounce itself and exactly one shadow ray for direct lighting at that
			// vertex (next-event estimation -- with a single point light and no area light to sample,
			// this is what actually makes the light visible at all; pure BSDF sampling would need to
			// hit a zero-size point by chance). Indirect bounces use a cosine-weighted hemisphere
			// sample against the surface normal, which is the standard basic/diffuse-only choice: for
			// a Lambertian BRDF under cosine-weighted sampling the usual f*cosTheta/pdf throughput
			// term collapses to just the surface albedo (see getCosHemisphereSample, mathlib.slang).

			float3 throughput = 1.0f;
			float3 bounceOrigin = ray.Origin;
			float3 bounceDir = ray.Direction;
			Payload bouncePayload = payload;

			const uint totalHits = 1 + scene.pathTracingBounces; // primary hit + N indirect bounces

			[loop]
			for (uint bounce = 0; bounce < totalHits; ++bounce)
			{
				// Whether temporal resampling smooths THIS vertex's contribution over time -- computed
				// up front (not just before the resampling branch further below) since it also decides
				// where this bounce's own emissive term goes, right after the trace. Covers the
				// primary hit's direct lighting (bounce==0) and the first indirect bounce (bounce==1)
				// alike -- direct lighting isn't a special case here, it's resampled the same way
				// indirect and emissive contributions are (see GI_RESERVOIR_SLOTS_PER_PIXEL's comment,
				// gi_io.h, for why each needs its own reservoir slot rather than sharing one).
				bool giActive = (scene.enableRestirGI != 0) && (bounce <= 1);

				if (bounce > 0)
				{
					RayDesc bounceRay;
					bounceRay.Origin = bounceOrigin;
					bounceRay.Direction = bounceDir;
					bounceRay.TMin = 0.001f;
					bounceRay.TMax = INFINITE;

					bouncePayload = (Payload) 0;
					bouncePayload.primitiveIndex = -1;
					bouncePayload.instanceId = -1;

					TraceRay2(
						g_tlas, RAY_FLAG_NONE,
						0x2, // secondary only
						0, 0, 0, bounceRay, bouncePayload);

					// Deferred into vertexRadiance below instead, when GI resamples this vertex: a
					// self-luminous surface at vertex 2 contributes radiance toward vertex 1 exactly
					// the same way NEE-lit radiance does, so GI should resample it too instead of
					// letting it always bypass resampling unconditionally (confirmed live: emissive
					// geometry looked identical with GI on or off, since this line used to always
					// add it directly here regardless).
					if (!giActive)
						hdr += throughput * bouncePayload.emissiveColor;

					if (bouncePayload.instanceId == -1)	// sky hit
					{
						// No surface at a sky hit, so there's nothing for GI to resample here even
						// when giActive -- fall back to counting it directly instead of dropping it.
						if (giActive)
							hdr += throughput * bouncePayload.emissiveColor;
						break;
					}
					bouncePayload.interpolatedNormal = normalize(bouncePayload.interpolatedNormal);

					// Fog over this bounce segment attenuates everything gathered beyond it (further
					// bounces and their shadow rays), since it all has to travel back through this
					// segment to reach the camera.
					throughput *= EvalFogTransmittance(bouncePayload.minT);
				}

				float3 bounceSurfacePos = bounceOrigin + bounceDir * bouncePayload.minT
					+ bouncePayload.interpolatedNormal * 0.001f;

				float3 V = normalize(bounceOrigin - bounceSurfacePos);

				// See the direct-lighting branch above for why this samples at most
				// kMaxShadowRaysPerPoint lights picked by power, and normalizes by invPdf.
				uint sampleCount = min(kMaxShadowRaysPerPoint, scene.lightCount);

				// Accumulated separately from hdr (instead of the direct "hdr +=" every other branch
				// in this file uses) when temporal resampling is active for this vertex -- see the
				// giActive branch below for why: it needs this vertex's OWN radiance as a standalone
				// value to resample, before deciding how much of it (this frame's fresh value, or a
				// reprojected one) actually reaches hdr. Every other vertex still adds it straight to
				// hdr exactly as before (the "else" branch below).
				//
				// Seeded with this vertex's own emissive term ONLY for bounce>0 (the deferred-emissive
				// handling just above, itself only reachable when bounce>0) -- bounce==0's own
				// emissive is already counted once, unconditionally, at this function's very top
				// (hdr = payload.emissiveColor) regardless of resampling, so seeding it here too for
				// bounce==0 would double-count it. Diffuse-only now -- see hdrSpecular's own comment
				// for where this vertex's specular NEE lobe goes instead.
				float3 vertexRadiance = (giActive && bounce > 0) ? bouncePayload.emissiveColor : 0.0f;
				// This vertex's own specular NEE contribution, accumulated separately (see
				// hdrSpecular's own comment) -- never resampled/EMA-smoothed regardless of giActive,
				// unlike vertexRadiance above.
				float3 vertexRadianceSpecular = 0.0f;

				[loop]
				for (uint s = 0; s < sampleCount; ++s)
				{
					float invPdf;
					uint li = PickUniformRandomLight(scene.lightCount, rndState, invPdf);
					float3 lightPos = JitterLightPosition(lights[li].position, bounceSurfacePos, lights[li].sourceRadius, rndState);
					float3 lightColor = lights[li].color * EvalLightFalloff(length(lightPos - bounceSurfacePos), lights[li].radius) * (invPdf / sampleCount);

					// one shadow ray per sampled light, softened by JitterLightPosition above
					float3 shadow = perSample(bounceSurfacePos, bouncePayload.interpolatedNormal, lightPos, rndState);

					float3 L = normalize(lightPos - bounceSurfacePos);
					float3 directDiffuse, directSpecular;
					EvalGGXBRDFSplit(bouncePayload.interpolatedNormal, V, L, bouncePayload.materialColor, bouncePayload.metallic, max(bouncePayload.roughness, kMinPunctualLightRoughness), bouncePayload.specularOcclusionFromAlbedo, directDiffuse, directSpecular);
					vertexRadiance += directDiffuse * lightColor * shadow;
					float3 specularTerm = directSpecular * lightColor * shadow;
					vertexRadianceSpecular += specularTerm;

					// See specularHitDistNumerator/Denominator's own comment -- weight this light's
					// distance by how much of the (throughput-scaled) specular signal it actually
					// contributed, in the same units hdrSpecular itself accumulates in below.
					float specWeight = Luminance(throughput * specularTerm);
					specularHitDistNumerator += specWeight * length(lightPos - bounceSurfacePos);
					specularHitDistDenominator += specWeight;

					// Fog in-scattering is deliberately NOT resampled below even when ReSTIR GI is
					// active for this vertex: it's a smooth, comparatively small-magnitude term next
					// to surface NEE lighting, computed here from THIS frame's own freshly-traced
					// segment either way, regardless of which reservoir sample wins resampling.
					hdr += throughput * EvalInscattering(bounceDir, bounceSurfacePos, lightPos, lightColor, shadow);
				}

				hdrSpecular += throughput * vertexRadianceSpecular;

				if (giActive)
				{
					// Temporal resampling (exponential moving average) of this vertex's contribution
					// -- see gi_reservoir.slang's header comment for exactly what this does and does
					// not do. `throughput` here is bounce 0's own BRDF/pdf-weighted lobe-sampling
					// throughput for bounce==1 (1.0 unchanged for bounce==0, the primary hit); combined
					// with vertexRadiance it's exactly the value the "else" branch below would
					// otherwise add to hdr unconditionally, before any smoothing.
					float3 candidateContribution = throughput * vertexRadiance;

					// bounce (0 or 1, since giActive requires bounce<=1) selects which of this pixel's
					// GI_RESERVOIR_SLOTS_PER_PIXEL reservoir slots to blend with -- direct lighting and
					// the first indirect bounce are different quantities at different vertices, so
					// each gets its own slot/history rather than sharing one. Reprojects THIS
					// iteration's own vertex (bounceSurfacePos/bouncePayload) -- at bounce==0 that's
					// exactly the primary hit (bouncePayload==payload, bounceSurfacePos==surfacePos per
					// the setup above the loop), and at bounce==1 it's the actual indirect-bounce
					// surface, not the primary hit's.
					GIReservoir prevReservoir = ValidateAndReprojectGIReservoir(pixel, bounce, motion, bounceSurfacePos, bouncePayload.interpolatedNormal, scene, gc);

					GIReservoir merged;
					merged.contribution = IsValidGIReservoir(prevReservoir)
						? lerp(candidateContribution, prevReservoir.contribution, gc.historyWeight)
						: candidateContribution;
					merged.valid = 1;
					// Persist for next frame's reprojection -- this vertex's OWN generating surface,
					// same reasoning as the reprojection call just above.
					merged.generatingViewZ = mul(GetViewMatrix(), float4(bounceSurfacePos, 1.0f)).z;
					merged.generatingNormalPacked = RTXDI_EncodeNormalizedVectorToSnorm2x16(bouncePayload.interpolatedNormal);

					hdr += merged.contribution;

					g_giReservoirs[gc.outputBufferIndex * gc.reservoirPitch + index * GI_RESERVOIR_SLOTS_PER_PIXEL + bounce] = merged;
				}
				else
				{
					hdr += throughput * vertexRadiance;
				}

				if (bounce + 1 == totalHits)
					break; // last hit: no need to sample a further bounce direction

				// Stochastic lobe selection for the next bounce direction: each bounce either samples
				// the GGX specular lobe (importance-sampled around the half-vector, so low roughness
				// gives tight near-mirror bounces and high roughness spreads towards diffuse-like
				// scattering) or the existing cosine-weighted diffuse lobe -- picking one per bounce,
				// weighted by specProb, rather than evaluating both, is what keeps this "one ray per
				// bounce" instead of turning into its own multi-sample estimator. V (view direction,
				// toward the previous vertex) was already computed above for the direct-lighting term.
				float3 f0 = GetF0(bouncePayload.materialColor, bouncePayload.metallic, bouncePayload.specularOcclusionFromAlbedo);
				float specProb = saturate(max(f0.r, max(f0.g, f0.b)));

				if (nextRand(rndState) < specProb)
				{
					float3 H = importanceSampleGGX(rndState, bouncePayload.interpolatedNormal, bouncePayload.roughness);
					bounceDir = reflect(-V, H);

					float NdotL = dot(bouncePayload.interpolatedNormal, bounceDir);
					if (NdotL <= 0.0f)
						break; // sampled microfacet reflects below the surface -- nothing further to gather on this path

					float NdotV = saturate(dot(bouncePayload.interpolatedNormal, V));
					float NdotH = saturate(dot(bouncePayload.interpolatedNormal, H));
					float VdotH = saturate(dot(V, H));
					float3 F = F_Schlick(f0, VdotH);
					float G = G1_GGX(NdotV, bouncePayload.roughness) * G1_GGX(NdotL, bouncePayload.roughness);
					// Standard NDF-importance-sampling throughput (Karis, "Real Shading in Unreal
					// Engine 4", 2013): the D term and most of the analytic BRDF's
					// 1/(4*NdotV*NdotL) cancel against the sampling PDF, leaving
					// F*G*VdotH/(NdotH*NdotV).
					throughput *= F * G * VdotH / max(NdotH * NdotV, 1e-4f) / specProb;
				}
				else
				{
					bounceDir = normalize(getCosHemisphereSample(rndState, bouncePayload.interpolatedNormal));
					// glTF energy conservation: metals have no diffuse component (same convention
					// EvalGGXBRDF uses); dividing by (1-specProb) keeps the stochastic estimator
					// unbiased since this lobe is only taken a (1-specProb) fraction of the time.
					throughput *= bouncePayload.materialColor * (1.0f - bouncePayload.metallic) / (1.0f - specProb);
				}

				bounceOrigin = bounceSurfacePos;
			}
		}
	}

	// Fog over the primary ray's camera-to-surface segment (start ray.Origin, end surfacePos,
	// distance payload.minT -- ray.Direction is normalized, so minT already is the Euclidean
	// distance). This is on top of the shadow-ray/bounce-segment fog already folded into hdr above
	// via EvalFogTransmittance, and covers the last leg back to the camera. Applies to a miss too
	// (payload.minT is INFINITE there): the flat sky color set above is meant to be fully
	// extinguished -- see below for what replaces it.
	hdr *= EvalFogTransmittance(payload.minT);
	hdrSpecular *= EvalFogTransmittance(payload.minT);

	if (payload.instanceId == -1)
	{
		// The sky has no finite surface to sample in-scattering at (unlike the hit branches above,
		// which sample it at the surface and let the multiply above attenuate it over the full
		// segment), so this is added after that multiply instead of before -- sampled essentially
		// at the camera (distance ~0), it needs no attenuation of its own. Without this, fog
		// visibly cut off right at the horizon: a foggy, glowing ground giving way to an untouched,
		// unglowing sky, and no forward-scatter halo when looking toward the light. Same
		// uniform light sampling + invPdf normalization, same as the bounce loop above.
		uint sampleCount = min(kMaxShadowRaysPerPoint, scene.lightCount);

	    [loop]
		for (uint s = 0; s < sampleCount; ++s)
		{
			float invPdf;
			uint li = PickUniformRandomLight(scene.lightCount, rndState, invPdf);
			float3 lightPos = JitterLightPosition(lights[li].position, ray.Origin, lights[li].sourceRadius, rndState);
			float3 lightColor = lights[li].color * EvalLightFalloff(length(lightPos - ray.Origin), lights[li].radius) * (invPdf / sampleCount);
			float3 skyVisibility = perSample(ray.Origin, ray.Direction, lightPos, rndState);
			hdr += EvalInscattering(ray.Direction, ray.Origin, lightPos, lightColor, skyVisibility);
		}
	}

	float3 ldr = payload.materialColor;

	if (scene.visualizeMode == VisualizeMode::EEnum::EVM_Shaded)
	{
		ldr = hdr; // filmicToneMapping(hdr);
	}
	// EVM_NormalWS/AlbedoColor/Roughness/ViewZ/MotionVector ("(GBuffer)" in the UI, see
	// shader_io.h's typeUI) read back the G-buffer textures written earlier in this function
	// (search "G-buffer targets"), rather than payload/local values directly -- this way the
	// visualization actually exercises the same binding/format the textures are stored with,
	// instead of only ever showing the pre-write value regardless of a G-buffer bug.
	if (scene.visualizeMode == VisualizeMode::EEnum::EVM_NormalWS)
	{
		ldr = g_gbufferNormalRoughness[pixel].xyz * 0.5f + 0.5f;
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_AlbedoColor)
	{
		ldr = g_gbufferAlbedo[pixel].rgb;
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_PositionWS)
	{
		ldr = frac(surfacePos);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_DepthNorm)
	{
		// todo: dot product with forward
		float normDepth = saturate((length(surfacePos - GetCameraPos()) - scene.zNear) / (scene.zFar - scene.zNear));
		ldr = pow(normDepth, float3(1 / 6.0f, 1 / 3.0f, 1 / 1.0f)); // todo
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_DepthWS)
	{
		ldr = frac(payload.minT);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_InstanceIndex)
	{
		ldr = s2h_indexToColor(payload.instanceIndex);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_Roughness)
	{
		ldr = g_gbufferNormalRoughness[pixel].w.xxx;
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_ViewZ)
	{
		// Banded via frac(), matching EVM_DepthWS's style, since raw view-space Z is usually too
		// large a range to see meaningful variation in a single 0..1 color.
		ldr = frac(g_gbufferViewZ[pixel] * 0.1f);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_MotionVector)
	{
		// Motion is stored as a fraction of the screen (see the G-buffer write above), typically a
		// small float -- scaled up for visibility here so ordinary motion is still visible as a
		// color shift instead of rounding away to the same mid-gray as a static camera.
		float2 motionVis = g_gbufferMotionVector[pixel];
		ldr = float3(saturate(motionVis * 4.0f + 0.5f), 0.5f);
	}


	float4 color = float4(ldr, payload.instanceId != -1 ? 1.0f : 0.0f);

	// visualize interpolated normal (looks good)
//	ldr = normalize(payload.interpolatedNormal) * 0.5f + 0.5f;
	
//	float4 newColor = float4(s2h_accurateSRGBToLinear(color.rgb), color.a);
	float4 newColor = color;

	if (scene.temporalAccumulator != 0)
	{
		newColor.rgb = newColor.rgb + prevColor.rgb;
		hdrSpecular += prevColorSpecular.rgb;
	}

	newColor.a = payload.minT;

	{
		uint2 resolution = (uint2)scene.resolution;
		float4* interactionResult = (float4*)scene.outputAddress + resolution.x * resolution.y + kInteractionResultOffset;
		int pickedIndex = asint(interactionResult[0].w);
		// interactionResult[1].x is interaction.slang's own bMaterial flag (LMB+Ctrl) -- whether
		// pickedIndex above is a material index or an entity index, see its own comment.
		bool pickedIsMaterial = asuint(interactionResult[1].x) != 0;

		// interactionResult[0].w is interaction.slang's pick result: with Ctrl held, the hit
		// triangle's material index (re-derived via GetMaterialID(), see its own comment, so every
		// pixel sharing that material highlights, not just the one entity/triangle under the
		// cursor); otherwise the entity index (InstanceID(), see Payload::instanceId's comment) --
		// compare against payload.instanceId, not payload.instanceIndex (a different, TLAS-array-
		// position index space that doesn't line up with entities once any scene object has no
		// mesh, e.g. cameras/lights -- see Renderer::createTopLevelAS's comment).
		bool isMatch = payload.instanceId != -1 && (pickedIsMaterial
			? pickedIndex == (int)GetMaterialID(payload.instanceId, payload.primitiveIndex)
			: pickedIndex == payload.instanceId);
		if (pickedIndex != -1 && isMatch)
			newColor.a *= -1.0f;
	}


	output[index] = newColor;

	{
		// See g_outputBufferSpecular's own comment (App.cpp) -- same per-pixel indexing
		// as output[] above (outputSpecular itself declared near prevColorSpecular, above).
		float specularHitDist = specularHitDistDenominator > 0.0f ? (specularHitDistNumerator / specularHitDistDenominator) : 0.0f;
		outputSpecular[index] = float4(hdrSpecular, specularHitDist);
	}

	// This frame's packed normal, read back next frame via RtxdiConstants::prevNormalAddress for
	// RTXDI's temporal/spatial neighbor validity checks (RAB_GetGBufferSurface, RtxdiBridge.slang)
	// -- paired with newColor.a (payload.minT) above, which already doubles as that lookup's depth.
	{
		RtxdiConstants* rc = (RtxdiConstants*)pushData.rtxdiConstantsAddress;
		uint* normals = (uint*)rc.normalAddress;
		normals[index] = RTXDI_EncodeNormalizedVectorToSnorm2x16(payload.interpolatedNormal);
	}

}

[shader("miss")]
void rayMissMain(inout Payload payload)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

	payload = (Payload) 0;
	payload.materialColor = float3(0.5f, 0.5f, 0.5f);
	payload.emissiveColor = getSkyColor(WorldRayDirection());
	payload.primitiveIndex = -1;
	payload.instanceId = -1;
	payload.instanceIndex = -1;
	payload.minT = INFINITE;
}

[shader("closesthit")]
void rayClosestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	float3 bary = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

	uint entityID = InstanceID(); // gl_InstanceCustomIndexEXT, user defined -- the *entity's* index (see Renderer::createTopLevelAS)
	uint instanceIndex = InstanceIndex(); // Which instance was hit, system generated
	// entityID -> meshID: one indirection, since a TLAS instance can only carry one InstanceID() and
	// interaction.slang's pick ray needs that to be the entity index, not the mesh index (materials
	// always come from Mesh::m_primitiveMaterials below, never from instance data) -- see
	// RayPushData::entityMeshIndexAddress's comment.
	uint meshID = ((uint*)pushData.entityMeshIndexAddress)[entityID];
	uint triID = PrimitiveIndex(); // Which triangle within the mesh was hit
	
	payload.instanceIndex = instanceIndex;
	payload.primitiveIndex = triID; // triangle or splat
	payload.minT = RayTCurrent();

	// [meshID] = { IB, VB, primitiveMaterials }
	uint64_t* meshBuffer = (uint64_t*)pushData.meshBufferAddress + 3 * meshID;
	// IB
	uint* indexBuffer = (uint*)meshBuffer[0];
	// VB
	Vertex * vertexBuffer = (Vertex *)meshBuffer[1];

	Vertex v0 = vertexBuffer[indexBuffer[triID * 3 + 0]];
	Vertex v1 = vertexBuffer[indexBuffer[triID * 3 + 1]];
	Vertex v2 = vertexBuffer[indexBuffer[triID * 3 + 2]];

	float3 vertexNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
	float2 uv = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;
	// .w = handedness sign (glTF convention) -- see the TBN construction below.
	float4 tangent = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;

	float2 lightmapUV = bary.x * v0.lightmapUV + bary.y * v1.lightmapUV + bary.z * v2.lightmapUV;

	float3 triangleNormal = normalize(cross(v1.position - v0.position, v2.position - v0.position));
//	vertexNormal = triangleNormal;

	// Per-primitive material (see Mesh::m_primitiveMaterials): every mesh always has this fully
	// populated, one valid material index per triangle -- lets one TLAS instance carry multiple
	// materials, matching the rasterizer's per-submesh draw loop, which instanceCustomIndex alone
	// couldn't (only one value per instance).
	uint64_t primitiveMaterialsAddress = meshBuffer[2];
	uint materialID = ((uint*)primitiveMaterialsAddress)[triID];

	MaterialData* materialBuffer = (MaterialData*)pushData.materialBufferAddress + materialID;

	// Normals need the inverse-transpose of the instance's object-to-world transform, not the
	// transform itself -- using it directly (or skipping the transform, as before) only happens to
	// look right for instances with no rotation (identity-ish transforms), and silently breaks for
	// any rotated/scaled instance. mul(v, WorldToObject3x4()) is the standard HLSL/Slang ray tracing
	// idiom for this: WorldToObject3x4() is the inverse of the instance transform, and mul(vector,
	// matrix) here computes transpose(matrix) * vector -- i.e. the inverse-transpose.
	float3 interpolatedNormal = mul(vertexNormal, (float3x3)WorldToObject3x4());
	
	if (dot(interpolatedNormal, WorldRayDirection()) > 0.0f)
	//	if (dot(triangleNormal, vertexNormal) < 0.0f)
		interpolatedNormal = -interpolatedNormal;

	// Normal mapping: tangent transforms with the instance's regular object-to-world matrix (unlike
	// the normal above, a tangent is a direction ALONG the surface, not perpendicular to it, so it
	// doesn't need the inverse-transpose). Re-orthogonalized against the (already flip-corrected)
	// shading normal via Gram-Schmidt -- interpolation across a triangle doesn't keep T exactly
	// perpendicular to N even when the source vertex data was clean.
	float3 rawTangent = mul((float3x3)ObjectToWorld3x4(), tangent.xyz);
	float3 worldTangent;
	if (dot(rawTangent, rawTangent) > 1e-12f)
	{
		worldTangent = normalize(rawTangent);
	}
	else
	{
		// Degenerate/absent tangent (Vertex::tangent left at its zero default -- e.g. a procedural
		// mesh with no real UV-derived tangent yet, like RenderComponent::createPlane's own "todo:
		// tangent", or a glTF file with no TANGENT attribute) -- synthesize *some* vector
		// perpendicular to the normal instead of normalize()-ing (0,0,0) into NaN, which would
		// otherwise poison every subsequent lighting calculation for this pixel via 0*NaN=NaN
		// (confirmed live: an entire procedural ground plane went black once normal mapping started
		// unconditionally building a TBN matrix here). Not UV-aligned, so a REAL normal map assigned
		// to tangent-less geometry would still look wrong the same way BSP's old GetOtherBaseVec-
		// based tangent did (see 3DBSPLevel.cpp's CalculateTangentsFromUV comment) -- but every
		// material's normal map defaults to flat (0,0,1) (g_hndDefaultNormalMap), which reads back
		// correctly regardless of T/B's actual direction, so this only matters once a real normal
		// map is assigned to tangent-less content.
		float3 up = abs(interpolatedNormal.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
		worldTangent = normalize(cross(up, interpolatedNormal));
	}
	float3 T = normalize(worldTangent - interpolatedNormal * dot(interpolatedNormal, worldTangent));
	float3 B = cross(interpolatedNormal, T) * tangent.w;

	// materialBuffer.normalMapTextureId is never kInvalidTextureId -- a material with no normal map
	// of its own already resolved to the flat (0,0,1) engine default at the CPU-side MaterialData
	// fill (see g_hndDefaultNormalMap, App.cpp) -- so this always samples something
	// meaningful, no branch needed for "does this material even have a normal map".
	uint normalTexId = min(materialBuffer.normalMapTextureId, kMaxRayTracingTextures - 1u);
	float3 tangentSpaceNormal = g_textures[normalTexId].SampleLevel(g_sampler, uv, 0).rgb * 2.0f - 1.0f;
	interpolatedNormal = normalize(tangentSpaceNormal.x * T + tangentSpaceNormal.y * B + tangentSpaceNormal.z * interpolatedNormal);

    // Traditional binding (not descriptor-heap) -- see g_textures' declaration comment for why.
	payload.materialColor = materialBuffer.albedoColor;

	if(materialBuffer.albedoTextureFade > 0.0f && scene.enableTextures != 0)
	{
		// the material's own glTF texture if it has one, otherwise the debug "Image 1"/"Image 2"
		// toggle (scene.texId) -- see resolveTextureId() in mathlib.slang. Clamped to this
		// pipeline's fixed-size g_textures[] array (raster's descriptor heap has no such cap).
		// Albedo textures upload into a VK_FORMAT_R8G8B8A8_SRGB image/view (see Texture.cpp's
		// m_isSRGB), so the GPU's texture unit already decodes gamma->linear here for free as part
		// of normal sampling/filtering -- no shader-side conversion needed.
		uint texId = min(resolveTextureId(materialBuffer.textureId, scene.texId), kMaxRayTracingTextures - 1u);
		payload.materialColor *= lerp(1.0f, g_textures[texId].SampleLevel(g_sampler, uv, 0).rgb, materialBuffer.albedoTextureFade);
	}

	// hacky
	if (scene.visualizeMode == VisualizeMode::EEnum::EVM_UV)
		payload.materialColor = float3(frac(uv*10), 0);					// todo: *10
	if (scene.visualizeMode == VisualizeMode::EEnum::EVM_LightmapUV)
		payload.materialColor = float3(frac(lightmapUV * 10), 0);		// todo: *10 
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_MaterialID)
		payload.materialColor = s2h_indexToColor(materialID);
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_MeshID)
		payload.materialColor = s2h_indexToColor(meshID);
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_Lightmap)
	{
			// descriptor heaps 
//			uint texId = resolveTextureId(materialBuffer.lightmapTextureId, scene.texId);
//			Texture2D    tex = Texture2D.Handle(uint2(texId, 0));
//			SamplerState samp = SamplerState.Handle(uint2(0, 0));
//			lightmap = tex.SampleLevel(samp, lightmapUV, 0).rgb;
		uint mrTexId = min(materialBuffer.lightmapTextureId, kMaxRayTracingTextures - 1u);
		payload.materialColor = g_textures[mrTexId].SampleLevel(g_sampler, lightmapUV, 0).rgb;
	}

	payload.emissiveColor = materialBuffer.emissiveColor * materialBuffer.emissiveIntensity;
	payload.roughness = materialBuffer.roughness;
	payload.metallic = materialBuffer.metallic;
	payload.specularOcclusionFromAlbedo = materialBuffer.specularOcclusionFromAlbedo;

	// glTF metallic-roughness texture: G=roughness, B=metallic, multiplied with the scalar
	// factors above (no fade -- glTF has no blend factor for this texture, unlike albedo).
	// Clamped to this pipeline's fixed-size g_textures[] array, same as the albedo texture above.
	if (materialBuffer.metallicRoughnessTextureId != kInvalidTextureId)
	{
		uint mrTexId = min(materialBuffer.metallicRoughnessTextureId, kMaxRayTracingTextures - 1u);
		float4 mr = g_textures[mrTexId].SampleLevel(g_sampler, uv, 0);
		payload.roughness *= mr.g;
		payload.metallic *= mr.b;
	}
	payload.interpolatedNormal = interpolatedNormal;
	payload.instanceId = entityID;
}
