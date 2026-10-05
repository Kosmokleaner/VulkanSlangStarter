#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __SLANG__
#define STATIC_CONST static const
static const float FLT_MAX = asfloat(0x7F7FFFFFu);
#else
typedef vec2 float2;
typedef vec3 float3;
typedef vec4 float4;
typedef mat4 float4x4;
#define STATIC_CONST const
#endif

// Vertex layout
STATIC_CONST int LVPosition = 0;
STATIC_CONST int LVNormal= 1;
STATIC_CONST int LVTangent = 2;
STATIC_CONST int LVTexCoord = 3;
STATIC_CONST int LVLightmapUV = 4;
//STATIC_CONST int LVColor    = 4;

// Ray tracing's traditional (non-heap) texture array size -- see g_textures in ray.slang and
// Renderer::m_rtTextureViews / Shader::createRay's binding 4 for why ray tracing can't just use
// the raster path's descriptor heap. A texture ID beyond this cap gets clamped to the last valid
// slot (ray.slang: min(resolveTextureId(...), kMaxRayTracingTextures - 1u)) instead of rejected --
// confirmed live as an entire BSP level (setup #2, ospdm2.bsp) rendering every surface with
// whatever ONE texture happens to occupy that last slot, once its real per-surface texture/lightmap
// count (typically 100+ for a Quake 3 map -- far more than the original budget of 32, itself
// already reduced to effectively 25 by the 7 non-material heap entries below) ran past it.
//
// 7 non-material entries always precede real scene textures in the SAME bindless heap
// Texture::uploadPixels() draws its index from: g_gbufferMotionVector, m_renderTarget's
// color/depth, and the two TAA history buffers (storage + sampled, see
// EnsureTaaHistoryCapacity/onViewportSizeChange, App.cpp) -- registered once, very
// early (App::init()/the first resize), before any real scene texture ever loads. 256
// leaves generous headroom above that for real per-scene textures on top; bump further (and adjust
// this count) if a future scene's texture budget still doesn't fit.
STATIC_CONST uint kMaxRayTracingTextures = 256;
// MaterialData::textureId sentinel meaning "this material has no per-material texture" -- falls
// back to whatever the caller's other texture-selection mechanism provides (e.g. SceneInfo::texId's
// debug toggle in shader.rast.slang/ray.slang).
STATIC_CONST uint kInvalidTextureId = 0xFFFFFFFFu;
// Internal sentinel used only while building Mesh::m_primitiveMaterials (see
// Mesh::buildPrimitiveMaterials) to catch a triangle no submesh covers -- never valid in the
// finished buffer ray.slang's rayClosestHitMain reads (every triangle always has a real material).
STATIC_CONST uint kInvalidMaterialIndex = 0xFFFFFFFFu;
// Max entries g_lights (App.cpp) can hold: the single global light plus every point
// light the loaded BSP level's "light" entities parsed into (C3DBSPLevel::m_Lights).
STATIC_CONST uint kMaxLights = 256;

// Offset (in float4 units, past the resolution.x*resolution.y pixel buffer) within
// scene.outputAddress's "extra data" trailing region (see EnsureOutputBufferCapacity,
// App.cpp) where interaction.slang's rayGenMain stores its single mouse-pick ray's
// result -- read by post.slang's fragmentMain the same frame to print it via s2h_printInt().
// Distinct from the 0..10 range laser.slang/ray.slang's TraceRay2 use for the laser's hit-point
// list, which lives in scene.prevOutputAddress's extra region instead (a different buffer within
// the same frame, so there's no actual overlap either way).
STATIC_CONST uint kInteractionResultOffset = 16;

#if defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__GNUC__) || defined(__clang__)
#define _CPP
#endif

#ifdef _CPP
#pragma pack(push, 1)
#endif

// Buffer-reference/raw-pointer struct access (e.g. (SceneInfo*)pushData.sceneInfoAddress) has no
// automatic matrix-layout reflection the way a traditional cbuffer/ConstantBuffer<T> binding does,
// so mul() and explicit row indexing (mat[i], as MatrixVectorMultiply in mathlib.slang uses) can
// silently disagree on majorness for the exact same bytes -- this was the root cause of a
// hard-to-diagnose device-lost/garbage-output bug in shader.rast.slang's worldFromClip. Declaring
// these fields row_major forces mul() to agree with the row-indexed convention this codebase's C++
// side already targets (matrices are uploaded via glm::transpose() specifically for that
// convention -- see e.g. sceneInfo.viewFromWorld in App.cpp). C++-only: no-op.
#ifdef _CPP
#define ROW_MAJOR
#else
#define ROW_MAJOR row_major
#endif


struct VisualizeMode
{
	enum EEnum {
		EVM_Shaded,
		// G-buffer targets, grouped together right after Default -- all of these read back the
		// actual g_gbufferX texture (ray.slang), not just a payload/local value computed some other
		// way; see their "(GBuffer)" UI labels below and ray.slang's comment at the same spot.
		EVM_NormalWS,
		EVM_AlbedoColor,
		EVM_Roughness,
		// linear view-space depth, banded via frac() (same style as EVM_DepthWS below)
		EVM_ViewZ,
		// color-coded 2D screen-space offset
		EVM_MotionVector,
		//		EVM_NormalTS,
		//		EVM_NormalOS,
		//		EVM_PositionOS,
		EVM_PositionWS,
		// in 0..1 range, zNear: 0, zFar:1
		EVM_DepthNorm,
		// Euclidian Distance to the camera
		EVM_DepthWS,
		// frac(uv), 0
		EVM_UV,
		// frac(lightmapUV), 0
		EVM_LightmapUV,
		//
		EVM_Lightmap,
		//
		EVM_MaterialID,
		//
		EVM_MeshID,
		//
		EVM_InstanceIndex,
	};
#ifdef _CPP
	// "(GBuffer)" marks every mode that reads back one of the g_gbufferX textures (ray.slang)
	// rather than a payload/local value computed some other way.
	static constexpr char* typeUI = (char*)"Shaded\0NormalWS (GBuffer)\0AlbedoColor (GBuffer)\0Roughness (GBuffer)\0ViewZ (GBuffer)\0MotionVector (GBuffer)\0PositionWS\0DepthNorm\0DepthWS\0UV\0LightmapUV\0Lightmap\0MaterialID\0MeshID\0InstanceIndex\0";
#endif
};

// One point light; see g_lights (App.cpp), rebuilt on the CPU every frame: [0] is
// always the single global light (g_lightPos/g_lightColor/g_lightIntensity), followed by every
// point light the loaded BSP level's "light" entities parsed into (C3DBSPLevel::m_Lights).
struct Light
{
	float3 position;
	// intensity-scaled but NOT distance-attenuated -- the value at the light's own position; see
	// EvalLightFalloff (mathlib.slang), applied per-shaded-point using distance and radius below.
	float3 color;
	// distance at which this light's contribution reaches 0 (see EvalLightFalloff)
	float radius;
	// Physical size of the light source, for soft shadows (ray.slang jitters the shadow ray's
	// target within this radius) -- NOT the same as radius above, which is a falloff distance.
	float sourceRadius;
};

// Scene information, stored in a GPU buffer and updated once per frame via vkCmdUpdateBuffer.
// The shader accesses it through a buffer device address (buffer reference), not a descriptor set.
struct SceneInfo
{
	uint64_t outputAddress;		 // float4, this frame's ray-traced output (write target)
	uint64_t prevOutputAddress;	 // float4, previous frame's ray-traced output (read-only; double-buffered for temporal smoothing, see temporalFade)
	// float4 (specular RGB, proxy hit distance), this frame's direct/NEE specular-lobe-only signal
	// (write target) -- see ray.slang's hdrSpecular and g_outputBufferSpecular (App.cpp)
	// for what does and doesn't feed this. Double-buffered/ping-ponged exactly like outputAddress/
	// prevOutputAddress above (same reason: Reference Accumulation mode needs last frame's specular
	// value too, to keep accumulating it the same way it already accumulates diffuse) -- NRD's own
	// dispatch only ever touches the "current" slot, same as it does for the diffuse buffer.
	uint64_t outputSpecularAddress;
	uint64_t prevOutputSpecularAddress;
	float2 resolution;         // Viewport resolution, todo: uint and 1/xy

	float animValue;          // Animation value (sine wave)
	int numData;            // Number of points in the data buffer #points
	int texId;              // Which texture to sample from the descriptor heap
	uint visualizeMode; //

	ROW_MAJOR float4x4 viewFromWorld; // aka view, worldToView
	ROW_MAJOR float4x4 clipFromView; // aka projection, viewToClip
	ROW_MAJOR float4x4 worldFromView; // aka invView, ViewToWorld
	ROW_MAJOR float4x4 viewFromClip; // aka invProjection, clipToView
	ROW_MAJOR float4x4 worldFromClip;
	// Previous frame's camera matrices -- taa_resolve.slang's own reprojection when rayTracing==0
	// (see taaMotionVectorHeapIndex's comment: ray tracing's own g_gbufferMotionVector is used
	// instead when rayTracing!=0). Kept right after the current-frame matrix block above (not at
	// this struct's tail with the other new TAA fields) so a float4x4's natural alignment can't
	// silently disagree between C++'s pack(1) layout and Slang's -- see this file's own note on
	// SceneInfo alignment; every matrix-sized field in this struct lives in this one contiguous
	// block for that reason. Same cache/update pattern as RtxdiConstants::prevViewFromWorld/
	// prevClipFromView (rtxdi_io.h) -- duplicated here rather than reused because this is read from
	// a plain GraphicsPushData-based shader with no RtxdiConstants pointer of its own.
	ROW_MAJOR float4x4 prevViewFromWorld;
	ROW_MAJOR float4x4 prevClipFromView;

	float3 wsCameraPos;
	// 0/1
	uint rayTracing;

	// in sRGB
	float3 clearColor;
	float zNear;

	// basic path tracing (rayGenMain, ray.slang): 0 = off, direct lighting only (single hit, no
	// bounces); N>0 = also trace N indirect diffuse bounces inline in rayGenMain's loop. One
	// path per pixel either way -- no extra samples, no denoiser.
	uint pathTracingBounces;
	uint frameNoiseSeed;
	// 0/1
	uint drawLights;
	// 0:hide/1:draw HUD
	uint infoHUD;

	uint temporalAccumulator;
	// 1/(temporalAccumulator+1)
	float invTemporalAccumulator;
	// 0.0f / 1.0f
	float temporalMixIn;
	// for frame Jitter
	uint frameIndex;

	// 0..1, 0.5 if jitter is disabled
	float2 temporalJitter;
	// 0..1, 0.5 if jitter is disabled
	float2 temporalJitterPrev;

	// g_lights (App.cpp), rebuilt on the CPU every frame -- see Light's declaration
	// comment above for what's at each index. Light picking/importance is now handled by RTXDI
	// (ReSTIR DI, see RtxdiBridge.slang/rtxdi_io.h) instead of a CPU-built CDF.
	uint64_t lightsAddress; // Light*

	// Basic homogeneous participating media (ray.slang: EvalFogTransmittance/EvalInscattering). A
	// per-channel extinction/scattering coefficient is fogDensity * fogColor (single-scattering
	// albedo of 1, i.e. no separate absorption); 0 density disables fog entirely. fogColor also
	// tints extinction (e.g. a blue fogColor extinguishes non-blue wavelengths faster, so blue
	// survives at range).
	float3 fogColor;
	float fogDensity;

	// world space
	float3 laserPos;
	// in radiants
	float laserAngle;

	// Henyey-Greenstein asymmetry (-1..1): >0 favors forward scattering (bright halo looking
	// toward a light through the fog), <0 favors backscattering (toward the viewer), 0 isotropic.
	float fogAsymmetry;
	// Vertical FOV, in degrees -- clipFromView above is already built from this (see
	// updateSceneBuffer(), App.cpp); carried here too as a raw value for anything that
	// wants the angle directly instead of deriving it back out of the projection matrix.
	float fieldOfViewDeg;
	uint lightCount;
	uint enableTextures;
	// RTXDI (ReSTIR DI) is no longer wired into rayGenMain -- kept only so the SceneInfo layout
	// doesn't shift (see shader_io.h's own note on field-order/alignment risk); always 0.
	uint enableRtxdi;
	// 0/1 -- see g_enableRestirGI (App.cpp)'s comment: temporal (exponential-moving-
	// average) resampling of per-vertex path-tracing contributions (ray.slang's basic path-tracing
	// bounce loop) -- applies uniformly to direct lighting at the primary hit and the first
	// indirect bounce alike, the same way emissive already applies at every vertex.
	uint enableRestirGI;

	// 0/1 -- gates the filmic tonemapper (post.slang, mathlib.slang's filmicToneMapping) applied
	// after ray tracing integration/exposure; off leaves the linear HDR color as-is (still goes
	// through the final linear->sRGB encode). See g_enableTonemapper (App.cpp).
	uint enableTonemapper;

	// Multiplies the sampled lightmap before it darkens materialColor (shader.rast.slang: "hdr =
	// lightmap * materialColor"). 1.0 = use the baked texel value as-is. A too-contrasty/crushed-
	// blacks look reported live points at the lightmap's overall brightness/scale being off, not
	// its gamma curve -- baked lightmaps are stored as raw linear radiosity, never gamma/sRGB-
	// encoded like a photographic texture (see the lightmap Texture's own m_isSRGB=false at load
	// time, 3DBSPLevel.cpp), so there's no decode step to get wrong here. Quake-era lightmaps were
	// conventionally baked deliberately dark ("overbright bits"), relying on the
	// engine to scale them back up (often ~2x) at render time so bright areas don't clip during
	// baking; this renderer currently applies no such compensation. Live tunable instead of
	// guessing the right constant blind. See g_lightmapIntensity (App.cpp).
	float lightmapIntensity;

	// FLT_MAX, FLT_MAX if not valid
	float2 pxMousePos;
	// bit 0: LMB, bit 31:Ctrl
	uint mouseButtons;
	float exposureMul;

	float zFar;
	float laserIntensity;

	// Image-based lighting (shader.rast.slang only, for now -- see IBLEnvironment.h). Heap indices
	// for TextureCube.Handle(uint2(index, 0)); iblSpecularMipCount is the specular map's real mip
	// count (roughness 0 = mip 0, increasing roughness maps toward the last mip -- see
	// fragmentMain's own comment for the current, deliberately simple linear formula).
	uint enableIBL;
	uint iblSpecularHeapIndex;
	uint iblDiffuseHeapIndex;
	uint iblSpecularMipCount;

	// Custom TAA resolve (taa_resolve.slang), see its own header comment. Heap indices (bindless,
	// Texture2D/RWTexture2D.Handle(uint2(index,0))) for: this frame's raw jittered color and depth
	// (both from m_renderTarget -- depth only meaningful when rayTracing==0, see taaMotionVector's
	// comment), ray tracing's own g_gbufferMotionVector (only meaningful when rayTracing!=0, reused
	// read-only here instead of recomputing it), and the two ping-pong history buffers -- which one
	// is "this frame's history to read" vs "this frame's resolved output to write" swaps every frame
	// (see g_taaHistoryIndex, App.cpp).
	uint taaCurrentColorHeapIndex;
	uint taaDepthHeapIndex;
	uint taaMotionVectorHeapIndex;
	uint taaHistoryReadHeapIndex;
	uint taaHistoryWriteHeapIndex;
	// 0 on the first frame after start/resize/scene reload -- nothing valid to reproject yet, so
	// taa_resolve.slang just copies the current frame through unblended.
	uint taaHistoryValid;
	// Real dimensions of the taaHistoryRead/WriteHeapIndex images -- NOT the same as `resolution`
	// above. Those are grow-only (EnsureTaaHistoryCapacity, App.cpp), sized to the largest
	// viewport ever requested, while `resolution` is today's (possibly smaller) viewport. A
	// Texture2D.SampleLevel() UV is normalized against the texture's own real size, so computing it
	// from `resolution` instead of this reads the wrong texel -- confirmed live as a resize-dependent
	// repeating/mis-scaled pattern in the resolved image (taa_resolve.slang's history sample silently
	// landing in the stale leftover region beyond the currently-written sub-rect, or wrapping via the
	// sampler's REPEAT addressing).
	uint taaHistoryCapacityWidth;
	uint taaHistoryCapacityHeight;
	// History/current blend factor, live-tunable (Settings window, g_taaHistoryWeight) -- see its own
	// comment for why this isn't a fixed shader constant.
	float taaHistoryWeight;
	// Post-blend unsharp-mask strength, live-tunable (Settings window, g_taa.sharpenAmount) -- see
	// taa_resolve.slang's own use of this for why: the history blend above is inherently a low-pass
	// filter (bilinear-resampled, blended frame over frame), and neither this nor a sampler mip bias
	// (createDescriptorHeap's own comment) can fully counteract that on its own -- the bias only
	// sharpens what the CURRENT frame's raw texture sample looks like, not the temporally-blended
	// result actually displayed.
	float taaSharpenAmount;
};

struct RayPushData
{
	uint64_t sceneInfoAddress;  // SceneInfo*
	uint64_t meshBufferAddress;
	uint64_t materialBufferAddress;
	// uint32[entityIndex] -> meshIndex (i.e. an index into meshBufferAddress above), one entry per
	// scene.m_sceneObjects entity -- see Renderer::createTopLevelAS's comment on
	// asInstance.instanceCustomIndex for why this indirection exists: InstanceID() in a closest-hit
	// shader is the entity's index, not the mesh's, since interaction.slang's pick ray needs the
	// entity index directly and a TLAS instance can only carry one InstanceID(). Built alongside
	// meshBufferAddress in App::dispatchRayTracing.
	uint64_t entityMeshIndexAddress;
	// Per-dispatch output target (float4), set by the caller before each Shader::dispatchRayTracing
	// call -- lets a second ray tracing shader write to a different buffer than g_ray's
	// g_outputBuffer while sharing sceneInfoAddress/meshBufferAddress/materialBufferAddress above.
	// g_ray itself still reads/writes via SceneInfo::outputAddress/prevOutputAddress (unchanged); a
	// new shader can read pushData.outputAddress/prevOutputAddress instead.
	uint64_t outputAddress;
	uint64_t prevOutputAddress;
	// RtxdiConstants* (rtxdi_io.h) -- RTXDI (ReSTIR DI) per-frame parameters/buffer addresses,
	// read by ray.slang's direct-lighting path via RtxdiBridge.slang.
	uint64_t rtxdiConstantsAddress;
	// GIConstants* (gi_io.h) -- ReSTIR GI (temporal-only indirect-bounce resampling) per-frame
	// parameters/buffer address, read by ray.slang's basic path-tracing bounce loop.
	uint64_t giConstantsAddress;
};

// Push data for the two small internal compute shaders that bridge this engine's own G-buffer /
// output-buffer conventions into and out of NRD's expected texture formats (shaders/nrd_prepare.slang,
// shaders/nrd_resolve.slang) -- see NrdIntegration.cpp. Both shaders read/write via a real Texture2D/
// RWTexture2D binding for the NRD-format image, but reach the engine's own accumulated radiance
// buffer (g_outputBuffer) the same buffer-device-address way every ray tracing shader does.
struct NrdBridgePushData
{
	uint64_t outputBufferAddress;  // float4* (the g_outputBuffer slot being denoised this frame)
	uint64_t outputSpecularBufferAddress;  // float4* (g_outputBufferSpecular -- ray.slang's hdrSpecular channel)
	// REBLUR_FrontEnd_GetNormHitDist's hitDistParams (nrd_pack.slang) -- kept in lockstep with
	// whatever's passed to nrd::ReblurSettings::hitDistanceParameters on the C++ side (see
	// NrdIntegration.cpp); only read by nrd_prepare.slang, ignored by nrd_resolve.slang.
	float hitDistParamsA;
	float hitDistParamsB;
	float hitDistParamsC;
	// This frame's real viewport size -- NOT the (possibly larger, grow-only) G-buffer/bridge
	// texture capacity a Texture2D::GetDimensions() call would report. Needed because g_outputBuffer
	// is a flat buffer ray.slang indexes as "pixel.y * <current viewport width> + pixel.x" (see
	// rayGenMain); indexing it with the texture capacity's width instead (as GetDimensions() would
	// give) reads the wrong texels whenever capacity > current viewport, corrupting every row after
	// the first -- confirmed live as a diagonal-tearing artifact before this fix.
	uint32_t viewportWidth;
	uint32_t viewportHeight;
	// Kept in lockstep with nrd::CommonSettings::denoisingRange (NrdIntegration.cpp). Only read by
	// nrd_resolve.slang: NRD does not write OUT_DIFF_RADIANCE_HITDIST for pixels with
	// "viewZ > denoisingRange" (sky/miss -- see ray.slang's viewZ write on a miss), so
	// nrd_resolve.slang must skip those pixels too, instead of overwriting g_outputBuffer's already-
	// correct sky color with g_nrdOutDiffRadianceHitDist's stale/never-written content there.
	float denoisingRange;
	// 0/1 -- see g_nrdShowValidation (App.cpp). When set, nrd_resolve.slang shows NRD's
	// own REBLUR_VALIDATION debug overlay (accumulated-history/disocclusion visualization) instead
	// of the denoised radiance, to diagnose why REBLUR isn't converging the way it should -- only
	// meaningful with VisualizeMode left at Default (otherwise there's no real noisy signal for NRD
	// to have denoised in the first place).
	uint32_t showValidation;
};

struct MaterialData
{
	// 0..1
	float3 albedoColor;
	//
	uint textureId;

	// 0..1
	float3 emissiveColor;
	// Multiplies emissiveColor before it's added to hdr (ray.slang/shader.rast.slang) -- lets an
	// emissive material read as a genuinely bright light source (lamps, screens, ...) rather than
	// being capped to the same 0..1 range as a plain albedo color. 1.0 = emissiveColor as-is. See
	// Material::m_emissiveIntensity (Material.h). Was unused padding (padding1[1]) before this --
	// same size/position, so no layout shift for the fields after it.
	float emissiveIntensity;

	// 0..1
	float roughness;
	// 0..1
	float metallic;
	// 0..1
	float albedoTextureFade;
	// glTF metallic-roughness texture (G=roughness, B=metallic); kInvalidTextureId if this
	// material has none, in which case roughness/metallic above are used as-is (no fade -- glTF
	// has no blend factor for this texture, see Material::m_hndMetallicRoughnessTexture).
	uint metallicRoughnessTextureId;
	//
	uint lightmapTextureId;

	// 0 = disabled (dielectric F0 stays the standard constant 4%, see GetF0's comment). >0 = darker
	// albedo suppresses specular reflectance too, not just diffuse -- a non-physical "specular
	// occlusion" approximation for hand-painted/flat-shaded content (BSP decals, Quake-era textures)
	// where "black" represents ink/shadow rather than a real dielectric material, which otherwise
	// still shows a ~4% specular sheen precisely where the diffuse term is darkest (confirmed live
	// as "black albedo areas look too bright" in ray tracing). See GetF0(baseColor, metallic,
	// specularOcclusionFromAlbedo) (mathlib.slang) for the exact formula and Material::
	// m_specularOcclusionFromAlbedo (Material.h) for the CPU-side default (0 for most materials,
	// 0.25 for BSP-generated ones, see 3DBSPLevel.cpp).
	float specularOcclusionFromAlbedo;

	// Tangent-space normal map (glTF's, a hand-assigned one, or a BSP-generated one -- see
	// GenerateNormalMapFromLuminance, Texture.h). Never kInvalidTextureId at this point: the CPU-side
	// MaterialData fill (App.cpp) already resolved a material with no normal map of its
	// own to g_hndDefaultNormalMap's heap index -- a flat (0,0,1) engine texture -- so every shader
	// consumer can unconditionally sample this with no branch and no special-case fallback logic.
	uint normalMapTextureId;
};

// Push data for the graphics pipeline (vkCmdPushDataEXT).
// With descriptor heap, the pipeline layout is VK_NULL_HANDLE, so traditional push constants
// and push descriptors cannot be used. Push data carries only the minimum needed per draw call:
// the address of the scene buffer (updated once per frame) and the per-draw color.
struct GraphicsPushData
{
	uint64_t sceneInfoAddress;  // SceneInfo*
	uint64_t materialBufferAddress;
	// 16bit materialID, 16bit meshId
	uint materialAndMeshID;
	float padding[3];
#ifdef _CPP
alignas(16)
#endif
	ROW_MAJOR float4x4 modelMatrix;
};

struct PushConstantCompute
{
  uint64_t bufferAddress;  // Buffer device address of the vertex buffer
  float rotationAngle;
  int numVertex;
};

// see C++ "struct Vertex : public shaderio::Vertex"
struct Vertex
{
	float3 position;
	// todo: optimize to 2/4 bytes, octahedral mapping
	float3 normal;
	// todo: optimize, unused, .xy can be use for lightmap uv
	float4 tangent;
	// UV
	float2 texCoord;
	// UV
	float2 lightmapUV;
};

#ifndef _CPP
struct VertexInput
{
	[[vk::location(LVPosition)]] // Vulkan layout(location = 0)
	float3 position : POSITION;
	[[vk::location(LVNormal)]]
	float3 normal : NORMAL;
	[[vk::location(LVTangent)]]
	float4 tangent : TANGENT;
	[[vk::location(LVTexCoord)]]
	float2 uv : TEXCOORD0;
	[[vk::location(LVLightmapUV)]]
	float2 lightmapUV : TEXCOORD1;
};
#endif

#ifdef _CPP
#pragma pack(pop)
#endif

#endif  // HOST_DEVICE_H
