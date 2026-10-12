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
	// 0:hide/1:draw HUD
	uint infoHUD;

	// in sRGB
	float3 clearColor;
	float zNear;


	uint temporalAccumulator;
	// 1/(temporalAccumulator+1)
	float invTemporalAccumulator;
	// 0.0f / 1.0f
	float temporalMixIn;
	// for frame Jitter
	uint frameIndex;

	// Basic homogeneous participating media (ray.slang: EvalFogTransmittance/EvalInscattering). A
	// per-channel extinction/scattering coefficient is fogDensity * fogColor (single-scattering
	// albedo of 1, i.e. no separate absorption); 0 density disables fog entirely. fogColor also
	// tints extinction (e.g. a blue fogColor extinguishes non-blue wavelengths faster, so blue
	// survives at range).
	float3 fogColor;
	float fogDensity;

	float zFar;
	float fieldOfViewDeg;
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


// Push data for the graphics pipeline (vkCmdPushDataEXT).
// With descriptor heap, the pipeline layout is VK_NULL_HANDLE, so traditional push constants
// and push descriptors cannot be used. Push data carries only the minimum needed per draw call:
// the address of the scene buffer (updated once per frame) and the per-draw color.
struct GraphicsPushData
{
	uint64_t sceneInfoAddress;  // SceneInfo*
	// 16bit materialID, 16bit meshId
	uint materialAndMeshID;
	float padding[2];
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
