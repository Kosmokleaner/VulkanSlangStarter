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

// Scene information, stored in a GPU buffer and updated once per frame via vkCmdUpdateBuffer.
// The shader accesses it through a buffer device address (buffer reference), not a descriptor set.
struct SceneInfo
{
	float2 resolution;         // Viewport resolution, todo: uint and 1/xy

	ROW_MAJOR float4x4 viewFromWorld; // aka view, worldToView
	ROW_MAJOR float4x4 clipFromView; // aka projection, viewToClip
	ROW_MAJOR float4x4 worldFromView; // aka invView, ViewToWorld
	ROW_MAJOR float4x4 viewFromClip; // aka invProjection, clipToView
	ROW_MAJOR float4x4 worldFromClip;

	float3 wsCameraPos;
	// 0:hide/1:draw HUD
	uint infoHUD;

	// in sRGB
	float3 clearColor;
	float zNear;

	float zFar;
	float fieldOfViewDeg;
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
