#include "shader_io.h"

#include "s2h/s2h_glsl.hlsl"
#include "s2h/s2h.hlsl"
#include "s2h/s2h_scatter.hlsl"
#include "s2h/s2h_3d.hlsl"

#include "mathlib.slang"

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
    float4 position : SV_Position; // Clip space position (required)
    float3 color : TEXCOORD0;      // Interpolated vertex color
    float2 uv : TEXCOORD1;         // Interpolated texture coordinates
};

struct PixelOutput
{
    float4 color : SV_TARGET; // Final color output (RGBA)
};

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Push data (vkCmdPushDataEXT): carries only the scene buffer address and per-draw color.
// With descriptor heap (VK_EXT_descriptor_heap), the pipeline layout is VK_NULL_HANDLE,
// so traditional push constants and push descriptors (UBOs) cannot be used.
[[vk::push_constant]]
ConstantBuffer<GraphicsPushData> pushData;

// Descriptor heap: textures and samplers live in GPU heap buffers (VK_EXT_descriptor_heap).
// We access them by heap index using Texture2D.Handle(uint2(index, 0)) and
// SamplerState.Handle(uint2(index, 0)). This replaces the old [[vk::binding(...)]] Sampler2D array.

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

struct VSOutput // AKA PSInput
{
	float2 uv : TEXCOORD0;
	// object space
	float3 osPos : TEXCOORD1;
	// view space
	float4 vsPos : TEXCOORD2;
	// view space
	float4 vscsPos : TEXCOORD3;
	// clip space, for xbox needs this to be last
	float4 csPos : SV_POSITION;
};

// [shader("vertex")] is Slang's way of marking the vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uint vertexID : SV_VertexID)
{
	int currentVertexID = vertexID; // GLSL gl_VertexIndex;

	VertexOutput output = (VertexOutput)0;
	
	// triangle fills screen, z=far
	{
		if (currentVertexID == 0)
			output.position = float4(-10, -10, 0.0f, 1);
		else if (currentVertexID == 1)
			output.position = float4(10, -10, 0.0f, 1);
		else if (currentVertexID == 2)
			output.position = float4(0, 20, 0.0f, 1);
	}
	
	// DirectX NDC y=1 is top, -1 is bottom
	// OpenGL/Vulkan NDC y=-1 is top, 1 is bottom
	output.position.y = -output.position.y;

	return output;
}


// [shader("pixel")] is Slang's way of marking the fragment shader entry point
[shader("pixel")]
PixelOutput fragmentMain(VertexOutput input, uint primitiveID : SV_PrimitiveID)
{
    PixelOutput output;

    // Access the scene info buffer via buffer device address (same buffer, updated once per frame)
    SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

    float2 fragPos = (input.position.xy / scene.resolution) * 2.0 - 1.0;
	
	ContextGather ui;
    // pxPos is the integer pixel position + 0.5f (pixel centered)
	s2h_init(ui, input.position.xy);

	float4 linearBackground = float4(0.1f, 0.1f, 0.1f, 1);

	{
		float4x4 worldToClip = mul(GetPerspMatrix(), GetViewMatrix());
		float4x4 INV_VIEW_PROJECTION = inverse(worldToClip);
		
		Context3D context;
		float3 worldPos;
	    {
			float2 screenPos = fragPos.xy;
			screenPos.y = -screenPos.y;
//			float4 worldPosHom = mul(float4(screenPos, S2S_NEAR(), 1), INV_VIEW_PROJECTION);
			float4 worldPosHom = mul(INV_VIEW_PROJECTION, float4(screenPos, S2S_NEAR(), 1));
			worldPos = worldPosHom.xyz / worldPosHom.w;
		}
		s2h_init(context, S2S_CAMERA_POS(), normalize(worldPos - S2S_CAMERA_POS()));
		s2h_drawSkybox(context);
		s2h_drawCheckerBoard(context, float3(0, 0, 0));
		linearBackground.rgb = context.dstColor.rgb;
	}

    // Note: ui.dstColor is premultiplied
	float4 linearColor = linearBackground * (1.0f - ui.dstColor.a) + ui.dstColor;
	output.color = float4(s2h_accurateLinearToSRGB(linearColor.rgb), linearColor.a);
	
	output.color.a = 1.0f;

	return output;
}
