#include "shader_io.h"

#include "s2h/s2h_glsl.hlsl"
#include "s2h/s2h.hlsl"
#include "s2h/s2h_3d.hlsl"

#include "mathlib.slang"

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
    float4 position : SV_Position; // Clip space position (required)
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



void Test(inout ContextGather ui)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	s2h_setScale(ui, 2);

	// s2h_printTxt() takes at most 6 characters per call
	s2h_printTxt(ui, _R, _E, _S, _COLON, _SPACE);
	s2h_printInt(ui, (int)scene.resolution.x);
	s2h_printTxt(ui, _SPACE);
	s2h_printInt(ui, (int)scene.resolution.y);
	s2h_printLF(ui);
}


VSOutput computeVS(uint vertexId, float4x4 worldToClip, float4x4 worldToView, float4x4 viewToClip)
{
	VSOutput output = (VSOutput) 0;
	uint id = vertexId % 6;

	float2 uv = float2(0, 0);
	if (id == 1)
		uv = float2(1, 0);
	if (id == 2 || id == 3)
		uv = float2(1, 1);
	if (id == 4)
		uv = float2(0, 1);
	

	// x:-1 / 1, y:-1 / 1
//	float2 xy = uv * 2.0f - 1.0f;
	float2 xy = uv * 2.0f + 1;

	output.osPos = float3(xy, 0);
	
	/*
	if (vertexId == 0)
		output.osPos = float3(-1.1, -1.1, 0);
	if (vertexId == 1)
		output.osPos = float3(1.1, -1.1, 0);
	if (vertexId == 2)
		output.osPos = float3(0, 1.1, 0);
	*/
	
	{
		output.vsPos = mul(worldToView, float4(output.osPos, 1));
		output.vscsPos = mul(viewToClip, mul(worldToView, float4(output.osPos, 1)));
		output.csPos = mul(worldToClip, float4(output.osPos, 1));
	}

	output.uv = float2(uv.x, 1 - uv.y);

	return output;
}


// [shader("vertex")] is Slang's way of marking the vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(VertexInput input, uint vertexID : SV_VertexID)
{
	int currentVertexID = vertexID; // GLSL gl_VertexIndex;
	
	VertexOutput output = (VertexOutput)0;

	//	one triangle fills screen, z=near
	{
		if (currentVertexID == 0)
			output.position = float4(-10, -10, 1.0f, 1);
		else if (currentVertexID == 1)
			output.position = float4(10, -10, 1.0f, 1);
		else if (currentVertexID == 2)
			output.position = float4(0, 20, 1.0f, 1);
	}
	
	// DirectX NDC y=1 is top, -1 is bottom
	// OpenGL/Vulkan NDC y=-1 is top, 1 is bottom
	output.position.y = -output.position.y;

	return output;
}

void s2h_printFloat3(inout ContextGather ui, float3 m)
{
	for (int y = 0; y < 3; ++y)
	{
		s2h_printFloat(ui, m[y]);
		s2h_printTxt(ui, _SPACE);
	}
}

void s2h_printFloat4(inout ContextGather ui, float4 m)
{
	for (int y = 0; y < 4; ++y)
	{
		s2h_printFloat(ui, m[y]);
		s2h_printTxt(ui, _SPACE);
	}
}

void s2h_printFloat4x4(inout ContextGather ui, float4x4 m)
{
	for (int y = 0; y < 4; ++y)
	{
		s2h_printFloat(ui, m[y][0]);
		s2h_printTxt(ui, _SPACE);
		s2h_printFloat(ui, m[y][1]);
		s2h_printTxt(ui, _SPACE);
		s2h_printFloat(ui, m[y][2]);
		s2h_printTxt(ui, _SPACE);
		s2h_printFloat(ui, m[y][3]);
		s2h_printLF(ui);
	}
}

// [shader("pixel")] is Slang's way of marking the fragment shader entry point
[shader("pixel")]
PixelOutput fragmentMain(VertexOutput input, uint primitiveID : SV_PrimitiveID)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

    PixelOutput output;
	
	ContextGather ui;
    // pxPos is the integer pixel position + 0.5f (pixel centered)
	s2h_init(ui, input.position.xy);
	s2h_setCursor(ui, float2(200, 300));
	s2h_setScale(ui, 1);

	if (scene.infoHUD != 0)
		Test(ui);
	
	float4 linearBackground = float4(0.1f, 0.1f, 0.1f, 0);

	float depth = 0;

	float2 fragPos = (input.position.xy / scene.resolution) * 2.0f - 1.0f;

	// UV gradient
	linearBackground = float4(input.position.xy / scene.resolution, 0, 1);

	// Note: ui.dstColor is premultiplied
	float4 linearColor = linearBackground * (1.0f - ui.dstColor.a) + ui.dstColor;

	output.color = float4(s2h_accurateLinearToSRGB(linearColor.rgb), linearColor.a);

	if (output.color.a < 1.0f)
		clip(-1);


	output.color.a = 1.0f;
	
	return output;
}
