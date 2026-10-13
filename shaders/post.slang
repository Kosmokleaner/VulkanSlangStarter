// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
// SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
// SPDX-License-Identifier: Apache-2.0
// Derived from shaders/shader.rast.slang in https://github.com/nvpro-samples/vk_minimal_latest,
// modified by Martin Mittring.

#include "shader_io.h"

#include "s2h/s2h_glsl.hlsl"
#include "s2h/s2h.hlsl"
#include "s2h/s2h_3d.hlsl"

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

// Per-draw data (scene buffer address, model matrix), see GraphicsPushData. g_post is non-bindless,
// so this arrives as classic push constants (vkCmdPushConstants2, see Shader::cmdPushData).
[[vk::push_constant]]
ConstantBuffer<GraphicsPushData> pushData;

// Non-bindless texture: classic descriptor set 0, binding 0 (combined image sampler). g_post is
// created with 1 texture slot (see Shader::m_textureCount), App::recordGraphicsCommands binds it.
[[vk::binding(0, 0)]]
Sampler2D g_texture;

//------------------------------------------------------------------------------

struct VSOutput // AKA PSInput
{
	float2 uv : TEXCOORD0;
	// clip space, for xbox needs this to be last
	float4 csPos : SV_POSITION;
};



void Test(inout ContextGather ui)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	s2h_setScale(ui, 2);

	s2h_printTxt(ui, _S, _h, _a, _d, _e, _r);
	s2h_printTxt(ui, _2);
	s2h_printTxt(ui, _H, _u, _m, _a, _n);
	s2h_printLF(ui);
	s2h_printLF(ui);

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
	float2 xy = uv * 2.0f + 1;

	float3 osPos = float3(xy, 0);
	
	output.csPos = mul(worldToClip, float4(osPos, 1));

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
	s2h_setCursor(ui, float2(200, 100));
	s2h_setScale(ui, 1);

	if (scene.infoHUD != 0)
		Test(ui);
	
	float4 linearBackground = float4(0.1f, 0.1f, 0.1f, 0);

	float depth = 0;

	float2 fragPos = (input.position.xy / scene.resolution) * 2.0f - 1.0f;

	// UV gradient
	linearBackground = float4(input.position.xy / scene.resolution, 0, 1);

	// Texture on top of the gradient, centered. Power-of-two scale (native size if it fits in 3/4 of
	// the viewport) so each pixel lands on a texel center of a single mip -- crisp, no filter blur.
	{
		uint texWidth, texHeight;
		g_texture.GetDimensions(texWidth, texHeight);
		float2 texSize = float2(texWidth, texHeight);
		float2 fit = scene.resolution * 0.75f / texSize;
		float scale = min(1.0f, exp2(floor(log2(min(fit.x, fit.y)))));
		float2 rectSize = texSize * scale;
		float2 rectMin = floor((scene.resolution - rectSize) * 0.5f);
		float2 texUV = (input.position.xy - rectMin) / rectSize;
		// sampled outside the branch so the implicit mip derivatives stay valid at the rect edges
		float4 texColor = g_texture.Sample(texUV);	// sRGB format: already decoded to linear
		if (all(texUV >= 0.0f) && all(texUV < 1.0f))
			linearBackground.rgb = lerp(linearBackground.rgb, texColor.rgb, texColor.a);
	}

	// Note: ui.dstColor is premultiplied
	float4 linearColor = linearBackground * (1.0f - ui.dstColor.a) + ui.dstColor;

	output.color = float4(s2h_accurateLinearToSRGB(linearColor.rgb), linearColor.a);

	if (output.color.a < 1.0f)
		clip(-1);


	output.color.a = 1.0f;
	
	return output;
}
