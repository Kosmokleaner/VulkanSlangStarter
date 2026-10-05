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



void Test(inout ContextGather ui)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	uint2 resolution = (uint2)scene.resolution;

	// 16MB extra data to maintain state over multiple frames
	float4* extraDataIn = (float4*)scene.outputAddress + resolution.x * resolution.y;
	float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

	s2h_setScale(ui, 2);

	s2h_printTxt(ui, _I, _N, _COLON);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataIn[0]);
	s2h_printTxt(ui, _COLON);
	s2h_printInt(ui, asuint(extraDataIn[0].x));
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataIn[1]);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataIn[2]);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataIn[3]);
	s2h_printLF(ui);
	s2h_printLF(ui);

	s2h_printTxt(ui, _O, _U, _T, _COLON);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataOut[0]);
	s2h_printTxt(ui, _COLON);
	s2h_printInt(ui, asuint(extraDataOut[0].x));
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataOut[1]);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataOut[2]);
	s2h_printLF(ui);
	s2h_printFloat4(ui, extraDataOut[3]);

	s2h_printLF(ui);
	s2h_printLF(ui);
	s2h_printTxt(ui, _F, _O, _V, _COLON);
	s2h_printFloat4(ui, scene.fieldOfViewDeg);

	s2h_printLF(ui);
	s2h_printLF(ui);
	{
		// interaction.slang's rayGenMain wrote this earlier in the same frame -- see
		// kInteractionResultOffset's comment (shader_io.h) for why this is scene.outputAddress
		// (not prevOutputAddress/extraDataIn/extraDataOut above).
		float4* interactionResult = (float4*)scene.outputAddress + resolution.x * resolution.y + kInteractionResultOffset;
		int objectIndex = asint(interactionResult[0].w);

		// s2h_printTxt() takes at most 6 characters per call
		s2h_printTxt(ui, _O, _b, _j, _e, _c, _t);
		s2h_printTxt(ui, _COLON, _SPACE);
		s2h_printInt(ui, objectIndex);
	}


/*
	float4x4 worldToView = GetViewMatrix();
	float4x4 viewToClip = GetPerspMatrix();
	
//	float4x4 worldToClip = mul(viewToClip, worldToView);
	float4x4 worldToClip = MultiplyMatrices(viewToClip, worldToView);
	
		
	s2h_setCursor(ui, float2(10, 40 + 0 * 40));
	s2h_printTxt(ui, _W, _2, _V, _COLON);
	s2h_setCursor(ui, float2(60, 40 + 0 * 40));
	s2h_printFloat4x4(ui, worldToView);
	s2h_printLF(ui);

	s2h_setCursor(ui, float2(10, 40 + 1 * 40));
	s2h_printTxt(ui, _V, _2, _C, _COLON);
	s2h_setCursor(ui, float2(60, 40 + 1 * 40));
	s2h_printFloat4x4(ui, viewToClip);
		
	s2h_setCursor(ui, float2(10, 40 + 2 * 40));
	s2h_printTxt(ui, _W, _2, _C, _COLON);
	s2h_setCursor(ui, float2(60, 40 + 2 * 40));
	s2h_printFloat4x4(ui, worldToClip);
	
	s2h_printLF(ui);
	s2h_printTxt(ui, _C, _a, _m, _P, _o, _s);
	s2h_printTxt(ui, _COLON, _SPACE);
	s2h_printFloat3(ui, S2S_CAMERA_POS());

	s2h_printLF(ui);
	s2h_printTxt(ui, _z, _R, _a, _n, _g, _e);
	s2h_printTxt(ui, _COLON, _SPACE);
	s2h_printFloat(ui, scene.zNear);
	s2h_printTxt(ui, _SPACE);
	s2h_printFloat(ui, scene.zFar);

	for (int i = 0; i < 4; ++i)
	{
		s2h_setCursor(ui, float2(10, 200 + i * 40));
		s2h_printTxt(ui, '[');
		s2h_printInt(ui, i);
		s2h_printTxt(ui, ']');
		s2h_printLF(ui);

		VSOutput v = computeVS(i, worldToClip, worldToView, viewToClip);
		
		s2h_setCursor(ui, float2(60, 200 + i * 40));
		s2h_printTxt(ui, _O, _S, _COLON, _SPACE);
		s2h_printFloat3(ui, v.osPos);
		s2h_printLF(ui);

		s2h_printTxt(ui, _V, _S, _COLON, _SPACE);
		s2h_printFloat4(ui, v.vsPos);
		s2h_printLF(ui);

		s2h_printTxt(ui, _V, _S, _C, _S, _COLON, _SPACE);
		s2h_printFloat4(ui, v.vscsPos); // hom
		s2h_printTxt(ui, _EQUAL, '>', _SPACE);
		s2h_printFloat3(ui, v.vscsPos.xyz / v.vscsPos.w); // after pers divide
		s2h_printLF(ui);

		s2h_printTxt(ui, _C, _S, _COLON, _SPACE);
		s2h_printFloat4(ui, v.csPos); // hom
		s2h_printTxt(ui, _EQUAL, '>', _SPACE);
		s2h_printFloat3(ui, v.csPos.xyz / v.csPos.w); // after pers divide
		s2h_printLF(ui);
	}
	
	s2h_printLF(ui);
	s2h_printLF(ui);

	
	for (int materialID = 0; materialID < 2; ++materialID)
	{
		MaterialData * materialBuffer = (MaterialData *)pushData.materialBufferAddress + materialID;
		s2h_printFloat3(ui, materialBuffer.albedoColor); // after pers divide
		s2h_printLF(ui);
	}
*/
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
	
	if (false)
	{
		// mul
		output.vsPos = mul(worldToView, float4(output.osPos, 1));
		output.vscsPos = mul(viewToClip, mul(worldToView, float4(output.osPos, 1)));
		output.csPos = mul(worldToClip, float4(output.osPos, 1));
	}
	else
	{
		// MatrixVectorMultiply
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

// @return 0/1 
int getRayTracingSelection(VertexOutput input, int2 offset)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;
	int2 pixel = (int2)input.position.xy + offset;
	uint2 resolution = (uint2)scene.resolution;
	float4* rayTraceData = (float4*)scene.outputAddress;

	if((uint)pixel.x < resolution.x
	&& (uint)pixel.y < resolution.y
	&& rayTraceData[pixel.y * resolution.x + pixel.x].a < 0.0f)
		return 1;

	return 0;
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
	// 0..1
	float selected = 0.0f;
	
	// ray tracing
	if(scene.rayTracing != 0)
	{
		uint2 pixel = (uint2)input.position.xy;
		float4* rayTraceData = (float4 *)scene.outputAddress;
		float4* rayTraceSpecularData = (float4 *)scene.outputSpecularAddress;
		uint index = pixel.y * (uint)scene.resolution.x + pixel.x;
		// See ray.slang's hdrSpecular comment -- the direct/NEE specular lobe was split out of the
		// diffuse channel above into its own NRD-denoised channel, recombined here for display; both
		// use the same invTemporalAccumulator scaling (a no-op multiply-by-1 outside Reference
		// Accumulation mode, since NRD keeps temporalAccumulator at 1 -- see g_outputBufferSpecular's
		// own comment for why the specular buffer is ping-ponged/accumulated the same way).
		linearBackground = float4((rayTraceData[index].rgb + rayTraceSpecularData[index].rgb) * scene.invTemporalAccumulator, 1);

		if (scene.visualizeMode == VisualizeMode::EEnum::EVM_Shaded && scene.enableTonemapper != 0)
		{
			// tonemap after integration
			linearBackground.rgb = filmicToneMapping(linearBackground.rgb);
		}
		depth = abs(rayTraceData[index].a);

		int inside = getRayTracingSelection(input, int2(0, 0));
		int outside = getRayTracingSelection(input, int2(1, 0)) 
					+ getRayTracingSelection(input, int2(-1, 0))
					+ getRayTracingSelection(input, int2(0, 1))
					+ getRayTracingSelection(input, int2(0, -1));

		selected = saturate(outside) - inside * 0.75f;
	}

	float2 fragPos = (input.position.xy / scene.resolution) * 2.0f - 1.0f;
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

	// Note: ui.dstColor is premultiplied
	float4 linearColor = linearBackground * (1.0f - ui.dstColor.a) + ui.dstColor;

	output.color = float4(s2h_accurateLinearToSRGB(linearColor.rgb), linearColor.a);

	if(scene.drawLights != 0)
	{
		Context3D context;

		s2h_init(context, ray.Origin, ray.Direction);
		context.depth = depth;
		// tube light
//		s2h_drawLineWS(context, scene.wsLightPos - float3(0,0, scene.lightHalfLength), scene.wsLightPos + float3(0, 0, scene.lightHalfLength), float4(scene.lightColor, 1), 0.15f);

		Light* lights = (Light*)scene.lightsAddress;

		[loop]
		for (uint li = 0; li < scene.lightCount; ++li)
		{
			float3 lightPos = lights[li].position;
			float3 lightColor = lights[li].color;
//			float lightRadius = lights[li].radius;

			s2h_drawSphereWS(context, lightPos, float4(lightColor, 1), 0.02f);
		}

		// laser physical
//		s2h_drawSphereWS(context, float3(0, 1.5f, -2), float4(1, 0, 0, 1), 0.02f);
//		s2h_drawLineWS(context, float3(0, 1.5f, -2), float3(0, 1.5f, -2) - float3(0,0,0.5f), float4(0.1f, 0.1f, 0.1f, 1.0f), 0.1f);

/*
		{
			uint2 resolution = (uint2)scene.resolution;
			// 16MB extra data to maintain state over multiple frames
			float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

			uint count = asuint(extraDataOut[0].x);

			for(uint i = 1; i < count; ++i)
			{
				float3 p0 = extraDataOut[1 + i - 1].xyz;
				float3 p1 = extraDataOut[1 + i].xyz;

				s2h_drawLineWS(context, p0, p1, float4(1, 0, 0, 1), 0.005f);
			}
		}
*/

		output.color = lerp(output.color, context.dstColor, context.dstColor.a);
	}

	output.color = lerp(output.color, float4(0,1,0,1), selected);

	if (output.color.a < 1.0f)
		clip(-1);

	output.color.a = 1.0f;
	
	return output;
}
