#include "shader_io.h"

#include "s2h/s2h_glsl.hlsl"
#include "s2h/s2h.hlsl"

#include "mathlib.slang"

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
    float4 position : SV_Position; // Clip space position (required)
    float2 uv : TEXCOORD0; // Interpolated texture coordinates
	float3 normal: TEXCOORD1; // Interpolated world space normal, not normalized
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
}

// @return viewToClip aka ClipFromView
float4x4 GetPerspMatrix()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	return ComputePerspectiveMatrix(scene.fieldOfViewDeg * PI / 180.0f, scene.resolution.x / (float) scene.resolution.y, scene.zNear, scene.zFar);
//	return transpose(scene.clipFromView);
}

//------------------------------------------------------------------------------

float3 GetCameraPos()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;
	return scene.wsCameraPos;
}
 
// [shader("vertex")] is Slang's way of marking the vertex shader entry point
[shader("vertex")]
VertexOutput vertexMain(const VertexInput input, uint vertexID : SV_VertexID)
{
	int currentVertexID = vertexID; // GLSL gl_VertexIndex;

    // Access the scene info buffer via buffer device address (updated once per frame)
    // Cast the uint64 address to a typed pointer -- Slang compiles this to SPIR-V untyped pointer access
    SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

    float3 pos = input.position;

    VertexOutput output;
    output.position = float4(pos, 1.0);
//    output.color    = input.color;
//	output.color = normalize(input.normal) * 0.5f + 0.5f;
	output.uv = input.uv;
	output.normal = mul(input.normal, transpose((float3x3)pushData.modelMatrix));

    // Transform vertex position to clip space
	float4 p = output.position;

	p = mul(pushData.modelMatrix, p);

	p = mul(GetViewMatrix(), p);
	p = mul(GetPerspMatrix(), p);

	output.position = p;
	
//	p = mul(GetPerspMatrix(), p);
	
	//	first triangle fills screen, z=far
	if(false)
	{
		if (currentVertexID == 0)
			output.position = float4(-10, -10, 0.9, 1);
		else if (currentVertexID == 1)
			output.position = float4(10, -10, 0.9, 1);
		else if (currentVertexID == 2)
			output.position = float4(0, 20, 0.9, 1);
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
	
	float4 triangleColor = 1.0f;

	float3 Origin = GetCameraPos();

	float4 clipPos = float4(fragPos, input.position.z, 1);

	// World-space reconstruction, mirroring ray.slang's rayGenMain camera-ray unprojection (which is
	// proven correct) instead of the C++-precomputed scene.worldFromClip: compute the inverse
	// view-projection locally from the same GetPerspMatrix()/GetViewMatrix() this shader already
	// uses for the forward transform in vertexMain, so there's only one matrix convention in play.
	float3 surfacePos;
	{
		float4x4 worldToClip = MultiplyMatrices(GetPerspMatrix(), GetViewMatrix());
		float4x4 INV_VIEW_PROJECTION = inverse(worldToClip);

		// Undo vertexMain's NDC Y-flip before applying the inverse, same as rayGenMain does.
		float2 screenPos = clipPos.xy;
		screenPos.y = -screenPos.y;
		float4 worldPosHom = mul(INV_VIEW_PROJECTION, float4(screenPos, clipPos.z, 1));
		surfacePos = worldPosHom.xyz / worldPosHom.w;
	}
	float3 interpolatedNormal = normalize(input.normal);
	
	uint materialID = pushData.materialAndMeshID >> 16;

/*	
	// Texture2D.Handle(uint2(index, 0)) creates a handle from the resource heap at the given index.
	// SamplerState.Handle(uint2(0, 0)) gets the linear sampler at index 0 in the sampler heap.
	// This is the Slang equivalent of GLSL's layout(descriptor_heap) access.
	{
		// this is inherently bindless
		uint texId = resolveTextureId(materialBuffer.textureId, scene.texId);
		Texture2D    tex  = Texture2D.Handle(uint2(texId, 0));
		SamplerState samp = SamplerState.Handle(uint2(0, 0));
		triangleColor = tex.Sample(samp, input.uv);
	}
*/
	float3 V = normalize(Origin - surfacePos);

	float3 hdr = float3(0,0,1); // blue

	// todo: tonemapper
	float3 ldr = hdr;
	
	// m_renderTarget is VK_FORMAT_R8G8B8A8_UNORM (App.cpp) -- no hardware sRGB encode on
	// write, so it has to happen here, same as post.slang's own final output write. Previously only
	// the EVM_Lightmap branch above did this itself and everything else (the main lit path,
	// EVM_AlbedoColor, ...) wrote raw linear values straight to an 8-bit display target -- silently
	// masked before the texture sRGB-decode fix (Texture.cpp) by textures ALSO never being decoded
	// on read, so the two missing conversions happened to cancel out; fixing just the read side
	// exposed this as genuinely too-dark output (confirmed live: EVM_AlbedoColor looked far darker
	// than the source art). Applied unconditionally, matching post.slang's own convention of
	// encoding every visualize mode's output, not just EVM_Shaded.
	output.color = float4(s2h_accurateLinearToSRGB(ldr), 1);
		
	return output;
}
