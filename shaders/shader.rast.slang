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
	output.uv = input.uv;

    // Transform vertex position to clip space
	float4 p = output.position;

	output.position = p;
	
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

    float2 fragPos = (input.position.xy / scene.resolution) * 2.0f - 1.0f;
	
	float3 ldr = (0, 0, 1); // blue
	
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
