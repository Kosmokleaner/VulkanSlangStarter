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
	float2 lightmapUV : TEXCOORD2; // Interpolated world space normal, not normalized
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

	return ComputePerspectiveMatrix(scene.fieldOfViewDeg * PI / 180.0f, scene.resolution.x / (float) scene.resolution.y, scene.zNear, scene.zFar);
//	return transpose(ComputePerspectiveMatrix(45.0f * PI / 180.0f, 16.0f / 9.0f, 0.1, 1000.0f));
//	return transpose(scene.clipFromView);

	// test
//	return float4x4( // Note: S2H prints out in this form
//		2.41f, 0.0,   3.0, 0.0,
//		0.0,   2.41f, 0.0, 0.0,
//		0.0,   0.0,   0.0, 1.0,
//		0.0,   0.0,   0.1, 0.0);
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
	output.lightmapUV = input.lightmapUV;
//	output.normal = input.normal;// todo: transform
	output.normal = mul(input.normal, transpose((float3x3)pushData.modelMatrix));

    // Transform vertex position to clip space
//	output.position = GetPerspMatrix() * GetViewMatrix() * pushData.modelMatrix * output.position;
	float4 p = output.position;

	p = mul(pushData.modelMatrix, p);

	
//	p = mul(p, pushData.modelMatrix);
	// mul(m, v): Treats v as a column vector and multiplies matrix m by it.
//	p = mul(p, pushData.modelMatrix);
//	p = mul(p, transpose(GetViewMatrix()));
	p = mul(GetViewMatrix(), p);
//	p = p * GetViewMatrix(); // no
//	p = GetViewMatrix() * p; // no
//	p = mul(GetViewMatrix(), p);
//	p = mul(GetPerspMatrix(), p);
//	p = MatrixVectorMultiply(transpose(GetPerspMatrix()), p);
	p = mul(GetPerspMatrix(), p);

	// redo differently
//	p = MatrixVectorMultiply(MultiplyMatrices(GetPerspMatrix(), GetViewMatrix()), output.position);

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

	// AA jitter (see scene.temporalJitter's own comment, App.cpp's updateSceneBuffer):
	// ray.slang's rayGenMain applies the identical jitter by sampling the primary ray at pixel
	// center + (temporalJitter-0.5) instead of the pixel center -- i.e. it samples the scene shifted
	// by that sub-pixel delta. To make the rasterizer (which always samples at fixed pixel centers)
	// produce the same result, the *geometry* has to move by the opposite delta instead: shifting
	// what's rendered left is equivalent to sampling further right. Scaled by w since this is still
	// homogeneous clip space, pre perspective-divide. Both temporalJitter and NDC x/y here already
	// agree on "Y increases downward" (pixel-space and this shader's post-flip NDC both do), so no
	// extra sign flip is needed for that -- only the geometry-vs-sample-point negation above.
	output.position.xy -= (scene.temporalJitter - 0.5f) * (2.0f / scene.resolution) * output.position.w;

	// hack
//	output.position.zw = float2(0.95f, 10);	// after divide z needs to be in 0..1 range
//	output.position.xy *= 0.125;
//	output.position.xy *= output.position.w;

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
	
	MaterialData * materialBuffer = (MaterialData *)pushData.materialBufferAddress + materialID;
	float3 materialColor = materialBuffer.albedoColor;

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
	// See Texture.cpp's m_isSRGB comment -- albedo textures upload as VK_FORMAT_R8G8B8A8_SRGB, so
	// the GPU already decoded gamma->linear on sample, same as the ray tracing path.
	materialColor *= lerp(1.0f, triangleColor.rgb, materialBuffer.albedoTextureFade);

	float metallic = materialBuffer.metallic;
	float roughness = materialBuffer.roughness;

	// glTF metallic-roughness texture: G=roughness, B=metallic, multiplied with the scalar
	// factors above (no fade -- glTF has no blend factor for this texture, unlike albedo).
	if (materialBuffer.metallicRoughnessTextureId != kInvalidTextureId)
	{
		Texture2D    mrTex  = Texture2D.Handle(uint2(materialBuffer.metallicRoughnessTextureId, 0));
		SamplerState samp   = SamplerState.Handle(uint2(0, 0));
		float4 mr = mrTex.Sample(samp, input.uv);
		roughness *= mr.g;
		metallic *= mr.b;
	}

	// White (no darkening/tinting), NOT resolveTextureId's scene.texId debug fallback -- that
	// fallback exists so a texture-less material can preview the "Image 1"/"Image 2" debug images on
	// its ALBEDO (see its call site above), which makes sense there but not here: a lightmap is a
	// multiplicative light/shadow term, not a color preview, and hdr below starts as
	// "lightmap * materialColor" -- feeding it a debug image (now g_taa's own ReplaceMe checkerboard,
	// see Renderer::init()'s comment, permanently registered at heap index 0 = scene.texId's default
	// value) instead of a neutral 1.0 multiplied the material's own correct albedo by that
	// checkerboard directly, confirmed live as every non-BSP material (nothing else has a real
	// lightmap -- only a loaded BSP level's own per-surface lightmap texture ever sets
	// materialBuffer.lightmapTextureId) rendering discolored/checkerboard-tinted. ray.slang's own
	// equivalent lightmap line is commented out entirely for the same reason (see its own comment).
	float3 lightmap = float3(1.0f, 1.0f, 1.0f);
	if (materialBuffer.lightmapTextureId != kInvalidTextureId)
	{
		Texture2D    tex = Texture2D.Handle(uint2(materialBuffer.lightmapTextureId, 0));
		SamplerState samp = SamplerState.Handle(uint2(0, 0));
		lightmap = tex.Sample(samp, input.lightmapUV).rgb;
		// Decodes the lightmap texture (uploaded as UNORM, no hardware sRGB decode -- see the
		// lightmap Texture's own m_isSRGB=false at load time, 3DBSPLevel.cpp) as if it were
		// sRGB-encoded data. Re-added after a live report of insufficient contrast in raster mode:
		// exposure (g_ExposureLog) is a uniform linear scale and can't reshape contrast at all --
		// only a nonlinear curve like this can pull shadows down relative to highlights. An earlier
		// note recorded this as tested-and-not-helping, but that was against a DIFFERENT complaint
		// (too much contrast) made before shader.rast.slang's tonemapper-always-on bug was fixed;
		// that fix changes the picture enough that this is worth re-testing on its own merits.
		lightmap = s2h_accurateSRGBToLinear(lightmap);
		// See shader_io.h's SceneInfo::lightmapIntensity comment.
		lightmap *= scene.lightmapIntensity;
	}


	// g_lights (App.cpp): [0] is always the single global light (g_lightPos/
	// g_lightColor/g_lightIntensity), followed by every point light the loaded BSP level's "light"
	// entities parsed into (C3DBSPLevel::m_Lights). No shadow rays here (the rasterizer has no
	// acceleration structure to trace against, unlike ray.slang) -- every light is unoccluded.
	Light* lights = (Light*)scene.lightsAddress;
	float3 V = normalize(Origin - surfacePos);

	float3 hdr = lightmap * materialColor;
	// Baked-lightmap materials (BSP surfaces) already have every static light's contribution
	// pre-integrated into `lightmap` -- lights[] contains those SAME lights (see this block's own
	// comment above), so also running the per-light loop against them here would double-count.
	// Only materials with no lightmap (materialBuffer.lightmapTextureId == kInvalidTextureId, so
	// `lightmap` is the neutral 1.0 default set above) need real per-light shading instead.
	if (materialBuffer.lightmapTextureId == kInvalidTextureId)
	{
		hdr = 0;
		[loop]
		for (uint li = 0; li < scene.lightCount; ++li)
		{
			float3 L = normalize(lights[li].position - surfacePos);
			float3 lightColor = lights[li].color * EvalLightFalloff(length(lights[li].position - surfacePos), lights[li].radius);
			hdr += lightColor * EvalGGXBRDF(interpolatedNormal, V, L, materialColor, metallic, max(roughness, kMinPunctualLightRoughness));
		}
	}
	hdr += materialBuffer.emissiveColor * materialBuffer.emissiveIntensity;



	// See SceneInfo::enableTonemapper's comment -- this path never respected the checkbox at all
	// before (post.slang's ray tracing path did), so it was always applying filmicToneMapping's
	// contrast-boosting/shoulder-rolloff S-curve unconditionally. Confirmed live as a too-contrasty,
	// crushed-blacks look that persisted regardless of the lightmap sRGB/intensity investigation --
	// consistent with a tonemap curve rather than a lightmap decode issue, since a curve distorts
	// contrast/shape while exposure (already gated to EVM_Shaded below) only scales everything
	// uniformly and can't fix that on its own.
	float3 ldr = scene.enableTonemapper != 0 ? filmicToneMapping(hdr) : hdr;

	if (scene.visualizeMode == VisualizeMode::EEnum::EVM_NormalWS)
	{
		ldr = interpolatedNormal * 0.5f + 0.5f;
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_AlbedoColor)
	{
		ldr = materialColor;
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_PositionWS)
	{
		ldr = frac(surfacePos);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_DepthNorm)
	{
		ldr = pow(input.position.z, float3(1 / 6.0f, 1 / 3.0f, 1 / 1.0f)); // todo
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_DepthWS)
	{
//		ldr = frac(payload.minT);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_UV)
	{
		ldr = float3(frac(input.uv*10), 0);	// todo *10
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_LightmapUV)
	{
		ldr = float3(frac(input.lightmapUV * 10), 0);	// todo *10
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_MaterialID)
	{
		ldr = s2h_indexToColor(pushData.materialAndMeshID >> 16);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_MeshID)
	{
		ldr = s2h_indexToColor(pushData.materialAndMeshID & 0xffff);
	}
	else if (scene.visualizeMode == VisualizeMode::EEnum::EVM_Lightmap)
	{
		ldr = lightmap;
	}

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
