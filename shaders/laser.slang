
// 0/1
#define SKY_IS_EMISSIVE 0


#include "shader_io.h"

#include "mathlib.slang"

//------------------------------------------------------------------------------
// Resource Bindings
//------------------------------------------------------------------------------

// Push data (vkCmdPushDataEXT): carries only the scene buffer address and per-draw color.
// With descriptor heap (VK_EXT_descriptor_heap), the pipeline layout is VK_NULL_HANDLE,
// so traditional push constants and push descriptors (UBOs) cannot be used.
[[vk::push_constant]]
ConstantBuffer<RayPushData> pushData;

// Descriptor heap: textures and samplers live in GPU heap buffers (VK_EXT_descriptor_heap).
// We access them by heap index using Texture2D.Handle(uint2(index, 0)) and
// SamplerState.Handle(uint2(index, 0)). This replaces the old [[vk::binding(...)]] Sampler2D array.

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure g_tlas;

// Traditional (non-heap) texture binding for the closest-hit shader's texture lookup.
// VK_EXT_descriptor_heap can't be used on this pipeline: a descriptor-heap pipeline requires
// layout == VK_NULL_HANDLE (VUID-VkRayTracingPipelineCreateInfoKHR-flags-11311), which conflicts
// with g_tlas's traditional descriptor set binding above. Indexed either with a per-material
// glTF texture id (MaterialData::textureId, same index the graphics pass's descriptor heap uses)
// or, if the material has none, with scene.texId (the debug "Image 1"/"Image 2" toggle) -- see
// resolveTextureId() in mathlib.slang. Capped at kMaxRayTracingTextures entries; ids beyond the
// cap are clamped there.
[[vk::binding(3, 0)]]
SamplerState g_sampler;
[[vk::binding(4, 0)]]
Texture2D g_textures[kMaxRayTracingTextures];

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

	return transpose(scene.clipFromView);
}

float3 GetCameraPos()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;
	return scene.wsCameraPos;
}


#define S2S_CAMERA_POS() GetCameraPos()
#define S2S_NEAR() 0.1f

//------------------------------------------------------------------------------

struct Payload
{
    // in world space, not normalized during intersection but afterwards
	float3 interpolatedNormal;
    // -1 if not set
	int instanceIndex;
    // used like z buffer, the smallest valus is used
	float minT;
};

[shader("raygeneration")]
void rayGenMain()
{
	uint2 pixel = DispatchRaysIndex().xy;

	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	// 16MB extra data to maintain state over multiple frames. This region starts right after the
	// full-viewport pixel buffer (scene.resolution.x * scene.resolution.y float4s -- see
	// App::EnsureOutputBufferCapacity), NOT after DispatchRaysDimensions(): g_laser is
	// dispatched at a small fixed size ({8,8}, see App.cpp), so using its own dispatch
	// dimensions here pointed at the wrong offset (into the middle of the main pixel buffer
	// instead of past it) -- must match post.slang's Test(), which reads this same region via
	// scene.resolution.
	uint2 resolution = (uint2)scene.resolution;
	float4* extraDataIn = (float4*)scene.outputAddress + resolution.x * resolution.y;
	float4* extraDataOut = (float4*)scene.prevOutputAddress + resolution.x * resolution.y;

	// todo
	if(all(pixel != int2(0, 0)))
		return;


	RayDesc ray;



	ray.Origin = scene.laserPos;
	ray.Direction = float3(sin(scene.laserAngle), 0, cos(scene.laserAngle));

/*
	ray.Origin = S2S_CAMERA_POS();
	{
		float2 fragPos = 0.0f;	// screen center
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
*/

	ray.TMin = 0.001; // Minimum distance to avoid self-intersection
	ray.TMax = INFINITE; // Maximum distance (infinite for primary rays)

	uint totalHits = 10;
	uint hitCount = 0;

	extraDataOut[1 + hitCount++] = float4(ray.Origin, 0);

	[loop]
	for (uint bounce = 0; bounce < totalHits; ++bounce)
	{
		Payload payload = (Payload)0;
		payload.instanceIndex = -1;

		TraceRay(
			g_tlas,
			RAY_FLAG_NONE,
			0xFF,
			0, // hit group offset
			0, // multiplier
			0, // miss shader
			ray,
			payload);

		payload.interpolatedNormal = normalize(payload.interpolatedNormal);

		if (payload.instanceIndex != -1)
		{
			float3 surfacePos = ray.Origin + ray.Direction * payload.minT;

			extraDataOut[1 + hitCount++] = float4(surfacePos, 0);

			ray.TMin = 0.001f;
			ray.Origin = surfacePos;
			ray.Direction = reflect(ray.Direction, payload.interpolatedNormal);
		}
		else
		{
			float3 farPos = ray.Origin + ray.Direction * 100.0f;

			// sky hit
			extraDataOut[1 + hitCount++] = float4(farPos, 0);
			break;
		}
	}

	extraDataOut[0] = float4(asfloat(hitCount), 0, 0, 0);
/*

	// Only normalize on an actual hit: on a miss, rayMissMain zeroes the whole payload (including
	// interpolatedNormal), and normalize((0,0,0)) is a 0/0 division -- NaN, which then poisons every
	// downstream use of this value for that pixel.
	if (payload.instanceIndex != -1)
		payload.interpolatedNormal = normalize(payload.interpolatedNormal);
	
	// scene.frameNoiseSeed is 0 (same pattern every frame) or frame-varying (a different pattern
	// each frame), depending on the "Frame Noise Randomization" checkbox -- see shader_io.h.
	uint rndState = initRand(dot(DispatchRaysIndex(), uint3(82927, 21313, 1)), 0x12345678 + scene.frameNoiseSeed);
		
	float3 surfacePos = ray.Origin + ray.Direction * payload.minT;

    // start slightly above the surface to avoid hit with own surface
	surfacePos += payload.interpolatedNormal * 0.001f;

	{
		// Basic path tracing, inline in this loop (no shader-side recursion, no denoiser, one path
		// per pixel): walk the primary hit plus scene.pathTracingBounces further bounces, each one
		// TraceRay for the bounce itself and exactly one shadow ray for direct lighting at that
		// vertex (next-event estimation -- with a single point light and no area light to sample,
		// this is what actually makes the light visible at all; pure BSDF sampling would need to
		// hit a zero-size point by chance). Indirect bounces use a cosine-weighted hemisphere
		// sample against the surface normal, which is the standard basic/diffuse-only choice: for
		// a Lambertian BRDF under cosine-weighted sampling the usual f*cosTheta/pdf throughput
		// term collapses to just the surface albedo (see getCosHemisphereSample, mathlib.slang).

				if (bouncePayload.instanceIndex == -1)	// sky hit
				{
					hdr += throughput * bouncePayload.emissiveColor;
					break;
				}
				bouncePayload.interpolatedNormal = normalize(bouncePayload.interpolatedNormal);
			}

			float3 bounceSurfacePos = bounceOrigin + bounceDir * bouncePayload.minT
				+ bouncePayload.interpolatedNormal * 0.001f;

			float unShadowed = perSample(bounceSurfacePos, bouncePayload.interpolatedNormal, pointLightPos, rndState);

			float3 V = normalize(bounceOrigin - bounceSurfacePos);
			float3 L = normalize(pointLightPos - bounceSurfacePos);
			float3 direct = EvalGGXBRDF(bouncePayload.interpolatedNormal, V, L, bouncePayload.materialColor, bouncePayload.metallic, max(bouncePayload.roughness, kMinPunctualLightRoughness));
			hdr += throughput * direct * lightIntensity * unShadowed;

			if (bounce + 1 == totalHits)
				break; // last hit: no need to sample a further bounce direction

			// Stochastic lobe selection for the next bounce direction: each bounce either samples
			// the GGX specular lobe (importance-sampled around the half-vector, so low roughness
			// gives tight near-mirror bounces and high roughness spreads towards diffuse-like
			// scattering) or the existing cosine-weighted diffuse lobe -- picking one per bounce,
			// weighted by specProb, rather than evaluating both, is what keeps this "one ray per
			// bounce" instead of turning into its own multi-sample estimator. V (view direction,
			// toward the previous vertex) was already computed above for the direct-lighting term.
			float3 f0 = GetF0(bouncePayload.materialColor, bouncePayload.metallic);
			float specProb = saturate(max(f0.r, max(f0.g, f0.b)));

			{
				bounceDir = normalize(getCosHemisphereSample(rndState, bouncePayload.interpolatedNormal));
				// glTF energy conservation: metals have no diffuse component (same convention
				// EvalGGXBRDF uses); dividing by (1-specProb) keeps the stochastic estimator
				// unbiased since this lobe is only taken a (1-specProb) fraction of the time.
				throughput *= bouncePayload.materialColor * (1.0f - bouncePayload.metallic) / (1.0f - specProb);
			}

			bounceOrigin = bounceSurfacePos;
		}
	}
*/
//	output[index] = newColor;
}

[shader("miss")]
void rayMissMain(inout Payload payload)
{
}

[shader("closesthit")]
void rayClosestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	float3 bary = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

	uint entityID = InstanceID(); // gl_InstanceCustomIndexEXT, user defined -- the *entity's* index (see Renderer::createTopLevelAS)
	uint instanceIndex = InstanceIndex(); // Which instance was hit, system generated
	// entityID -> meshID: one indirection, since a TLAS instance can only carry one InstanceID() and
	// interaction.slang's pick ray needs that to be the entity index, not the mesh index (materials
	// always come from Mesh::m_primitiveMaterials below, never from instance data) -- see
	// RayPushData::entityMeshIndexAddress's comment.
	uint meshID = ((uint*)pushData.entityMeshIndexAddress)[entityID];
	uint triID = PrimitiveIndex(); // Which triangle within the mesh was hit
	
	payload.instanceIndex = instanceIndex;
	payload.minT = RayTCurrent();

	// [meshID] = { IB, VB, primitiveMaterials }
	uint64_t* meshBuffer = (uint64_t*)pushData.meshBufferAddress + 3 * meshID;
	// IB
	uint* indexBuffer = (uint*)meshBuffer[0];
	// VB
	Vertex * vertexBuffer = (Vertex *)meshBuffer[1];

	Vertex v0 = vertexBuffer[indexBuffer[triID * 3 + 0]];
	Vertex v1 = vertexBuffer[indexBuffer[triID * 3 + 1]];
	Vertex v2 = vertexBuffer[indexBuffer[triID * 3 + 2]];

	float3 objectNormal = bary.x * v0.normal + bary.y * v1.normal + bary.z * v2.normal;
	float2 uv = bary.x * v0.texCoord + bary.y * v1.texCoord + bary.z * v2.texCoord;
//	float4 tangent = bary.x * v0.tangent + bary.y * v1.tangent + bary.z * v2.tangent;

	// Per-primitive material (see Mesh::m_primitiveMaterials): every mesh always has this fully
	// populated, one valid material index per triangle -- lets one TLAS instance carry multiple
	// materials, matching the rasterizer's per-submesh draw loop, which instanceCustomIndex alone
	// couldn't (only one value per instance).
	uint64_t primitiveMaterialsAddress = meshBuffer[2];
	uint materialID = ((uint*)primitiveMaterialsAddress)[triID];

	MaterialData* materialBuffer = (MaterialData*)pushData.materialBufferAddress + materialID;

	// Normals need the inverse-transpose of the instance's object-to-world transform, not the
	// transform itself -- using it directly (or skipping the transform, as before) only happens to
	// look right for instances with no rotation (identity-ish transforms), and silently breaks for
	// any rotated/scaled instance. mul(v, WorldToObject3x4()) is the standard HLSL/Slang ray tracing
	// idiom for this: WorldToObject3x4() is the inverse of the instance transform, and mul(vector,
	// matrix) here computes transpose(matrix) * vector -- i.e. the inverse-transpose.
	float3 interpolatedNormal = mul(objectNormal, (float3x3)WorldToObject3x4());
	
	payload.interpolatedNormal = interpolatedNormal;
}
