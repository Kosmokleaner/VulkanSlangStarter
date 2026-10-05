
// Mouse-picking pass: a copy of ray.slang's raygen/miss/closesthit trio, stripped down to a
// single, non-reflective ray shot through the mouse cursor's screen position (instead of one ray
// per pixel) -- no lighting, no shadows, no bounces, no RTXDI. Dispatched at a fixed {1,1} size
// (see App::dispatchRayTracing), it only needs to know which object (if any) is under
// the cursor. The result is written into the same buffer ray.slang writes its own output to
// (scene.outputAddress), in the "extra data" region past the visible pixels -- see
// kInteractionResultOffset (shader_io.h) -- for post.slang's fragmentMain to read and print via
// s2h_printInt() this same frame. Later this can be read back on the CPU instead/as well; for now
// it stays GPU-side.

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

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure g_tlas;

// Traditional (non-heap) texture binding -- unused here (the closest-hit shader below needs no
// material/texture lookup, only which instance was hit), but the descriptor set layout is shared
// across every ray tracing Shader (see Shader::createRay), so these still have to be declared to
// match it, same as laser.slang.
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
	// InstanceID() (gl_InstanceCustomIndexEXT) of the hit instance -- the picked entity's own raw
	// index into scene.m_sceneObjects (see Renderer::createTopLevelAS's comment on
	// asInstance.instanceCustomIndex), not InstanceIndex() (a TLAS-instance-array position, a
	// different, mesh-only index space) -- CPU side reads this straight into GetHandleByIndex().
	// -1 if not set
	int entityIndex;
	int materialID;
	// used like z buffer -- also doubles as "did we hit anything" (INFINITE on a miss)
	float minT;
};

[shader("raygeneration")]
void rayGenMain()
{
	SceneInfo * scene = (SceneInfo *)pushData.sceneInfoAddress;

	bool bMaterial = (scene.mouseButtons & (1u << 31u)) != 0;	// LMB + Ctrl

	uint2 resolution = (uint2)scene.resolution;
	// same "extra data" trailing region ray.slang/laser.slang use, on the buffer ray.slang
	// actually writes its own pixel output to (scene.outputAddress) so this frame's post pass can
	// read it back immediately -- see kInteractionResultOffset's comment (shader_io.h).
	float4* result = (float4*)scene.outputAddress + resolution.x * resolution.y + kInteractionResultOffset;

	result[1] = float4(asuint(bMaterial?1:0),0,0,0);

	if (any(scene.pxMousePos == FLT_MAX) || (scene.mouseButtons & 0x1) == 0)
	{
		*result = float4(0, 0, 0, asfloat(-1));
		return;
	}

	RayDesc ray;
	ray.Origin = S2S_CAMERA_POS();
	ray.TMin = 0.001; // Minimum distance to avoid self-intersection
	ray.TMax = INFINITE;
	{
		float4x4 worldToClip = MultiplyMatrices(GetPerspMatrix(), GetViewMatrix());
		float4x4 INV_VIEW_PROJECTION = inverse(worldToClip);

		float3 worldPos;
		{
			float2 screenPos = (scene.pxMousePos / scene.resolution) * 2.0f - 1.0f;
			screenPos.y = -screenPos.y;
			float4 worldPosHom = mul(INV_VIEW_PROJECTION, float4(screenPos, S2S_NEAR(), 1));
			worldPos = worldPosHom.xyz / worldPosHom.w;
		}
		ray.Direction = normalize(worldPos - S2S_CAMERA_POS());
	}

	Payload payload = (Payload)0;
	payload.entityIndex = -1;
	payload.materialID = -1;

	// Single ray, no reflection/bounce: RAY_FLAG_NONE, plain TraceRay (not TraceRay2 -- no debug
	// line/laser rendering needed for a pick ray).
	TraceRay(g_tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

	float3 worldHitPos = ray.Origin + ray.Direction * payload.minT;

	float alpha = asfloat(payload.entityIndex); // LMB

	if(bMaterial)
		alpha = asfloat(payload.materialID);

	*result = float4(payload.entityIndex != -1 ? worldHitPos : float3(0, 0, 0), alpha);
}

[shader("miss")]
void rayMissMain(inout Payload payload)
{
	payload.entityIndex = -1;
	payload.materialID = -1;
	payload.minT = INFINITE;
}

[shader("closesthit")]
void rayClosestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
	uint entityID = InstanceID(); // gl_InstanceCustomIndexEXT, user defined -- the *entity's* index (see Renderer::createTopLevelAS)
	uint triID = PrimitiveIndex(); // Which triangle within the mesh was hit
	uint meshID = ((uint*)pushData.entityMeshIndexAddress)[entityID];
	uint64_t* meshBuffer = (uint64_t*)pushData.meshBufferAddress + 3 * meshID;
	uint64_t primitiveMaterialsAddress = meshBuffer[2];
	uint materialID = ((uint*)primitiveMaterialsAddress)[triID];

	// Only the entity index and hit distance are needed for picking -- no vertex/normal/material
	// lookup like ray.slang's/laser.slang's closest-hit shaders.
	payload.entityIndex = InstanceID(); // gl_InstanceCustomIndexEXT, user defined -- see Payload's comment
	payload.materialID = materialID;
	payload.minT = RayTCurrent();
}
