// Custom TAA (Temporal Anti-Aliasing) resolve -- from scratch, no DLSS/FSR2/NRD involved. See
// g_taaHistory's own declaration comment (App.cpp) for why this exists: AA jitter
// (g_enableAA/doJitter, updateSceneBuffer()) intentionally runs in every mode now, including NRD
// Realtime, but nothing was resolving the resulting per-frame sub-pixel sampling into a stable image
// there -- this pass is that resolve, shared by both the rasterizer and ray tracing.
//
// Runs as a full-screen "draw" (3 vertices, no bound vertex buffer -- see vertexMain) purely to
// invoke the fragment shader once per pixel; it has no real color/depth attachment at all (see the
// vkCmdBeginRendering call site, App.cpp) and writes its result via a bindless
// RWTexture2D.Handle(...) store instead of SV_TARGET, the same way ray.slang's rayGenMain writes its
// G-buffer targets.
//
// Algorithm: reproject last frame's resolved history to this pixel via a motion vector (ray
// tracing's own g_gbufferMotionVector when rayTracing!=0; reconstructed from this frame's real
// rasterized depth + camera reprojection otherwise -- ray tracing doesn't write real depth into
// m_renderTarget, see taaDepthHeapIndex's comment in shader_io.h), clamp it into the current frame's
// local 3x3 neighborhood (the standard ghosting fix: a stale/wrong reprojection can't drag in a
// color the current frame has no support for), and blend -- mostly history, a little current frame,
// for a temporally stable result. Falls back to the raw current frame whenever there's no history
// yet (first frame after start/resize/scene reload) or the reprojected UV lands off-screen.

#include "shader_io.h"
#include "mathlib.slang"

struct VertexOutput
{
	float4 position : SV_Position;
};

// Push data (vkCmdPushDataEXT): only sceneInfoAddress is used here; this reuses GraphicsPushData
// (like post.slang/shader.rast.slang) purely so the same push-data call site/struct can push it,
// materialBufferAddress/materialAndMeshID/modelMatrix are unused by this shader.
[[vk::push_constant]]
ConstantBuffer<GraphicsPushData> pushData;

// Same "3 vertices fill the screen" idiom as post.slang/shader.rast.slang, minus the VertexInput
// parameter those take -- this draw never binds a vertex buffer, so there's nothing to read even if
// declared.
[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID)
{
	VertexOutput output = (VertexOutput)0;

	if (vertexID == 0)
		output.position = float4(-10, -10, 1.0f, 1);
	else if (vertexID == 1)
		output.position = float4(10, -10, 1.0f, 1);
	else
		output.position = float4(0, 20, 1.0f, 1);

	// DirectX NDC y=1 is top, -1 is bottom; OpenGL/Vulkan NDC y=-1 is top, 1 is bottom -- same flip
	// every other vertex shader here applies.
	output.position.y = -output.position.y;

	return output;
}

[shader("pixel")]
void fragmentMain(VertexOutput input)
{
	SceneInfo* scene = (SceneInfo*)pushData.sceneInfoAddress;

	// Defensive: heap indices are only valid once EnsureTaaHistoryCapacity/onViewportSizeChange has
	// run at least once (App.cpp) -- the C++ call site already guards the whole dispatch
	// on this, but checking here too costs nothing and this project has already paid dearly once this
	// session for an unguarded invalid heap-index access (see VK_EXT_device_fault's write-to-null
	// investigation) to ever risk it again casually.
	if (scene.taaCurrentColorHeapIndex == kInvalidTextureId || scene.taaHistoryWriteHeapIndex == kInvalidTextureId)
		return;

	uint2 pixel = (uint2)input.position.xy;

	Texture2D currentTex = Texture2D.Handle(uint2(scene.taaCurrentColorHeapIndex, 0));
	float4 current = currentTex.Load(int3(pixel, 0));

	RWTexture2D<float4> historyOut = RWTexture2D<float4>.Handle(uint2(scene.taaHistoryWriteHeapIndex, 0));

	if (scene.taaHistoryValid == 0)
	{
		historyOut[pixel] = current;
		return;
	}

	// Motion vector, current -> previous, in screen-fraction units (matches ray.slang's own
	// g_gbufferMotionVector convention exactly, and reused directly when it's the source below).
	float2 motion;
	if (scene.rayTracing != 0)
	{
		if (scene.taaMotionVectorHeapIndex == kInvalidTextureId)
			return;
		Texture2D mvTex = Texture2D.Handle(uint2(scene.taaMotionVectorHeapIndex, 0));
		motion = mvTex.Load(int3(pixel, 0)).rg;
	}
	else
	{
		if (scene.taaDepthHeapIndex == kInvalidTextureId)
			return;
		Texture2D depthTex = Texture2D.Handle(uint2(scene.taaDepthHeapIndex, 0));
		float depth = depthTex.Load(int3(pixel, 0)).r;

		// Background/sky (depth cleared to 0, never written by a real fragment -- see
		// ComputePerspectiveMatrix's own comment below for why 0 means infinitely far here, not
		// "nearest"): no real geometry to reproject via a motion vector, but falling through to the
		// SAME blend/clamp/sharpen path below with motion=(0,0) (this pixel's own sky sample didn't
		// move) instead of discarding history outright lets background pixels keep accumulating
		// history too.
		//
		// That specifically matters at silhouette edges, not just open sky: a jittered rasterized
		// edge toggles a given pixel between "foreground" and "background" classification from frame
		// to frame (this frame's sub-pixel jitter offset happens to land the edge just past this
		// pixel, or just before it). Unconditionally bypassing history here on every frame classified
		// as background threw away whatever had already accumulated at that pixel instead of blending
		// it in -- the confirmed root cause (via two consecutive --selftest captures of a fully static
		// raster scene, zero camera motion, differing only at exactly these silhouette pixels, and
		// unaffected by raising scene.taaHistoryWeight) of the user-reported "TemporalAA has visible
		// jitter movement...when camera is stationary, raster" bug. motion=(0,0) is only exactly
		// correct for a non-rotating camera (true sky parallax depends on rotation, which this doesn't
		// account for) -- not worse than before though, since a rotating camera already got zero
		// motion compensation for background pixels either way.
		if (depth <= 0.0f)
			motion = float2(0.0f, 0.0f);
		else
		{
		// shader.rast.slang's vertexMain subtracts this same offset from the rasterized clip position
		// (its own AA jitter comment) -- input.position.xy below is that already-jittered pixel, so
		// ndc here is the JITTERED NDC, not the true one this reconstruction needs. Left uncorrected,
		// every reconstructed world position -- and therefore every reprojected motion vector -- is
		// off by this frame's jitter offset, which changes every frame (ComputeTemporalJitter's
		// Halton sequence): exactly the kind of per-frame reprojection error that reads as shimmer.
		//
		// X and Y need OPPOSITE signs here, not a uniform "+jitterShift": vertexMain applies its
		// jitter shift to clip-space xy AFTER its own "output.position.y = -output.position.y" NDC
		// flip (DirectX/GLM y-up -> Vulkan y-down), but this ndc.y below has *already* been flipped
		// back (the "ndc.y = -ndc.y" line below) to match this project's un-flipped camera-space
		// convention -- so that flip already re-introduces the Y jitter shift once on its own; adding
		// it again (as an earlier version of this fix did) doubled it, a Y-only error that grew/
		// shrank with the jitter sequence every frame.
		float2 ndc = (input.position.xy / scene.resolution) * 2.0f - 1.0f;
		ndc.y = -ndc.y;
		float2 jitterShift = (scene.temporalJitter - 0.5f) * (2.0f / scene.resolution);
		ndc += float2(jitterShift.x, -jitterShift.y);

		// Do NOT reconstruct this via scene.clipFromView's inverse -- shader.rast.slang's vertexMain
		// (the shader that actually wrote this depth buffer) doesn't use scene.clipFromView at all;
		// its GetPerspMatrix() calls ComputePerspectiveMatrix() (mathlib.slang), whose Z/W row is a
		// deliberately non-standard hack (see that function's own comment): it writes clipPos.z as a
		// hardcoded constant and clipPos.w = view-space Z, so after the perspective divide this
		// depth buffer holds `0.1 / viewZ`, NOT the linear-ish 0..1 range scene.clipFromView (a
		// completely different, standard glm::perspectiveLH_ZO matrix used for ray tracing/RTXDI)
		// would produce. Un-projecting this depth through scene.clipFromView's inverse silently
		// reconstructed a world position with the wrong, non-linearly-distorted depth -- confirmed
		// live as a reprojection artifact whose scale visibly grew with camera distance traveled
		// ("shrinks when flying backward"). Recovering the real view-space Z directly from the known
		// `0.1/viewZ` relationship, then unprojecting XY with the same tanHalfFov/aspect terms
		// ComputePerspectiveMatrix's own [0][0]/[1][1] use, sidesteps the bad Z row entirely.
		float viewZ = 0.1f / depth;
		float tanHalfFov = tan(scene.fieldOfViewDeg * PI / 180.0f * 0.5f);
		float aspect = scene.resolution.x / scene.resolution.y;
		float2 viewXY = ndc * viewZ * float2(aspect, 1.0f) * tanHalfFov;

		// scene.viewFromWorld is world-to-view (see its own "aka view, worldToView" comment,
		// shader_io.h) despite what its name suggests -- invert it to get the view-to-world
		// transform this needs.
		float4 worldPosHom = mul(inverse(scene.viewFromWorld), float4(viewXY, viewZ, 1));
		float3 worldPos = worldPosHom.xyz;

		// Reproject through the *previous* frame's camera to find where this same world point was
		// on screen last frame -- camera motion only (no per-object motion), same limitation
		// g_gbufferMotionVector's own ray-traced version already has (it reprojects a fixed hit
		// world position through the camera's motion too, not the object's).
		float4x4 prevWorldToClip = MultiplyMatrices(scene.prevClipFromView, scene.prevViewFromWorld);
		float4 prevClipPos = mul(prevWorldToClip, float4(worldPos, 1));
		float2 prevNdc = prevClipPos.xy / prevClipPos.w;
		float2 prevScreenPos = (prevNdc * float2(0.5f, -0.5f) + 0.5f) * scene.resolution;

		motion = (prevScreenPos - (float2(pixel) + 0.5f)) / scene.resolution;
		}
	}

	// motion is in units of a fraction of `resolution` (this frame's viewport, see its own comment);
	// convert to an actual pixel position in that same space first and validate against `resolution`
	// -- NOT against the history texture's own (larger, grow-only) dimensions, see
	// taaHistoryCapacityWidth/Height's own comment -- before normalizing by the texture's real size
	// for the SampleLevel() below. Checking prevUV against [0,1] directly (as if the texture's real
	// size equalled `resolution`) let a reprojection that's actually off-screen silently land in the
	// stale leftover region beyond this frame's written sub-rect instead of being caught here.
	float2 prevPixelPos = (float2(pixel) + 0.5f) + motion * scene.resolution;

	if (any(prevPixelPos < 0.0f) || any(prevPixelPos > scene.resolution) || scene.taaHistoryReadHeapIndex == kInvalidTextureId)
	{
		historyOut[pixel] = current;
		return;
	}

	Texture2D historyTex = Texture2D.Handle(uint2(scene.taaHistoryReadHeapIndex, 0));
	SamplerState samp = SamplerState.Handle(uint2(0, 0));
	float2 prevUV = prevPixelPos / float2(scene.taaHistoryCapacityWidth, scene.taaHistoryCapacityHeight);
	float4 history = historyTex.SampleLevel(samp, prevUV, 0.0f);

	// Neighborhood clamp: bound the reprojected history to this frame's local 3x3 min/max box around
	// this pixel -- the standard TAA ghosting fix. A stale reprojection (disocclusion, something that
	// just moved into view) gets pulled back toward what the current frame actually supports instead
	// of dragging an old, now-wrong color forward.
	float3 minC = current.rgb;
	float3 maxC = current.rgb;
	const int2 maxPixel = int2(scene.resolution) - 1;
	[unroll]
	for (int dy = -1; dy <= 1; ++dy)
	{
		[unroll]
		for (int dx = -1; dx <= 1; ++dx)
		{
			if (dx == 0 && dy == 0)
				continue;
			int2 samplePixel = clamp(int2(pixel) + int2(dx, dy), int2(0, 0), maxPixel);
			float3 c = currentTex.Load(int3(samplePixel, 0)).rgb;
			minC = min(minC, c);
			maxC = max(maxC, c);
		}
	}
	float3 clampedHistory = clamp(history.rgb, minC, maxC);

	// Mostly history for temporal stability; a small blend-in of the current frame every frame keeps
	// the image responsive instead of needing an explicit history-reset heuristic beyond the
	// first-frame/off-screen cases already handled above. Live-tunable -- see its own comment
	// (shader_io.h) for why this isn't a fixed constant.
	float3 resolved = lerp(current.rgb, clampedHistory, scene.taaHistoryWeight);

	// Unsharp mask: the history blend above is inherently a low-pass filter (bilinear-resampled,
	// blended frame after frame), which no amount of sampler mip bias can fully counteract -- that
	// bias only sharpens the CURRENT frame's own raw texture sample, not this temporally-smoothed
	// result. Extract high-frequency detail from a "what detail should be here" reference and inject
	// it back into the resolved color -- the standard unsharp-mask trick.
	//
	// That reference is the HISTORY texture (already sampled above into `history`), NOT the CURRENT
	// frame -- confirmed live as the actual root cause of a user-reported "TemporalAA has visible
	// jitter movement" bug that persisted even with a fully stationary camera in raster mode: two
	// consecutive --selftest captures of the same static scene, zero camera/scene change between
	// them (only normal frame advancement), differed at exactly the object/sky silhouette pixels.
	// CURRENT is this frame's raw, single-sample rasterization, shifted by a different sub-pixel
	// jitter offset every frame (ComputeTemporalJitter's Halton sequence, App.cpp) -- at any hard
	// edge it's always exactly one extreme color or the other, never antialiased, so sharpening ITS
	// own edges every frame re-injects exactly the per-frame jitter noise the history blend above
	// just spent effort averaging out. HISTORY is already temporally blended across many frames and
	// barely changes frame-to-frame with a static camera, so sharpening relative to ITS own
	// neighborhood sharpens the image's real, stable detail instead of the jitter pattern.
	{
		float2 historyTexel = 1.0f.xx / float2(scene.taaHistoryCapacityWidth, scene.taaHistoryCapacityHeight);
		float3 up    = historyTex.SampleLevel(samp, prevUV + float2(0.0f, -historyTexel.y), 0.0f).rgb;
		float3 down  = historyTex.SampleLevel(samp, prevUV + float2(0.0f,  historyTexel.y), 0.0f).rgb;
		float3 left  = historyTex.SampleLevel(samp, prevUV + float2(-historyTexel.x, 0.0f), 0.0f).rgb;
		float3 right = historyTex.SampleLevel(samp, prevUV + float2( historyTexel.x, 0.0f), 0.0f).rgb;
		float3 blurred = (up + down + left + right) * 0.25f;
		float3 detail = history.rgb - blurred;
		resolved += detail * scene.taaSharpenAmount;
	}

	historyOut[pixel] = float4(resolved, 1.0f);
}
