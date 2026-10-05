#pragma once
#include "vulkanBase.h"
#include "Renderer.h"
#include "Shader.h"
#include "GBufferTarget.h"

// All state for the custom TAA (Temporal Anti-Aliasing) resolve pass (shaders/taa_resolve.slang) --
// see that shader's own header comment for the algorithm. Grouped into one struct/global (matching
// this codebase's NrdDenoiser/g_nrdDenoiser convention, NrdIntegration.h) instead of a dozen loose
// globals in App.cpp, which is where every one of these used to live before this file
// existed -- see git history for the original per-field comments this groups together.
struct TaaResolveState
{
	// Ping-pong history images: taa_resolve.slang writes its resolved output via a bindless
	// RWTexture2D.Handle(...) store (like ray.slang's own G-buffer writes), not a real color
	// attachment -- see EnsureTaaHistoryCapacity's (App.cpp) comment for the full
	// reasoning, including why ping-ponging is needed at all (reading and writing the same image in
	// one dispatch isn't safe once a neighborhood clamp is involved). GBufferTarget (shared with the
	// unrelated raster g_gbufferAlbedo/etc. globals, see its own header) so createGBufferTarget()/
	// destroyGBufferTarget() (App.cpp) can build/free these the same way.
	GBufferTarget history[2];

	// Storage-image (write) registration for history[] in the bindless descriptor heap.
	uint32 historyHeapIndex[2] = { ~0u, ~0u };
	// Same two images, registered again as SAMPLED (filtered) heap entries -- a storage image only
	// supports point Load(), but reprojected (sub-pixel) history reads need bilinear filtering.
	uint32 historySampledHeapIndex[2] = { ~0u, ~0u };

	// Which of history[]/historyHeapIndex[] this frame resolves *into* -- decided once, at the top
	// of drawFrame(), before ImGui::Image() (which needs to reference the right descriptor before
	// the resolve pass that fills it in has even been recorded yet) and reused by updateSceneBuffer()/
	// the resolve dispatch later that same frame. 1-historyIndex is this frame's history (read).
	uint32 historyIndex = 0;

	// False on the first frame after start/resize/scene reload -- nothing valid in either history
	// slot yet, so taa_resolve.slang just copies the current frame through unblended that frame.
	bool historyValid = false;

	// Grow-only capacity (EnsureTaaHistoryCapacity) -- NOT necessarily today's viewport size, see
	// SceneInfo::taaHistoryCapacityWidth/Height's own comment (shader_io.h) for why the shader needs
	// to know this separately from `resolution`.
	uint32 capacityWidth = 0;
	uint32 capacityHeight = 0;

	// ImGui descriptor sets for the two history images, created once alongside them (not per-frame --
	// ImTexture::update() allocates/frees an ImGui descriptor set every call, which every-frame
	// ping-pong switching would do 60+ times a second for no reason).
	utils::ImTexture imTextureHistory[2];

	// Heap index for g_gbufferMotionVector (read-only here; ray.slang already writes it via its own
	// traditional RT descriptor binding, unrelated to this heap registration) and for
	// m_renderTarget's current color/depth (read-only, this frame's raw jittered image and, when
	// rayTracing==0, the rasterizer's real per-pixel depth for taa_resolve.slang's own camera
	// reprojection -- see SceneInfo::taaDepthHeapIndex's comment, shader_io.h). All three
	// re-registered whenever the underlying image is recreated (resize) -- a heap index refers to a
	// specific VkImageView, which doesn't survive that.
	uint32 gbufferMotionVectorHeapIndex = ~0u;
	uint32 renderTargetColorHeapIndex = ~0u;
	uint32 renderTargetDepthHeapIndex = ~0u;

	// History/current blend factor, live-tunable (Settings window) -- see its own comment
	// (shader_io.h's SceneInfo::taaHistoryWeight) for why this isn't a fixed shader constant.
	float historyWeight = 0.95f;

	// Post-blend unsharp-mask strength, live-tunable (Settings window) -- see its own comment
	// (shader_io.h's SceneInfo::taaSharpenAmount) for why the history blend needs this on top of the
	// sampler's own mip bias (createDescriptorHeap, App.cpp).
	float sharpenAmount = 0.5f;

	// The resolve shader itself -- auto-compiled/hot-reloadable via the existing IAppInit/EventManager
	// mechanism, same as g_gfx/g_post/g_ray.
	Shader resolveShader{ "shaders/taa_resolve.slang", "vertexMain", "fragmentMain" };
};

extern TaaResolveState g_taa;
