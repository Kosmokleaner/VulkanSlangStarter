/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


 // ------------------------------------------------------------------

 // single header libarary
#define VMA_IMPLEMENTATION

#include "global.h"	// include first in any cpp

#include <crtdbg.h>	// _CrtSetReportMode/_CrtSetReportFile, used in main() to suppress the assert dialog
#include <cstdio>	// freopen_s, used in main() to attach stdio to a launching console (AttachConsole)
#include <iostream>	// std::cout/std::ios, ditto

#include "resource.h"
#include "Shader.h"
#include "RelativeMouseInput.h"
#include "EventManager.h"
#include "Scene.h"
#include "Material.h"
#include "Platform.h"
#include "AppendArena.h"
#include "3DBSPLevel.h"
#include "Texture.h"
#include "IBLEnvironment.h"
#include "Renderer.h"
#include "Selection.h"
#include "WindowPersist.h"
#include "IniArchive.h"
#include "Reflection.h"
#include "TestMode.h"
#include "TestFramework.h"
#include "DeferredRelease.h"
#include "App.h"
#include "RenderComponent.h"
#include "CameraComponent.h"
#include "LightComponent.h"
#include "Transform.h"
#include "ImGuizmo.h"
#include "imgui_stdlib.h"
#include <glm/gtc/type_ptr.hpp>
#include <filesystem>

#if USE_RAY_TRACING
// rtxdi::ReSTIRDIContext (host-side parameter/buffer-size math only -- no D3D/Vulkan calls of its
// own, see extern/Rtxdi/Source/ReSTIRDI.cpp) and rtxdi::FillNeighborOffsetBuffer/
// CalculateReservoirBufferParameters (extern/Rtxdi/Source/RtxdiUtils.cpp). The parameter structs
// this transitively includes (RTXDI_RuntimeParameters etc.) are also used un-namespaced from
// shaders/rtxdi_io.h (see Buffer.h's own #include of the same headers, at global scope, before its
// `namespace shaderio` block) -- both inclusions resolve to the same single definition via the
// normal #ifndef header guards.
#include "Rtxdi/DI/ReSTIRDI.h"

// NVIDIA NRD (Real-time Denoiser) -- vendored as a prebuilt library (extern/NRD, built locally
// from https://github.com/NVIDIA-RTX/NRD commit bf877181058988ec5785f82c4189be0be75c1902, since it
// ships no binary releases; see readme.md's extern section), not vendored source like RTXDI, since
// its shaders are pre-embedded SPIR-V/DXIL/DXBC in the .lib -- nothing left for this project's own
// build to compile. NRD.h's API is pure C (extern "C", only POD structs/function pointers cross
// the boundary), so linking one Release-CRT build into both Debug/Release VVerse configs is safe
// -- same reasoning as extern\GLFW\glfw3.lib's existing IgnoreSpecificDefaultLibraries workaround.
//
// All actual instance/pipeline/dispatch management lives in NrdIntegration.h/.cpp (g_nrdDenoiser)
// -- see g_nrdRealtimeMode below for the Realtime/Reference toggle this exposes in the Settings
// window, and dispatchRayTracing()'s call site for where it runs each frame.
#include "NrdIntegration.h"
#include "GBufferTarget.h"
#include "TaaResolve.h"
#endif

WindowPersist g_windowPersist;

CRelativeMouseInput g_MouseInput;
uint32 g_temporalAccumulator = 0;
uint32 g_frameIndex = 0;

Renderer g_renderer;

Selection g_selection;

CEntityHandle g_currentCamera;

Shader g_gfx("shaders/shader.rast.slang", "vertexMain", "fragmentMain");
Shader g_post("shaders/post.slang", "vertexMain", "fragmentMain");
Shader g_s2h_3d("shaders/s2h_3d.slang", "vertexMain", "fragmentMain");
// Ray tracing shader. Explicitly dispatched by name from App::dispatchRayTracing()/
// updateRayDescriptorSet() below, the same way g_gfx/g_post are explicitly bound in
// recordGraphicsCommands() -- no frame graph, no automatic discovery. Adding another ray tracing
// shader means declaring it here (e.g. Shader g_ray2(...)) and adding its own explicit dispatch
// call at the two call sites below; everything else (SBT, per-frame-slot descriptor set) is
// self-contained in Shader.
Shader g_ray("shaders/ray.slang", "rayGenMain", "rayMissMain", "rayClosestHitMain");
Shader g_laser("shaders/laser.slang", "rayGenMain", "rayMissMain", "rayClosestHitMain");
// Mouse-picking pass: a single, non-reflective ray through the mouse cursor (dispatched at a fixed
// {1,1} size, see dispatchRayTracing()), writing which object is under it into the same buffer
// g_ray writes its own output to -- see interaction.slang and kInteractionResultOffset (shader_io.h).
Shader g_interaction("shaders/interaction.slang", "rayGenMain", "rayMissMain", "rayClosestHitMain");

// FLT_MAX, FLT_MAX if not valid
float2 g_pxMousePos = float2(FLT_MAX, FLT_MAX);
// bit 0: LMB, bit 31:Ctrl
uint32 g_mouseButtons = 0;

// Screen-space rect of the "Viewport" window's content region, captured once per frame (see the
// mouse-position block in run()) so the gizmo (drawn later, in drawFrame()) can be positioned with
// ImGuizmo::SetRect() using the same rect the mouse-picking code already computes.
ImVec2 g_viewportScreenPos(0, 0);
ImVec2 g_viewportSize(0, 0);

// GPU->CPU readback of interaction.slang's picked-object index (see kInteractionResultOffset,
// shader_io.h): two uint32s (index, isMaterial -- interaction.slang's result[0].w and result[1].x,
// contiguous in the source buffer so one copy grabs both) per frame-in-flight
// (Renderer::kMaxFramesInFlight). App::dispatchRayTracing() copies into slot
// m_swapchain.getFrameResourceIndex()'s pair every frame; App::GetPickResult() reads that
// same pair back ~kMaxFramesInFlight frames later, right after prepareFrameResources()'s existing
// timeline-semaphore wait has already proven that slot's GPU work (including this copy) completed
// -- so the read never stalls the GPU waiting on a copy that hasn't finished, at the cost of that
// same couple-frame-old lag. Host-visible and persistently mapped: read directly, no per-frame
// map/unmap. Raw array -- GetPickResult() is the only intended reader, everything else should go
// through it instead of indexing this (or knowing about frame-in-flight slots) directly.
utils::Buffer g_ReadbackBuffer;
uint32* g_ReadbackMapped = nullptr;

// See global.h's own declaration/comment (its ASSERT() redefinition is the actual reader) -- set
// below in main(), right after AttachParentConsole().
bool g_suppressAssertDialogs = false;

// Flat (0,0,1) tangent-space normal -- "no bump" -- engine texture (isEngineTexture=true, like
// ReplaceMe/Renderer.cpp; not sRGB, see Texture.h's m_isSRGB comment: this is direction data, not
// color), created once in init(). Every material's own MaterialData::normalMapTextureId is resolved
// to this texture's heap index whenever Material::m_hndNormalMap is unset (see the MaterialData
// fill loop below), so ray.slang's normal-map sampling never needs a branch for "does this material
// even have one" -- it unconditionally samples a normal map, and an unset one just samples this.
CTexHandle g_hndDefaultNormalMap;

C3DBSPLevel g_BSPLevel;

// [mesh->getHandle().index] = IB+VB
Buffer g_meshBuffer(EF_Type_Unknown);
// [entity->getHandle().index] = mesh->getHandle().index -- see RayPushData::entityMeshIndexAddress
// (shader_io.h) for why a closest-hit shader needs this indirection instead of using InstanceID()
// (an entity index, see Renderer::createTopLevelAS) to index g_meshBuffer directly.
Buffer g_entityMeshIndexBuffer(EF_Type_Unknown);
// [material->getHandle().index] = Material
Buffer g_materialBuffer(EF_Type_Unknown);
// [0] = the single global light (g_lightPos/g_lightColor/g_lightIntensity), [1..] = every point
// light g_BSPLevel parsed from its level's "light" entities. Rebuilt on the CPU every frame (see
// updateLights()) -- allocated once, lazily, sized for shaderio::kMaxLights, then updated in place
// each frame via vkCmdUpdateBuffer (same pattern as m_sceneInfoBuffer).
Buffer g_lights(EF_Type_Unknown);
// Set by updateLights() each frame; read by App::updateSceneBuffer() for SceneInfo::lightCount
uint32 g_lightCount = {};

CRenderCompHandle g_plane;

shaderio::VisualizeMode::EEnum g_visualizeMode = {};

float3 g_skyRGB = { 0.2f, 0.2f, 0.3f };
// multiplier for sky color
float g_skyIntensity = 1.0f;

// Image-based lighting (raster path only so far -- see IBLEnvironment.h/shader.rast.slang).
// Loaded once in App::init(); not part of setupScene1()/setupScene2()'s per-scene
// content or the save/load format, same as g_lightMesh etc.
IBLEnvironment g_ibl;
bool g_flatSkyColor = true;

float g_ExposureLog = 1.0f;
bool g_enableTonemapper = false;
// See shader_io.h's SceneInfo::lightmapIntensity comment -- Quake-style "overbright bits"
// compensation, live-tunable since the right scale isn't known yet.
float g_lightmapIntensity = 1.0f;

float3 g_laserPos = float3(0.0f, 0.5f, -2.0f);
float g_laserAngle = 0.0f;
float g_laserIntensity = 15.4f;

// Homogeneous participating media (ray.slang): colored Beer-Lambert extinction plus
// Henyey-Greenstein forward/backward in-scattering toward the light. 0 density = off.
float3 g_fogColor = float3(0.5f, 0.6f, 0.7f);
float g_fogDensity = 0.0f;
// -1..1: >0 forward scattering, <0 backscattering, 0 isotropic
float g_fogAsymmetry = 0.7f;

std::string g_sceneFileName = "scene.aa";

bool g_InfoHUD = false;
bool g_drawLights = false;
bool g_enableLights = true;
bool g_enableTextures = true;
bool g_enableAA = true;

// Default mirrors ContextCreateInfo::enableValidationLayers' own NDEBUG-based default
// (vk_framework.h) so contextConfig.enableValidationLayers can just copy this unconditionally
// (App::init()) without changing behavior when no flag is passed. The --validate=0/1 command-line
// flag (main(), parsed before App::init() builds contextConfig) overrides it either way -- most
// usefully --validate=0 to let a Debug build skip the Vulkan Validation Layers for a quick perf
// check without rebuilding Release. Validation is instance-creation-time only in Vulkan, so this
// can't be flipped while the app is already running -- it has to be read before vkCreateInstance.
#ifdef NDEBUG
bool g_enableValidationLayers = false;
#else
bool g_enableValidationLayers = true;
#endif

#if USE_RAY_TRACING
bool g_rayTracing = true;
// ReSTIR DI (RTXDI) toggle -- see the "ReSTIR DI" checkbox (Settings window) and SceneInfo::enableRtxdi's
// comment (shader_io.h). Off falls back to plain uniform-light-picking direct lighting at the primary
// hit, letting RTXDI's own contribution to noise (e.g. "boiling") be compared against a plain estimator
// with everything else (denoiser, G-buffer, etc.) held equal.
bool g_enableRtxdi = true;
// Temporal/spatial reservoir reuse validity thresholds (RTXDI_DISpatioTemporalResamplingParameters,
// applied every frame in updateSceneBuffer() -- see its own call site) -- live-tunable (Settings
// window) and persisted (Settings.ini) since the right values are a scene-dependent judgment call,
// not something to hardcode. RTXDI_IsValidNeighbor (extern/Rtxdi) rejects reuse between the
// reprojected previous-frame pixel and the current one once their view-space depths differ by more
// than this fraction (RTXDI_CompareRelativeDifference) or their normals' dot product falls below
// this cosine -- any camera motion perturbs both slightly, so tight thresholds here read as
// reservoirs "invalidating" on the slightest movement instead of smoothly carrying reuse forward.
// Defaults match GetDefaultReSTIRDISpatioTemporalResamplingParams (ReSTIRDI.cpp), this project's
// starting point before any tuning.
float g_restirDepthThreshold = 0.1f;
float g_restirNormalThreshold = 0.5f;
int g_restirMaxHistoryLength = 20;
// Diagnostic-only: shows NRD's own REBLUR_VALIDATION debug overlay (accumulated-history/
// disocclusion visualization) instead of the denoised image -- see the "NRD Validation Overlay"
// checkbox (Settings window) and NrdDenoiser::DispatchParams::showValidation's comment. Only
// meaningful with VisualizeMode left at Default and NRD Realtime Denoise on.
bool g_nrdShowValidation = false;
// 0 = direct lighting only; N>0 = also trace N inline indirect bounces per pixel in
// rayGenMain (ray.slang). See the "Path Tracing Bounces" combo in the Settings window.
uint32 g_pathTracingBounces = 2;
// on = a different RNG seed (and so a different noise pattern) each frame; off = the exact
// same pattern every frame. See SceneInfo::frameNoiseSeed (shader_io.h) and rayGenMain
// (ray.slang), which is the only shader that consumes it.
bool g_animateNoise = true;
#endif

CRenderCompHandle g_lightMesh;

// https://sketchfab.com/3d-models/mech-m-6k-6eba9b1addb7411a85b0c86e41a39a2e 1k textures, there is a 2k texture version
//Mesh g_mesh("resources/mech_m-6k.glb", 0.03f);
// https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/SheenWoodLeatherSofa/README.md
CRenderCompHandle g_mesh; // no texture?
//
//Mesh g_mesh("resources/LiamCar.glb");
// https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/DiffuseTransmissionPlant/README.md
//Mesh g_mesh("resources/DiffuseTransmissionPlant.glb");
// https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/ABeautifulGame/README.md
//Mesh g_mesh("resources/ABeautifulGame.glb", 3.0f);	// many wasteful 2K textures

// double-buffered ray-traced output, ping-ponged per frame (see App::updateSceneBuffer)
// so this frame's fresh result can be temporally blended against last frame's without racing it
Buffer g_outputBuffer[2] = { Buffer(EF_Type_Unknown), Buffer(EF_Type_Unknown) };
// This frame's direct/NEE specular-lobe-only radiance + proxy hit distance (see ray.slang's
// hdrSpecular) -- NRD's REBLUR_DIFFUSE_SPECULAR specular channel input. Double-buffered/ping-ponged
// exactly like g_outputBuffer above -- Reference Accumulation mode needs last frame's specular
// value too, to keep accumulating it the same way it already accumulates diffuse; NRD's own
// dispatch only ever touches the "current" slot, same as it already does for g_outputBuffer.
Buffer g_outputBufferSpecular[2] = { Buffer(EF_Type_Unknown), Buffer(EF_Type_Unknown) };

#if USE_RAY_TRACING
// RTXDI (ReSTIR DI) state for the primary-hit direct-lighting term -- see shaders/rtxdi_io.h and
// shaders/rtxdi/RtxdiBridge.slang. Lazily (re)created in updateSceneBuffer() to at least the
// current viewport size (grow-only, same philosophy as EnsureOutputBufferCapacity/g_outputBuffer).
std::unique_ptr<rtxdi::ReSTIRDIContext> g_restirDIContext;
// One persistent buffer of packed reservoirs (RTXDI_PackedDIReservoir), addressed via
// g_restirDIContext's reservoir buffer parameters + a per-frame ping-ponged array index (see
// updateSceneBuffer()) -- not itself double-buffered the way g_outputBuffer/g_rtxdiNormals are.
Buffer g_rtxdiReservoirs(EF_Type_Unknown);
// Filled once at startup (rtxdi::FillNeighborOffsetBuffer) and never touched again.
Buffer g_rtxdiNeighborOffsets(EF_Type_Unknown);
// Packed surface normal per pixel (RTXDI_EncodeNormalizedVectorToSnorm2x16), ping-ponged like
// g_outputBuffer: this frame's hit normal is written to one slot while RTXDI reads last frame's
// normal from the other for temporal/spatial neighbor validity checks (RAB_GetGBufferSurface,
// RtxdiBridge.slang) -- the matching depth for that check reuses g_outputBuffer's existing alpha
// channel (payload.minT), so no separate depth buffer is needed.
Buffer g_rtxdiNormals[2] = { Buffer(EF_Type_Unknown), Buffer(EF_Type_Unknown) };
// One small GPU-visible copy of shaderio::RtxdiConstants, updated in place each frame (same
// pattern as m_sceneInfoBuffer) -- read by ray.slang via RayPushData::rtxdiConstantsAddress.
Buffer g_rtxdiConstants(EF_Type_Unknown);
uint32 g_rtxdiCapacityWidth = 0;
uint32 g_rtxdiCapacityHeight = 0;
// Previous frame's camera matrices, cached in updateSceneBuffer() -- needed to compute this
// frame's screen-space motion for RTXDI's temporal reprojection (SceneInfo only ever carries the
// current frame's matrices).
glm::mat4 g_rtxdiPrevViewFromWorld = glm::mat4(1.0f);
glm::mat4 g_rtxdiPrevClipFromView = glm::mat4(1.0f);

// Temporal resampling of per-vertex path-tracing contributions -- see shaders/gi_io.h and
// shaders/gi_reservoir.slang for exactly what this is (and is not: no RIS/reservoir weighting, no
// spatial/cross-pixel reuse, no stored sample point -- a plain per-vertex exponential moving
// average now). Despite the "GI" name (kept for UI continuity), this applies uniformly to direct
// lighting at the primary hit too, the same way emissive already does at every vertex -- not a
// separate RTXDI-gated special case. Own reservoir buffer, own small constants buffer, own
// grow-only capacity tracking, mirroring g_rtxdiReservoirs/g_rtxdiConstants's own lifecycle above.
bool g_enableRestirGI = false;
float g_giDepthThreshold = 0.1f;
float g_giNormalThreshold = 0.5f;
// EMA blend factor -- same role/style as g_taa.historyWeight (TaaResolve.h), just for these
// per-vertex lighting terms instead of the whole frame.
float g_giHistoryWeight = 0.9f;
// One buffer holding two ping-pong halves (reservoirPitch elements each, GI_RESERVOIR_SLOTS_PER_PIXEL
// slots per pixel within each half), mirroring g_rtxdiReservoirs' own array-pitch convention -- see
// EnsureGIReservoirCapacity's comment for why this is one pitched buffer rather than two separate
// ones like g_rtxdiNormals.
Buffer g_giReservoirs(EF_Type_Unknown);
// One small GPU-visible copy of shaderio::GIConstants, updated in place each frame (same pattern
// as g_rtxdiConstants) -- read by ray.slang via RayPushData::giConstantsAddress.
Buffer g_giConstants(EF_Type_Unknown);
uint32 g_giCapacityWidth = 0;
uint32 g_giCapacityHeight = 0;
// False on the first frame after a reset (resetTemporal) or resize -- see GIConstants::historyValid's
// own comment (gi_io.h) for why this is a dedicated flag rather than reusing
// SceneInfo::temporalAccumulator the way ray.slang's own prevColor discard does.
bool g_giHistoryValid = false;

// This frame's and the previous frame's raw (non-transposed) world-to-view/view-to-clip matrices,
// cached in updateSceneBuffer() for g_nrdDenoiser.dispatch() (dispatchRayTracing()'s call site,
// further down this file) -- NRD's own CommonSettings wants plain GLM column-major matrices
// directly (see NrdIntegration.h), unlike sceneInfo.viewFromWorld/g_rtxdiPrevViewFromWorld, which
// are transposed for row_major shader consumption. Kept separate from RTXDI's own prev-matrix
// cache above to avoid needing to undo that transpose here.
glm::mat4 g_nrdWorldToView = glm::mat4(1.0f);
glm::mat4 g_nrdViewToClip = glm::mat4(1.0f);
glm::mat4 g_nrdPrevWorldToView = glm::mat4(1.0f);
glm::mat4 g_nrdPrevViewToClip = glm::mat4(1.0f);

// G-buffer render targets: written directly by ray.slang's rayGenMain (as RWTexture2D, via new
// bindings 5-8 on the shared ray tracing descriptor set -- see Shader::updateRayDescriptorSet),
// not the buffer-device-address idiom used elsewhere in this codebase, since these exist to feed
// a future GPU-side denoiser (NRD) that expects real textures. Purely additive for now -- nothing
// reads them yet; RTXDI keeps using its own separate g_rtxdiNormals buffer untouched. One frame's
// worth each (not double-buffered like g_rtxdiNormals/g_outputBuffer): nothing needs last frame's
// values yet, unlike RTXDI's own temporal validity checks.
// GBufferTarget itself now lives in GBufferTarget.h -- shared with TaaResolveState::history
// (TaaResolve.h), which needs the same {image, view} pair for its own ping-pong images.
GBufferTarget g_gbufferAlbedo;
GBufferTarget g_gbufferNormalRoughness;
GBufferTarget g_gbufferViewZ;
GBufferTarget g_gbufferMotionVector;
uint32 g_gbufferCapacityWidth = 0;
uint32 g_gbufferCapacityHeight = 0;
// Defined near EnsureGBufferCapacity; forward-declared here so App::destroy() (earlier in
// this file) can free these at shutdown -- a plain VkImage/VkImageView, not a Buffer/Shader, so
// they're never reached by the IAppInit/g_deInitListener mechanism destroy() otherwise relies on
// (same reasoning as that function's explicit BLAS cleanup, see its comment).
static void destroyGBufferTarget(GBufferTarget& target);

// Custom TAA resolve (taa_resolve.slang, shaders/taa_resolve.slang): a from-scratch neighborhood-
// clamp temporal resolve, replacing what would otherwise be DLSS/FSR2/NRD's own job -- written
// because turning AA jitter on unconditionally (see g_enableAA/doJitter, updateSceneBuffer()) left
// every frame under-sampled with nothing resolving it into a stable image, both in Reference mode
// (fine there -- freeze-and-accumulate already resolves it over many frames) and, worse, in NRD
// Realtime mode, where a fresh unaccumulated jittered sample every frame just flickered.
//
// All state (history images, their heap indices, the resolve shader itself, ...) lives in
// TaaResolveState/g_taa (TaaResolve.h) -- see that struct's own per-field comments.
//
// Reuses createGBufferTarget/destroyGBufferTarget (same STORAGE|SAMPLED|TRANSFER_DST usage, GENERAL
// layout) for the two ping-pong history images: taa_resolve.slang writes its resolved output via a
// bindless RWTexture2D.Handle(...) store (like ray.slang's own G-buffer writes), not a real color
// attachment, so it doesn't need COLOR_ATTACHMENT_BIT or a depth/stencil attachment at all -- the
// "draw" is 3 vertices with no bound vertex buffer, just to invoke the fragment shader once per pixel.
//
// Ping-pong because next frame's resolve needs to read this frame's freshly-written output as
// "history" while writing a *new* output -- reading and writing the same image in one dispatch isn't
// safe (the neighborhood clamp samples several texels, some of which other invocations may have
// already overwritten by the time they're read).

// Realtime denoising via NRD (g_nrdDenoiser, NrdIntegration.h) as an alternative to the default
// freeze-and-accumulate "Reference" behavior -- see the "NRD Realtime Denoise" checkbox in the
// Settings window and updateSceneBuffer()'s use of this flag to force per-frame reset accumulation
// (NRD needs a fresh single-sample signal every frame, not our own running sum).
bool g_nrdRealtimeMode = false;
// The original noise-reduction scheme, predating TAA/NRD/ReSTIR DI: freeze the camera and keep
// running-averaging g_outputBuffer (g_temporalAccumulator, updateSceneBuffer()) until it converges,
// resetting to a fresh single sample the instant anything moves. TAA (g_taa, TaaResolve.h) + NRD
// Realtime + ReSTIR DI now cover ordinary interactive use without needing this at all -- kept here,
// off by default, as an explicit opt-in "converged reference" mode for comparison (e.g. checking
// whether ReSTIR/NRD's result is actually converging toward the same answer), not the silent
// fallback whenever NRD Realtime happens to be off. See the "Reference Accumulation" checkbox
// (Settings window) and updateSceneBuffer()'s own use of this flag.
bool g_referenceAccumulation = false;
#endif

struct ImEnable_RAII
{
	bool is_enabled;

	ImEnable_RAII(bool enable)
		: is_enabled(enable)
	{
		if (!is_enabled)
		{
			const float alpha_multiplier = 0.5f;

			float target_alpha = ImGui::GetStyle().DisabledAlpha * alpha_multiplier;
			ImGui::PushStyleVar(ImGuiStyleVar_Alpha, target_alpha);
			ImGui::BeginDisabled(true);
		}
	}

	~ImEnable_RAII()
	{
		if (!is_enabled)
		{
			ImGui::EndDisabled();
			ImGui::PopStyleVar();
		}
	}
};

// --- VK_ERROR_DEVICE_LOST diagnostics -------------------------------------------------------
// Called from VK_CHECK (vk_framework.h) specifically on VK_ERROR_DEVICE_LOST. Queries whatever
// extra diagnostics the driver offers -- see the comment where these extensions get enabled,
// above the VK_EXT_device_fault/VK_NV_device_diagnostic_checkpoints push_back calls -- and logs
// them before the assert fires. Both are optional; each checks its own availability.

// pushed each frame
std::vector<std::string> g_UIErrors;

static void LogDeviceFaultInfo(VkDevice device)
{
	if (!vkGetDeviceFaultInfoEXT)
	{
		LOGE("VK_EXT_device_fault not available (not supported/enabled on this driver)");
		return;
	}

	VkDeviceFaultCountsEXT counts{ VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT };
	VkResult r = vkGetDeviceFaultInfoEXT(device, &counts, nullptr);
	if (r != VK_SUCCESS && r != VK_INCOMPLETE)
	{
		LOGE("vkGetDeviceFaultInfoEXT (counts) failed: %s", string_VkResult(r));
		return;
	}

	std::vector<VkDeviceFaultAddressInfoEXT> addressInfos(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT> vendorInfos(counts.vendorInfoCount);
	std::vector<uint8> vendorBinary(counts.vendorBinarySize);

	VkDeviceFaultInfoEXT info{ VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT };
	info.pAddressInfos = addressInfos.empty() ? nullptr : addressInfos.data();
	info.pVendorInfos = vendorInfos.empty() ? nullptr : vendorInfos.data();
	info.pVendorBinaryData = vendorBinary.empty() ? nullptr : vendorBinary.data();

	r = vkGetDeviceFaultInfoEXT(device, &counts, &info);
	if (r != VK_SUCCESS)
	{
		LOGE("vkGetDeviceFaultInfoEXT (info) failed: %s", string_VkResult(r));
		return;
	}

	LOGE("=== VK_EXT_device_fault ===");
	LOGE("description: %s", info.description);
	// Cross-reference reportedAddress against buffer addresses logged elsewhere (sceneInfoAddr,
	// indexAddr, vertexAddr, sbtAddr, output buffer, etc.) to identify which resource was involved.
	for (const auto& a : addressInfos)
		LOGE("  address fault: type=%d address=0x%llx precision=+-0x%llx",
			(int)a.addressType, (unsigned long long)a.reportedAddress, (unsigned long long)a.addressPrecision);
	for (const auto& v : vendorInfos)
		LOGE("  vendor info: %s code=%lld data=%lld", v.description, (long long)v.vendorFaultCode, (long long)v.vendorFaultData);
}

void setCurrentCamera()
{
	g_currentCamera = CEntityHandle();
	for (uint32 i = 0, count = g_scene.m_sceneObjects.GetElementCount(); i < count; ++i)
	{
		if (g_scene.m_sceneObjects.GetElementByIterator(i).hndCamera.IsValid())
		{
			g_currentCamera = g_scene.m_sceneObjects.GetHandleByIterator(i);
			break;
		}
	}
}

// Loads g_hndDefaultNormalMap once, lazily, the first time any setup path needs it -- same
// "load once, guarded by isValid()" idiom as g_ibl.load()'s own call sites (setupScene1(),
// setupScene2(), loadScene()), for the same reason: App::init() itself doesn't build a
// scene anymore (see setupScene1()'s own comment just below), and this is genuinely needed by
// every one of those paths (every material's own MaterialData::normalMapTextureId falls back to
// this texture's heap index -- see its own fill site in recordGraphicsCommands()), not just one.
void EnsureDefaultNormalMap(VulkanContext& vulkanContext)
{
	// isLoaded() (not just GetElementPtr() succeeding) is the right check: loadScene()'s
	// resetResources()/deinit() sweep destroys this texture's image like every other texture's,
	// but being an engine texture it's never removed from g_TextureManager -- so the manager slot
	// survives a reload with a live handle pointing at a Texture whose deinit() already reset it
	// back to an unloaded state (see Texture::deinit()/isLoaded()).
	Texture* existing = g_TextureManager.GetElementPtr(g_hndDefaultNormalMap);
	if (existing && existing->isLoaded())
		return;

	// See g_hndDefaultNormalMap's own comment -- flat (0,0,1) tangent-space normal, encoded the
	// standard (n*0.5+0.5) way: byte 128 = 0.5019.. ~ 0.0, byte 255 = 1.0. 1x1 is enough (it's a
	// constant, not real per-pixel data -- no mip chain beyond its own single level either).
	const uint32 kFlatNormal = 0xFFFF8080u; // little-endian RGBA8: R=80 G=80 B=FF A=FF
	Texture tex0;
	tex0.m_isSRGB = false; // direction data, not color -- see Texture.h's m_isSRGB comment
	tex0.loadFromMemory(vulkanContext, &kFlatNormal, 1, 1, "Default Normal Map (engine)");
	tex0.isEngineTexture = true;

	// Reuse the existing manager slot/handle if one survived a previous load -- pushing a new
	// element instead would leak that old (unloaded but never-removed) slot forever and leave
	// every already-resolved MaterialData::normalMapTextureId referencing this handle's original
	// heap index stale anyway.
	if (existing)
		*existing = tex0;
	else
		g_hndDefaultNormalMap = g_TextureManager.push_back(tex0);
}

// A brand new scene (setup #1/#2/load) shares nothing with whatever TAA/ReSTIR GI history buffer
// still holds from the PREVIOUS scene -- resize is the only other place this gets invalidated
// (onViewportSizeChange), so without this, switching scenes left taa_resolve.slang blending fresh
// frames against a stale previous scene's image at g_taa.historyWeight (0.9 by default), which
// stays visually dominant for many frames (confirmed live as setup #2's magenta BSP light markers
// bleeding into setup #1 as an all-over magenta tint/"discolored material" until it slowly faded
// out), and ReSTIR GI's own reservoir similarly blending stale, unrelated-scene radiance into fresh
// pixels. Originally only triggered from drawFrame()'s "resetTemporal" UI-button bookkeeping, which
// only a human clicking Setup#1/#2/Load ever went through -- calling setupScene1()/setupScene2()/
// loadScene() directly (as tests::RunSelfTest's scripted --selftest mode does) skipped it entirely,
// confirmed live as --selftest's own captured screenshots failing to match their own just-created
// gold images purely from this leftover history, not any real rendering difference. Called from
// those three functions directly now, not just drawFrame()'s UI layer, so ANY caller gets a clean
// scene by construction instead of needing to remember this side effect itself.
static void ResetTemporalHistory()
{
	g_temporalAccumulator = 0;
	g_taa.historyValid = false;
	g_giHistoryValid = false;
}

// Populates the scene with the same hand-authored content init() used to load unconditionally
// behind LOAD_BSP_ON_START/LOAD_GLTF_ON_START -- now a runtime action (the "setup #1" button)
// instead of a compile-time choice, so it can run after init() has already built an empty scene.
void setupScene1()
{
	ResetTemporalHistory();
	g_scene.m_sceneObjects.Clear();
//	g_MaterialManager.Clear();
	// Restored in case a previous setupScene3() zeroed it -- see its own comment.
	g_laserIntensity = 15.4f;

	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);

	// Created up front (materials don't need a command buffer) since g_plane.createPlane() below
	// needs a material handle -- every mesh always carries one now, see Mesh::m_primitiveMaterials.
	CMatHandle hndM0;
	{
		Material m0;
		m0.m_name = "ground";
		m0.m_albedoColor = float3(0.95f, 0.8f, 0.5f);
		m0.m_roughness = 0.35f;
		m0.m_metallic = 1.0f;
		m0.m_albedoTextureFade = 0.5f; // less texture
		hndM0 = g_MaterialManager.push_back(m0);
	}
	CMatHandle hndM1;
	{
		Material m1;
		m1.m_name = "couch";
		m1.m_albedoColor = float3(0.3f, 0.9f, 0.9f);
		m1.m_roughness = 0.8f;
		m1.m_metallic = 0.0f;
		m1.m_albedoTextureFade = 0.9f; // less texture
		hndM1 = g_MaterialManager.push_back(m1);
	}

	// Load the two hand-authored test images first, so they claim descriptor-heap slots
	// 0/1 -- matching the "Image 1"/"Image 2" debug radio buttons (m_imageID -> scene.texId,
	// see the Settings window) -- regardless of how many textures a loaded glTF file
	// registers afterward (see Texture::getHeapIndex). Resources are always resolved
	// relative to the repo root -- Visual Studio's default debugging working directory is
	// $(ProjectDir) (this project has no <LocalDebuggerWorkingDirectory> override, i.e. thed
	// repo root), and running from the command line is expected to match (cd to the repo
	// root first). Texture::loadFromFile already asserts clearly if a path doesn't resolve.
	CTexHandle hndTex;
	{
		Texture tex0;
		tex0.loadFromFile(vulkanContext, "resources/image1.jpg");
		hndTex = g_TextureManager.push_back(tex0);

		Texture tex1;
		tex1.loadFromFile(vulkanContext, "resources/image2.jpg");
		g_TextureManager.push_back(tex1);
	}

	EnsureDefaultNormalMap(vulkanContext);

	// emplace_back(), not push_back(MeshComponent(...)) -- see CObjManager::emplace_back's comment.
	g_mesh = g_MeshManager.emplace_back("resources/SheenWoodLeatherSofa.glb");
	g_MeshManager.GetElementPtr(g_mesh)->load(vulkanContext, glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), float3(1, 0, 0)));

	// emplace_back(), not push_back(MeshComponent(...)) -- see CObjManager::emplace_back's comment.
	g_plane = g_MeshManager.emplace_back("plane");
	g_MeshManager.GetElementPtr(g_plane)->createPlane(vulkanContext, 10, hndM0);
	g_MaterialManager.GetElementPtr(hndM0)->m_hndTexture = hndTex;

	// Image-based lighting: load once, lazily, the first time any setup path needs it -- see its
	// own call site comment in setupScene2()/loadScene() for why this can't just happen once in
	// App::init() instead.
	if (!g_ibl.isValid())
		g_ibl.load(vulkanContext, "resources/RGB16f/pisa_CM.dds", "resources/RGB16f/pisa_DIFFCM.dds");

	// createDescriptorHeap() (called once from init()) also builds the one-time sampler heap;
	// only the resource heap needs rebuilding here, to account for the textures just added --
	// same reasoning as loadScene()'s equivalent rebuild after reloading textures.
	g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	// Staging buffers created above are already sitting in g_renderer.m_allocator, no longer freed
	// here directly -- the main loop's run() now drains them into g_deferredRelease every frame
	// (see that call site's comment), the same frame-safe deferred destruction every other Buffer
	// already gets, instead of each caller managing its own staging buffers' lifetime by hand.

	// camera
	{
		CameraComponent camera;
		Entity s;
		s.hndCamera = g_CameraManager.push_back(camera);
		s.transform.SetPos(float3(0, 2, 8));
		g_scene.m_sceneObjects.push_back(s);
	}

	// light
	{
		LightComponent light;
		Entity s;
		s.hndLight = g_LightManager.push_back(light);
		s.transform.SetPos(float3(-0.7f, 2.6f, 0.1f));
		g_scene.m_sceneObjects.push_back(s);
	}

	// test scene e.g. couches
	{
		Entity s;
		s.hndRenderComp = g_plane;
		s.transform.SetPos(float3(0, 0, 0));
		s.transform.SetScale(float3(10, 10, 10));
		g_scene.m_sceneObjects.push_back(s);
		s.transform.SetScale(float3(1, 1, 1));

		s.hndRenderComp = g_mesh;
		s.transform.SetPos(float3(0, 1, 0));
		s.transform.SetRotation(float2(0.2f, 0.0f));
		g_scene.m_sceneObjects.push_back(s);

		int objCount = 8;
		for (int i = 0; i < objCount - 1; ++i)
		{
			float w = i * PI * 2 / objCount;
			s.transform.SetPos(float3(sin(w), 0, cos(w)) * 3.0f);
			s.transform.SetRotation(float2(w + 0.1f, 0.0f));
			g_scene.m_sceneObjects.push_back(s);
		}
	}

	setCurrentCamera();
}


void setupScene2()
{
	ResetTemporalHistory();
	g_scene.m_sceneObjects.Clear();
//	g_MaterialManager.Clear();
	// Restored in case a previous setupScene3() zeroed it -- see its own comment.
	g_laserIntensity = 15.4f;

	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);

	// emplace_back(), not push_back(MeshComponent(...)) -- see CObjManager::emplace_back's comment:
	// copying a MeshComponent through a temporary silently drops its Buffers off g_EventManager's
	// registration list, leaking their GPU memory at shutdown.
	g_lightMesh = g_MeshManager.emplace_back("resources/smoothBox.glb");
	g_MeshManager.GetElementPtr(g_lightMesh)->load(vulkanContext, glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), float3(1, 0, 0)));

	//	g_BSPLevel.Load(vulkanContext, "resources/q3dm3.bsp");	// smaller
	g_BSPLevel.Load(vulkanContext, "resources/ospdm2.bsp"); // larger works well batcula, very cyan https://ws.q3df.org/map/ospdm2/
	//	g_BSPLevel.Load(vulkanContext, "resources/platypus.bsp"); // textures are broken
	{
		Entity s;
		s.hndRenderComp = g_BSPLevel.getHandle();
		s.transform.SetPos(float3(0, 7, 20));
		s.transform.SetScale(float3(-1, 1, 1) * 0.01f);
		s.transform.SetRotation(float2(0.0f, -3.14159265f / 2));	// todo: verify
		g_scene.m_sceneObjects.push_back(s);
	}

/*
	// Visual markers only -- NOT LightComponent/hndLight entities. updateLights() (further down this
	// file) already turns every g_BSPLevel light into a shaderio::Light directly from g_BSPLevel
	// itself (with the correct per-light attenuation radius from the map file); if these markers also
	// carried hndLight, updateLights()'s separate "any entity with hndLight" loop would pick them up
	// too and count every BSP light twice (confirmed live: 141 BSP lights became 283 g_lights,
	// tripping the kMaxLights=256 truncation warning) -- and with a flat, hardcoded 50.0 radius
	// instead of the map's real per-light falloff, to boot.
	const glm::mat4 bspObjectMatrix = g_BSPTransform.GetObjectMatrix();
	const float bspScale = glm::abs(g_BSPTransform.GetScale().x);
	for (uint32 i = 0; i < g_BSPLevel.GetLightCount(); ++i)
	{
		const SBSPLight* bspLight = g_BSPLevel.GetLightNo(i);

		Entity s;
		s.hndRenderComp = g_lightMesh;
		// bspLight->m_vOrigin is in the level file's own local space -- needs the same object matrix
		// updateLights() applies to place it where the (rotated, 0.01-scaled) level actually ended up
		// in the world, or these markers land far outside the visible level.
		s.transform.SetPos(float3(bspObjectMatrix * float4(bspLight->m_vOrigin, 1.0f)));
		s.transform.SetScale(float3(1, 1, 1) * bspLight->m_fSourceRadius * bspScale);
		g_scene.m_sceneObjects.push_back(s);
	}
*/

	// outside light
	{
		LightComponent light;
		Entity s;
		s.hndLight = g_LightManager.push_back(light);
		s.hndRenderComp = g_lightMesh;
		s.transform.SetPos(float3(-0.7f, 2.6f, 0.1f));
		s.transform.SetScale(float3(1, 1, 1) * 0.1f);
		g_scene.m_sceneObjects.push_back(s);
	}

	// Image-based lighting: loaded lazily here (and in setupScene1()/loadScene(), each guarded the
	// same way) rather than once in App::init(), because doing it there registered the
	// specular/diffuse cubemaps into g_renderer.m_descriptorHeap at indices 0/1, ahead of whatever
	// real 2D texture a scene loads first -- Shader::updateRayDescriptorSet()'s texture array relies
	// on heap index 0 always being *some* real Texture2D (it's used as the fallback for every
	// unfilled array slot), and g_renderer.m_rtTextureViews[0] never gets written at all once IBL's
	// cubemap sits there instead. That silently starved updateRayDescriptorSet()'s own guard
	// (textureViews[0] == VK_NULL_HANDLE) into never writing g_tlas either, confirmed live as
	// vkCmdTraceRaysKHR's VUID-vkCmdTraceRaysKHR-None-08114 ("g_tlas'... never been updated") on
	// literally every frame after clicking either setup button. Loading IBL here instead, after
	// each setup's own real textures/BSP-embedded textures have already claimed the low indices,
	// preserves that invariant.
	if (!g_ibl.isValid())
		g_ibl.load(vulkanContext, "resources/RGB16f/pisa_CM.dds", "resources/RGB16f/pisa_DIFFCM.dds");
	EnsureDefaultNormalMap(vulkanContext);

	// createDescriptorHeap() (called once from init()) also builds the one-time sampler heap;
	// only the resource heap needs rebuilding here, to account for the textures just added --
	// same reasoning as loadScene()'s equivalent rebuild after reloading textures.
	g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	// Staging buffers created above are already sitting in g_renderer.m_allocator, no longer freed
	// here directly -- the main loop's run() now drains them into g_deferredRelease every frame
	// (see that call site's comment), the same frame-safe deferred destruction every other Buffer
	// already gets, instead of each caller managing its own staging buffers' lifetime by hand.

	// camera
	{
		CameraComponent camera;
		camera.fieldOfView = 55.0f;
		Entity s;
		s.hndCamera = g_CameraManager.push_back(camera);
		s.transform.SetPos(float3(0, 2, 8));
		// Default rotation (0,0) looks toward -Z (Transform::GetForward()), but the BSP entity
		// above sits at Z=+20 -- i.e. directly behind this camera's default facing. Yaw by PI to
		// face +Z instead, or the level is never in view (this was previously left unset, making
		// setup #2 look at empty space no matter which renderer -- easy to mistake for "ray tracing
		// shows nothing" when it's really "nothing is in front of the camera at all").
		s.transform.SetRotation(float2(PI, 0.0f));
		g_scene.m_sceneObjects.push_back(s);
	}

	setCurrentCamera();
}

// SEED (Electronic Arts)'s "Pica Pica Assets" (https://github.com/SEED-EA/pica-pica-assets),
// CC BY-NC 4.0 -- see resources/pica-pica-assets/LICENSE_AND_ATTRIBUTION.md for the full license
// text and attribution. The source .fbx was converted to this .glb via Blender (--background,
// bpy.ops.import_scene.fbx + bpy.ops.export_scene.gltf); no geometry/material/texture changes
// otherwise. Only this one of the nine vendored pica-pica-* asset folders is wired in here -- the
// rest are kept under resources/pica-pica-assets/ unconverted, for potential future use.
void setupScene3()
{
	ResetTemporalHistory();
	g_scene.m_sceneObjects.Clear();
	// g_laser's own fixed-position red light (ray.slang: "outLightColor = float4(15, 0.2, 0.1,
	// 0.4) * scene.laserIntensity") is tuned for setup1/2's larger framing -- at this much smaller,
	// close-in diorama scale it dominates the lighting with an unwanted red tint and isn't part of
	// this scene's own content anyway. setupScene1()/setupScene2() restore the default.
	g_laserIntensity = 0.0f;
	// Whatever atmospheric fog density/asymmetry Settings.ini happens to have persisted from a
	// previous interactive session (confirmed live: FogDensity=0.06 with FogAsymmetry=-0.99, an
	// extreme near-singular backscatter value) is NOT part of this scene's own content and isn't
	// reset by anything else -- tinted the open sky visible past the grid's edges. No fog wanted
	// for this scene at all. (Turned out NOT to be the ground plane's own ongoing magenta tint --
	// see the "ground" material's own comment below for that one's real, separate cause.)
	g_fogDensity = 0.0f;

	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);

	CMatHandle hndM0;
	{
		Material m0;
		m0.m_name = "ground";
		m0.m_albedoColor = float3(0.5f, 0.5f, 0.5f);
		// Without this, ray.slang's albedo sampling (its own "the material's own glTF texture if it
		// has one, otherwise the debug Image 1/Image 2 toggle" comment) kicks in whenever
		// albedoTextureFade > 0 -- its own default is 1.0, not 0 -- regardless of whether a real
		// texture (m_hndTexture) was ever assigned. With none assigned here, that fell through to
		// sampling the debug test image at full strength over this material's own albedo color,
		// which is what the persistent magenta/purple tint actually was -- entirely independent of
		// lighting (confirmed live: identical even with the scene's own light at 0 intensity).
		m0.m_albedoTextureFade = 0.0f;
		m0.m_roughness = 0.8f;
		m0.m_metallic = 0.0f;
		hndM0 = g_MaterialManager.push_back(m0);
	}

	EnsureDefaultNormalMap(vulkanContext);

	// emplace_back(), not push_back(MeshComponent(...)) -- see CObjManager::emplace_back's comment.
	g_mesh = g_MeshManager.emplace_back("resources/pica-pica-mini-diorama-01.glb");
	g_MeshManager.GetElementPtr(g_mesh)->load(vulkanContext);

	g_plane = g_MeshManager.emplace_back("plane");
	g_MeshManager.GetElementPtr(g_plane)->createPlane(vulkanContext, 10, hndM0);

	if (!g_ibl.isValid())
		g_ibl.load(vulkanContext, "resources/RGB16f/pisa_CM.dds", "resources/RGB16f/pisa_DIFFCM.dds");

	g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);

	// Diorama scaled 3x (per a live request -- its real-world size is ~0.6 x 0.5 x 0.2m, confirmed
	// via Blender's own bound_box on the converted .glb, which read as too small/dollhouse-like).
	const float kDioramaScale = 3.0f;

	// 10x10 grid of the same diorama (per a live request) -- every cell reuses the same g_mesh
	// handle (just a different Entity/transform per cell), so this is 100 TLAS instances of one
	// shared BLAS, not 100x the geometry/texture memory.
	const int kGridSize = 10;
	const float kGridSpacing = 2.0f; // world units between cell centers, > kDioramaScale's ~1.9m footprint
	const float kGridExtent = (kGridSize - 1) * kGridSpacing; // 31.5 -- for camera/ground framing below
	const float kGridOffset = kGridExtent * 0.5f; // centers the whole grid on the origin

	// camera -- pulled back and elevated for an overview of the whole grid, angled down toward its
	// center (see Transform::Rotate's own comment: positive pitch looks up, so negative looks down).
	{
		CameraComponent camera;
		Entity s;
		s.hndCamera = g_CameraManager.push_back(camera);
		s.transform.SetPos(float3(0, kGridExtent * 0.6f, kGridExtent * 0.9f));
		s.transform.SetRotation(float2(0.0f, -0.55f));
		g_scene.m_sceneObjects.push_back(s);
	}

	// light -- centered and raised well above the whole grid, not just one diorama, with intensity
	// bumped up to compensate for the much greater distance (inverse-square falloff).
	{
		LightComponent light;
		light.m_lightIntensity = 40.0f;
		Entity s;
		s.hndLight = g_LightManager.push_back(light);
		s.transform.SetPos(float3(0, kGridExtent * 0.5f, 0));
		g_scene.m_sceneObjects.push_back(s);
	}

	// ground
	{
		Entity s;
		s.hndRenderComp = g_plane;
		// Slightly below the dioramas' own floor (the .glb's bounds start at y=-0.0117 pre-scale,
		// confirmed via Blender's own bound_box) so the ground plane doesn't poke through/z-fight.
		s.transform.SetPos(float3(0, -0.02f, 0));
		// createPlane()'s halfSize argument (App.cpp, 10 below) makes its base size 20x20 world
		// units -- scaled here to comfortably cover the kGridExtent x kGridExtent grid footprint.
		s.transform.SetScale(float3(1, 1, 1) * (kGridExtent / 20.0f + 0.5f));
		g_scene.m_sceneObjects.push_back(s);
	}

	// grid of dioramas
	{
		for (int gz = 0; gz < kGridSize; ++gz)
		{
			for (int gx = 0; gx < kGridSize; ++gx)
			{
				Entity s;
				s.hndRenderComp = g_mesh;
				s.transform.SetPos(float3(gx * kGridSpacing - kGridOffset, 0, gz * kGridSpacing - kGridOffset));
				s.transform.SetScale(float3(kDioramaScale, kDioramaScale, kDioramaScale));
				g_scene.m_sceneObjects.push_back(s);
			}
		}
	}

	setCurrentCamera();
}


void saveScene()
{
	LOGI_AA_SAVE("saveScene '%s'", g_sceneFileName.c_str());

	TokenAppendArena arena;
	saveTextures(arena);
	saveMaterials(arena);
	saveMeshes(arena);
	saveLights(arena);
	saveCameras(arena);
	// Last: Entity::save() writes hndMesh/hndLight/hndCamera, which on load must resolve through
	// HandleRemapTable -- that only works if the referenced Mesh/Light/Camera's own ATokenCmdNew
	// already ran, i.e. was saved earlier in the file (see process(), AppendArena.cpp).
	g_scene.save(arena);

	FILE* out = 0;

	if (fopen_s(&out, g_sceneFileName.c_str(), "wb") != 0)
		ASSERT(0);
	if (out)
	{
		if (fwrite(arena.data(), sizeof(*arena.data()), arena.size(), out) != arena.size())
			ASSERT(0);
		fclose(out);
		out = 0;
	}
	if (APPEND_ARENA_LOG_SAVE)
		LOGI("");
}

void loadScene()
{
	ResetTemporalHistory();
	LOGI_AA_LOAD("loadScene '%s'", g_sceneFileName.c_str());

	// Clear() bumps every slot's Seed (see ObjManager.h), so a handle captured before Clear()
	// no longer matches after it -- and load()'s ATokenCmdSetMember commands still carry the
	// dstKey handles captured when scene.aa was saved (right after init(), before any Clear()
	// ever ran). Swapping in a brand new manager instead gives load() a virgin one (all Seeds
	// back at their initial value), matching what save() saw, so the saved handles resolve.
	g_scene.m_sceneObjects = CObjManager<CEntityHandle, Entity>();
	g_scene.m_sceneObjects.InitOnce(4096);

	g_MaterialManager = CObjManager<CMatHandle, Material>();
	g_MaterialManager.InitOnce(4096);

	// Same reasoning as g_MaterialManager above -- Light/Camera hold no GPU resources of their
	// own (unlike meshes/textures below), so a plain swap-in-fresh is enough.
	g_LightManager = CObjManager<CLightHandle, LightComponent>();
	g_LightManager.InitOnce(4096);

	g_CameraManager = CObjManager<CCameraHandle, CameraComponent>();
	g_CameraManager.InitOnce(4096);

	// Destroying GPU resources below (mesh BLAS handles, then textures) is only safe once the GPU
	// has actually finished with them -- they may be referenced by whichever frame(s) are still in
	// flight -- matching the same vkDeviceWaitIdle() the shutdown path (destroy()) already does
	// before its own equivalent cleanup loops.
	VkDevice device = g_renderer.m_context.getDevice();
	VK_CHECK(vkDeviceWaitIdle(device));

	// Without this, g_mesh/g_plane (pushed into g_MeshManager during init(), before any save/load
	// ever ran) are still in it when the "load meshes" loop below runs -- it would call load() on
	// g_plane too (a procedural mesh; see Mesh::m_loadFromFile's comment) alongside the freshly
	// deserialized meshes, since GetElementCount()/GetElementByIndex() would still enumerate them.
	// g_MeshManager now holds MeshComponent by value (not raw Mesh* -- see CObjManager::emplace_back's
	// comment), so replacing it below runs every still-alive mesh's real destructor -- which asserts
	// "forgot to call deinit()" (Buffer::~Buffer()) for each of its Buffer members unless deinit()
	// already ran, since deinit() is the only thing that actually releases the GPU allocation (see
	// DeferredRelease.h). MeshComponent::deinit() releases all of them, see its own comment.
	EventContext meshDeinitContext{ device, g_renderer.m_allocator };
	for (uint32 i = 0, count = g_MeshManager.GetElementCount(); i < count; ++i)
		g_MeshManager.GetElementByIterator(i).deinit(meshDeinitContext);
	g_MeshManager = CObjManager<CRenderCompHandle, RenderComponent>();
	g_MeshManager.InitOnce(4096);

	// Textures own real GPU resources (VkImage/VkImageView/VMA allocation), unlike Scene/Material --
	// discarding g_TextureManager without freeing those first leaks them, which VMA's shutdown-time
	// integrity check catches ("Some allocations were not freed before destruction of this memory
	// block!", vk_mem_alloc.h). resetResources() drops the descriptor heap's entries for those same
	// images, since they're about to be invalid and build() (called again below) would otherwise
	// create image views against freed images. Destroying them right away is only safe once the GPU
	// has actually finished with them (they're referenced by whichever frame(s) are still in
	// flight) -- matching the same vkDeviceWaitIdle() the shutdown path (destroy()) already does
	// before its own equivalent texture cleanup loop.
	//
	// Note: Texture doesn't inherit IAppInit yet (see Texture.h's "todo : public IAppInit" --
	// still commented out), so g_EventManager.callAppDeInit(EF_Type_Texture, ...) would not
	// actually reach any Texture; call deinit() on each one directly instead until that's wired up.
	for (uint32 i = 0, count = g_TextureManager.GetElementCount(); i < count; ++i)
		g_TextureManager.GetElementByIterator(i).deinit(device, g_renderer.m_allocator);

	// Texture::deinit() (just above) destroys each texture's m_rtView, but g_renderer.m_rtTextureViews
	// (Renderer.h) is a separate, fixed-size array of copies of those same handles (see
	// Texture::uploadPixels) -- nothing ever clears its slots when the texture that filled them is
	// destroyed. Left stale, updateRayDescriptorSet()'s "still valid?" check (!= VK_NULL_HANDLE) can't
	// tell a destroyed handle from a live one -- it isn't zero, just no longer valid -- so it gets
	// written into the descriptor set as-is, caught by the validation layer at vkUpdateDescriptorSets
	// ("Invalid VkImageView Object"). Clear them alongside resetResources() below, which drops the
	// descriptor heap's own (analogous) list of these same now-destroyed images for the same reason.
	for (VkImageView& view : g_renderer.m_rtTextureViews)
		view = VK_NULL_HANDLE;

	g_renderer.m_descriptorHeap.resetResources();

	// IBL's cubemaps (see IBLEnvironment.h) are registered in this same heap, at whatever indices
	// resetResources() above just dropped -- deinit() them too so g_ibl.isValid() correctly reports
	// false and the guarded g_ibl.load() call further down (this function's own equivalent of
	// setupScene1()/setupScene2()'s) re-registers fresh ones instead of sampling through stale,
	// now-meaningless heap indices.
	g_ibl.deinit(device, g_renderer.m_allocator);

//	g_TextureManager = CObjManager<CTexHandle, Texture>();
//	g_TextureManager.InitOnce(4096);
	// RemoveByIndex() swaps the last used element into the removed slot (see
	// CObjManager::SwapInUsedAndUsed) and shrinks UsedCount by one -- so after a removal, "it"
	// must be re-examined (it now holds the swapped-in element) and "count" must shrink with it,
	// or iteration eventually runs "it" past the real (smaller) UsedCount and GetHandleByIterator()
	// asserts (Iterator < UsedCount).
	for (uint32 it = 0, count = g_TextureManager.GetElementCount(); it < count; ++it)
	{
		CTexHandle hnd = g_TextureManager.GetHandleByIterator(it);
		auto& el = g_TextureManager.GetElementByIterator(it);
		if (!el.isEngineTexture)
		{
			g_TextureManager.RemoveByIndex(hnd.Index);
			--it;
			--count;
		}
	}
	
	TokenAppendArena arena;

	FILE* in = 0;

	extern size_t IO_GetFileSize(const char* Name);
	size_t size = IO_GetFileSize(g_sceneFileName.c_str());

	fopen_s(&in, g_sceneFileName.c_str(), "rb");

	if (in)
	{
		arena.resize(size / sizeof(*arena.data()));
		if (fread(arena.data(), sizeof(*arena.data()), arena.size(), in) != arena.size())
			ASSERT(0);
		fclose(in);
		in = 0;
	}

	{
		HandleRemapTable remapping;
		ArenaTokenRange range;

		range.arena = &arena;
		range.size = arena.size();

		process(range, remapping);

		// g_lightMesh (this file's own global, the small marker mesh dispatchRayTracing() falls back
		// to for a light entity with no hndRenderComp of its own -- see its own comment) is a cached
		// CRenderCompHandle set once by setupScene2(), not itself part of the saved scene data.
		// process() above just discarded and recreated every mesh under brand new handles (g_MeshManager
		// was fully reset above; saveMeshes() saves every mesh in it unconditionally, including this
		// one) -- left unrefreshed, this stale global still points at the pre-reload manager, and
		// dispatchRayTracing()'s "effectiveMesh.Index < g_MeshManager.GetElementCount()" assert fires
		// the instant a mesh-less light entity is rendered post-load. Route it through the same
		// remapping table every saved hndRenderComp reference already uses to survive the reload.
		if (g_lightMesh.IsValid())
		{
			CObjHandle newHnd = remapping.Get(3, g_lightMesh.Index);
			g_lightMesh.Seed = newHnd.Seed;
			g_lightMesh.Index = newHnd.Index;
		}
	}

	setCurrentCamera();

	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
		VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);

		// load textures
	//todo		g_EventManager.callAppInit(EF_Type_Texture, context);
		for (uint32 i = 0, count = g_TextureManager.GetElementCount(); i < count; ++i)
		{
			auto& el = g_TextureManager.GetElementByIterator(i);

			if(el.m_loadFromFile)
				el.loadFromFile(vulkanContext, el.m_name);
			// File-less texture (BSP lightmap, glTF-embedded image, ...) -- see
			// Texture::m_sourcePixels' own comment. Without this, it stays a registered-but-never-
			// uploaded Texture (no image, heap index 0 by default), which is exactly what silently
			// aliased whatever real texture happens to sit at heap slot 0 -- confirmed live as a
			// BSP level's lightmaps (and therefore most of its visible detail) going missing after
			// "setup #2" -> "save" -> "load".
			else if (!el.m_sourcePixels.empty())
				el.loadFromMemory(vulkanContext, el.m_sourcePixels.data(), el.m_sourceWidth, el.m_sourceHeight, "restored texture");
		}

		// load meshes
		for (uint32 i = 0, count = g_MeshManager.GetElementCount(); i < count; ++i)
		{
			auto& el = g_MeshManager.GetElementByIterator(i);

			if(!el.m_sourceIndices.empty())
			{
				el.createFromSource(vulkanContext);
				// createFromSource() only rebuilds geometry/BLAS -- m_subMeshes itself (the
				// submesh->material association) has to be rebuilt separately.
				//
				// m_subMeshes is already fully restored by now if the save file had memberIndex 4
				// data (see saveMeshes()/AppendArena.cpp's getSubMeshes()) -- covers a mesh with any
				// number of submeshes, e.g. a loaded BSP level's one-per-lightmap/texture-combo
				// assignment, which a single m_proceduralMaterial can't represent at all. Only fall
				// back to rebuildSingleSubMesh() (single submesh, m_proceduralMaterial) for an older
				// save file with no such data, matching this project's original, single-material-only
				// procedural mesh support.
				if (!el.m_subMeshes.empty())
					el.buildPrimitiveMaterials(vulkanContext, (uint32)(el.m_sourceIndices.size() / 3));
				else
					el.rebuildSingleSubMesh(vulkanContext);
				continue;
			}

			// saveMeshes() only ever serializes file-loaded meshes (see its m_loadFromFile check),
			// so m_filePath should always be real here -- this is just a defensive backstop against
			// a stale/hand-edited scene.aa, since load()'s tg3_parse_file() on an empty/bogus path
			// leaves the mesh with no geometry and Buffer::initRaw() asserts on the null data.
			if (!el.m_filePath.empty())
				el.load(vulkanContext);
		}

		// Image-based lighting: re-register (resetResources()/g_ibl.deinit() above dropped the old
		// registration) before every real texture above claims the low heap indices again -- see
		// setupScene2()'s equivalent call site for why this ordering (not once in
		// App::init()) matters.
		if (!g_ibl.isValid())
			g_ibl.load(vulkanContext, "resources/RGB16f/pisa_CM.dds", "resources/RGB16f/pisa_DIFFCM.dds");
		EnsureDefaultNormalMap(vulkanContext);

		// Texture::loadFromFile() (via uploadPixels) only registers each image with the descriptor
		// heap's in-memory list (DescriptorHeap::addTexture) -- the GPU-visible heap buffer itself is
		// a one-shot upload (build()) that init() already ran once. Without rebuilding it here, every
		// reloaded texture's descriptor slot is never actually written on the GPU, so shaders sample
		// unwritten data (observed as the texture rendering solid black after a load).
		g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

	if (APPEND_ARENA_LOG_LOAD)
		LOGI("");
}

static void LogCheckpointInfo()
{
	if (!vkGetQueueCheckpointDataNV)
	{
		LOGE("VK_NV_device_diagnostic_checkpoints not available (not supported/enabled on this driver)");
		return;
	}

	VkQueue queue = g_renderer.m_context.getGraphicsQueue().queue;

	uint32 count = 0;
	vkGetQueueCheckpointDataNV(queue, &count, nullptr);
	if (!count)
	{
		LOGE("No VK_NV_device_diagnostic_checkpoints data (queue never reached a checkpoint before dying, or already recovered)");
		return;
	}

	std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV{ VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV });
	vkGetQueueCheckpointDataNV(queue, &count, data.data());

	LOGE("=== VK_NV_device_diagnostic_checkpoints: last-reached marker(s) ===");
	for (const auto& cp : data)
		LOGE("  pipelineStage=%d marker=\"%s\"", (int)cp.stage, cp.pCheckpointMarker ? (const char*)cp.pCheckpointMarker : "(null)");
}

void OnVulkanDeviceLost()
{
	VkDevice device = g_renderer.m_context.getDevice();
	if (!device)
	{
		LOGE("OnVulkanDeviceLost: no device yet, nothing to query");
		return;
	}

	LogDeviceFaultInfo(device);
	LogCheckpointInfo();
}

// Drops a named VK_NV_device_diagnostic_checkpoints marker into cmd if the extension is available;
// no-op otherwise. marker must be a string literal / static storage -- the driver stores the raw
// pointer, it doesn't copy the string.
static void SetCheckpoint(VkCommandBuffer cmd, const char* marker)
{
	if (vkCmdSetCheckpointNV)
		vkCmdSetCheckpointNV(cmd, marker);
}

#pragma comment(lib, "winmm.lib")

/*--
 * Set every dynamic state required by shader objects before a draw.
 *
 * Without a graphics pipeline object, all of viewport/scissor, rasterization,
 * depth/stencil, blend, multisample, vertex input, and primitive topology are
 * dynamic. The list below covers what this sample uses; tessellation, mesh
 * shading, transform feedback, conservative rasterization, etc. would each
 * add their own vkCmdSet*EXT calls (and the matching device features).
 *
 * Doesn't touch any App state -- takes everything it needs as
 * parameters, so it's a free function rather than a member.
-*/
static void setGraphicsDynamicState(VkCommandBuffer cmd, const VkViewport& viewport, const VkRect2D& scissor)
{
	// Viewport / scissor (counts and values are both dynamic).
	vkCmdSetViewportWithCount(cmd, 1, &viewport);
	vkCmdSetScissorWithCount(cmd, 1, &scissor);

	// Vertex input: bindings (stride, input rate) and attributes (location, format, offset).
	// VK_EXT_vertex_input_dynamic_state replaces VkPipelineVertexInputStateCreateInfo.
	const VkVertexInputBindingDescription2EXT vertexBinding{
		.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		.divisor = 1,
	};
	const auto vertexAttributes = Vertex::getAttributeDescriptions();
	
	vkCmdSetVertexInputEXT(cmd, 1, &vertexBinding, uint32(vertexAttributes.size()), vertexAttributes.data());

	// Input assembly.
	vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);

	// Rasterization (most of these come from VK_EXT_extended_dynamic_state_3).
	vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
	vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
	vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
	vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
	vkCmdSetDepthClampEnableEXT(cmd, VK_FALSE);

	// Multisampling.
	vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
	const VkSampleMask sampleMask = 0xFFFFFFFF;
	vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
	vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
	// alphaToOne is required by the spec when its device feature is enabled and a
	// shader object is bound, even if we don't actually use it.
	vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);

	// Depth / stencil.
	vkCmdSetDepthTestEnable(cmd, VK_TRUE);
	vkCmdSetDepthWriteEnable(cmd, VK_TRUE);
	//vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_LESS_OR_EQUAL);
	vkCmdSetDepthCompareOp(cmd, VK_COMPARE_OP_GREATER_OR_EQUAL);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	vkCmdSetStencilTestEnable(cmd, VK_FALSE);

	// Color blend (for one color attachment). Match the previous pipeline's
	// alpha-blend setup; nothing varies between draws so we set it once.
	const VkBool32                blendEnable = VK_TRUE;
	const VkColorBlendEquationEXT blendEquation{
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.alphaBlendOp = VK_BLEND_OP_ADD,
	};
	const VkColorComponentFlags colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
	vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &blendEquation);
	vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorWriteMask);
	vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
}

// Minimal dynamic state for taa_resolve.slang's draw: no vertex buffer (its vertexMain takes no
// VertexInput at all), and critically no depth test/write -- setGraphicsDynamicState() above enables
// both, which would be invalid there (its pass has a real depth attachment; this one has none at
// all, see the vkCmdBeginRendering call site). Color blend state is skipped too, for the same reason
// (colorAttachmentCount is 0 here -- nothing to blend against).
static void setTaaResolveDynamicState(VkCommandBuffer cmd, const VkViewport& viewport, const VkRect2D& scissor)
{
	vkCmdSetViewportWithCount(cmd, 1, &viewport);
	vkCmdSetScissorWithCount(cmd, 1, &scissor);
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
	vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
	vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
	vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
	vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
	vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
	vkCmdSetDepthBiasEnable(cmd, VK_FALSE);
	vkCmdSetDepthClampEnableEXT(cmd, VK_FALSE);
	vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
	const VkSampleMask sampleMask = 0xFFFFFFFF;
	vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
	vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
	vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);
	vkCmdSetDepthTestEnable(cmd, VK_FALSE);
	vkCmdSetDepthWriteEnable(cmd, VK_FALSE);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	vkCmdSetStencilTestEnable(cmd, VK_FALSE);
}

/*--
 * End command buffer recording for the frame. Doesn't touch any App state -- a free
 * function rather than a member.
-*/
static void endCommandRecording(VkCommandBuffer cmd) { VK_CHECK(vkEndCommandBuffer(cmd)); }


/*------------------------------------------------------------------
## Main Rendering Loop (drawFrame())
1. Wait on the slot's timeline value, then acquire the next swapchain image
2. Update vertex positions via the compute shader (vkCmdPushConstants2 + BDA)
3. Record render commands:
   - Update the scene info buffer (vkCmdUpdateBuffer to a BDA-addressed buffer)
   - Bind sampler / resource descriptor heaps
   - Set every dynamic state required by shader objects once per frame
   - Bind (vertex, fragment) shader objects via vkCmdBindShadersEXT
   - Draw colored triangle, swap fragment shader, draw textured triangle
   - Render the offscreen RenderTarget into the swapchain image via ImGui
4. Submit and present (signals presentSemaphore[image], advances slot)

There are a lot of comments; to strip all the block documentation, replace
using this regular expression:
- Visual Studio: /\*--([\r\n]|.)*?-\*\/
- Others       : /\*--[\s\S]*?-\*\/

------------------------------------------------------------------*/

#include "vulkanBase.h"



//--- stb_image implementation (single-TU requirement) --------------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Screenshots (CaptureRenderTargetToPNG below) -- PNG specifically because it's lossless (no
// compression artifacts to confuse a later pixel-exact comparison, e.g. tests::RunSelfTest's own
// gold-image check), same reasoning stb_image_write's own header gives for preferring it over its
// JPEG writer.
#define STB_IMAGE_WRITE_IMPLEMENTATION
// stb_image_write.h's own PNG writer uses plain sprintf() internally (not this project's own code,
// nothing to fix on our side) -- MSVC's C4996 flags that as deprecated-in-favor-of-sprintf_s, and
// this project has no blanket _CRT_SECURE_NO_WARNINGS, so suppress it locally for just this vendored
// header instead of either editing third-party source or disabling the warning project-wide.
#pragma warning(push)
#pragma warning(disable: 4996)
#include "stb_image_write.h"
#pragma warning(pop)

// Captures renderTarget's current color image -- already the final, tonemapped/sRGB-encoded,
// ImGui-overlay-free frame (see this codebase's own extensive gamma/tonemap work) -- to a lossless
// PNG file, creating path's parent directory (e.g. "screenshots/") if it doesn't exist yet.
// Synchronous: submits its own single-time command buffer and blocks (endSingleTimeCommands' fence
// wait) until the GPU-side copy actually lands, so the file is complete and readable the moment this
// returns -- callers (the "Screenshot" UI button and App::selfTestTick()) never need to
// know this touched the GPU at all. VK_FORMAT_R8G8B8A8_UNORM only (asserted) -- the one format every
// RenderTarget in this codebase actually uses (m_renderTarget's own .color =
// {VK_FORMAT_R8G8B8A8_UNORM}), and the one stb_image_write's plain 4-bytes-per-pixel RGBA call below
// assumes. Defined here (not up with this file's other free functions) because it needs
// stbi_write_png, only declared once this file reaches STB_IMAGE_WRITE_IMPLEMENTATION above.
static bool CaptureRenderTargetToPNG(utils::RenderTarget& renderTarget, const std::string& path)
{
	ASSERT(renderTarget.getColorFormat() == VK_FORMAT_R8G8B8A8_UNORM);
	const VkExtent2D size = renderTarget.getSize();
	const VkDeviceSize bufferSize = VkDeviceSize(size.width) * size.height * 4;

	// One-off readback, not part of the steady-state per-frame render loop -- created and destroyed
	// right here instead of joining the persistent globals (g_ReadbackBuffer etc.) that exist
	// specifically because THEY are touched every frame.
	utils::Buffer staging = g_renderer.m_allocator.createBuffer(bufferSize,
		VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
		// Every RenderTarget color image in this codebase lives permanently in GENERAL layout (see
		// RenderTarget::create()'s own comment, extern/vk_framework.h) -- a valid copy-source layout,
		// no transition needed here.
		const VkBufferImageCopy region{
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageExtent = {size.width, size.height, 1},
		};
		vkCmdCopyImageToBuffer(cmd, renderTarget.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &region);
		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

	VmaAllocationInfo allocInfo{};
	vmaGetAllocationInfo(g_renderer.m_allocator, staging.allocation, &allocInfo);

	std::filesystem::path fsPath(path);
	if (fsPath.has_parent_path())
		std::filesystem::create_directories(fsPath.parent_path());

	// The render target's own alpha channel isn't display opacity (nothing composites this image
	// against anything) -- force it opaque so an image viewer doesn't show a spuriously transparent
	// PNG.
	std::vector<uint8_t> rgba((const uint8_t*)allocInfo.pMappedData, (const uint8_t*)allocInfo.pMappedData + bufferSize);
	for (size_t i = 3; i < rgba.size(); i += 4)
		rgba[i] = 255;

	const bool ok = stbi_write_png(path.c_str(), (int)size.width, (int)size.height, 4, rgba.data(), (int)size.width * 4) != 0;

	g_renderer.m_allocator.destroyBuffer(staging);
	return ok;
}

#define USE_SLANG 1

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <GLFW/glfw3native.h>

//--- Geometry -------------------------------------------------------------------------------------------------------------

// BGRA (Windows DIB byte order, read as a 32-bit int: 0xAARRGGBB) -> RGBA (0xAABBGGRR) for GLFWimage.
// Must preserve A and G untouched and only swap R/B -- the previous version OR'd the alpha byte
// into the green byte and always zeroed alpha, making the icon fully transparent wherever alpha
// compositing is actually respected (e.g. the Windows 11 taskbar), even though it looked fine in
// contexts that ignore/force-opaque the icon (the title bar).
unsigned int RGBSwizzle(unsigned int c) {
	return (c & 0xff00ff00) | ((c >> 16) & 0xff) | ((c & 0xff) << 16);
}



void Reflection(CIniArchive& archive, WindowPersist& ref, App& app)
{
	Reflection(archive, "Window_fullScreen", ref.fullscreen);
	Reflection(archive, "Window_maximized", ref.maximized);
	for (int i = 0; i < 4; ++i)
	{
		char str[256];
		sprintf_s(str, sizeof(str), "Window_rect%d", i);
		Reflection(archive, str, ref.data[i]);
	}
	Reflection(archive, "Window_maximized", ref.maximized);

	//

	Reflection(archive, "VSyncMode", (int&)app.m_vSyncMode);

	Reflection(archive, "FlatSkyColor", g_flatSkyColor);
	Reflection(archive, "SkyRGB", g_skyRGB);
	Reflection(archive, "SkyIntensity", g_skyIntensity);
	Reflection(archive, "ExposureLog", g_ExposureLog);
	Reflection(archive, "EnableTonemapper", g_enableTonemapper);
	Reflection(archive, "LightmapIntensity", g_lightmapIntensity);
	Reflection(archive, "FogColor", g_fogColor);
	Reflection(archive, "FogDensity", g_fogDensity);
	Reflection(archive, "FogAsymmetry", g_fogAsymmetry);
	Reflection(archive, "InfoHUD", g_InfoHUD);
	Reflection(archive, "DrawLights" , g_drawLights);
	Reflection(archive, "EnableLights", g_enableLights);
	Reflection(archive, "EnableTextures", g_enableTextures);
	Reflection(archive, "AntiAliasing", g_enableAA);
	Reflection(archive, "LaserPos", g_laserPos);
	Reflection(archive, "LaserAngle", g_laserAngle);
	Reflection(archive, "LaserIntensity", g_laserIntensity);
	// g_Camera's own Reflection() (see Reflection.cpp) covers FieldOfView along with position/rotation,
	// encoded as a single string under one key so multiple cameras wouldn't collide.
//	Reflection(archive, "Camera", g_Camera);

#if USE_RAY_TRACING
	Reflection(archive, "RayTracing", g_rayTracing);
	Reflection(archive, "PathTracingBounces", (int&)g_pathTracingBounces);
	Reflection(archive, "AnimateNoise", g_animateNoise);
	Reflection(archive, "NrdRealtimeMode", g_nrdRealtimeMode);
	Reflection(archive, "ReferenceAccumulation", g_referenceAccumulation);
	Reflection(archive, "NrdMaxBlurRadius", g_nrdDenoiser.maxBlurRadius);
	Reflection(archive, "NrdMinBlurRadius", g_nrdDenoiser.minBlurRadius);
	Reflection(archive, "NrdMaxAccumulatedFrameNum", (int&)g_nrdDenoiser.maxAccumulatedFrameNum);
	Reflection(archive, "EnableRestirGI", g_enableRestirGI);
	Reflection(archive, "GIDepthThreshold", g_giDepthThreshold);
	Reflection(archive, "GINormalThreshold", g_giNormalThreshold);
	Reflection(archive, "GIHistoryWeight", g_giHistoryWeight);
#endif
}

//--- App ------------------------------------------------------------------------------------------------------------
// See App.h for the class definition.

App::App(VkExtent2D size, bool hidden)
	: m_windowSize(size)
{
	// Windows 10 Creators update adds Per Monitor V2 DPI awareness context.
	// Using this awareness context allows the client area of the window
	// to achieve 100% scaling while still allowing non-client window content to
	// be rendered in a DPI sensitive fashion.
	SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

	CIniArchive iniArchive;
	iniArchive.Load("Settings.ini");
	Reflection(iniArchive, g_windowPersist, *this);

	// Vulkan Loader
	VK_CHECK(volkInitialize());
	// Create the GLFW window
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	// See this constructor's own @param hidden comment -- everything past this point (Vulkan
	// device, swapchain, render loop) is unaffected either way.
	if (hidden)
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	const char* windowTitle = "VVerse V 0.37";

	if (!g_windowPersist.IsValid())
	{
		g_windowPersist = {};
		g_windowPersist.data[0] = g_windowPersist.data[1] = 100;
		g_windowPersist.data[2] = 1280;
		g_windowPersist.data[3] = 768;
	}

	m_window = glfwCreateWindow(m_windowSize.width, m_windowSize.height, windowTitle, nullptr, nullptr);

	g_windowPersist.ApplyState(glfwGetWin32Window(m_window));

	// https://stackoverflow.com/questions/7375003/how-to-convert-hicon-to-hbitmap-in-vc
#if _WIN32
	{
		//      HICON hIcon = (HICON)LoadImage(GetModuleHandle(0), L"icon.ico", IMAGE_ICON, 0, 0, 0);
		HICON hIcon = (HICON)LoadImage(GetModuleHandle(0), MAKEINTRESOURCE(IDI_SMALL), IMAGE_ICON, 0, 0, 0);
		//        ASSERT(hIcon);

		if (hIcon)
		{
			HBITMAP hBITMAPcopy;
			ICONINFOEX IconInfo;
			BITMAP BM_32_bit_color;

			memset((void*)&IconInfo, 0, sizeof(ICONINFOEX));
			IconInfo.cbSize = sizeof(ICONINFOEX);
			GetIconInfoEx(hIcon, &IconInfo);

			hBITMAPcopy = (HBITMAP)CopyImage(IconInfo.hbmColor, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
			GetObject(hBITMAPcopy, sizeof(BITMAP), &BM_32_bit_color);
			//Now: BM_32_bit_color.bmBits pointing to BGRA data.(.bmWidth * .bmHeight * (.bmBitsPixel/8))

			//    BITMAP BM_1_bit_mask;
			//HBITMAP IconInfo.hbmMask is 1bit per pxl
			// From HBITMAP to BITMAP for mask
		//    hBITMAPcopy = (HBITMAP)CopyImage(IconInfo.hbmMask, IMAGE_BITMAP, 0, 0, LR_CREATEDIBSECTION);
		//    GetObject(hBITMAPcopy, sizeof(BITMAP), &BM_1_bit_mask);
			//Now: BM_1_bit_mask.bmBits pointing to mask data (.bmWidth * .bmHeight Bits!)

			ASSERT(BM_32_bit_color.bmBitsPixel == 32);

			GLFWimage images[1];
			images[0].width = BM_32_bit_color.bmWidth;
			images[0].height = BM_32_bit_color.bmHeight;

			std::vector<int> mem;
			mem.resize(images[0].width * images[0].height);
			int* src = (int*)BM_32_bit_color.bmBits;
			for (int y = images[0].height - 1; y >= 0; --y) {
				for (int x = 0; x < images[0].width; ++x) {
					// seems glfwSetWindowIcon() doesn't support alpha
					mem[y * images[0].width + x] = RGBSwizzle(*src++);
				}
			}
			images[0].pixels = (unsigned char*)mem.data();
			glfwSetWindowIcon(m_window, 1, images);
			glfwPollEvents();                     // Force icon to be immediately shown

			// glfwSetWindowIcon() only sets the per-window icon (WM_SETICON), which is what the
			// title bar reads. The taskbar button reads GLFW's internal window CLASS icon instead,
			// which is left at its default -- set it too so the taskbar matches the title bar.
			HWND hwnd = glfwGetWin32Window(m_window);
			SetClassLongPtr(hwnd, GCLP_HICON, (LONG_PTR)hIcon);
			SetClassLongPtr(hwnd, GCLP_HICONSM, (LONG_PTR)hIcon);
		}
	}
#endif


	g_MouseInput.Create(glfwGetWin32Window(m_window));
	init();
}

App::~App()
{
	destroy();

	// doing it later will not work, after this message the hwnd is invalid
	g_windowPersist.SaveState(glfwGetWin32Window(m_window));
	{
		CIniArchive iniArchive;
		Reflection(iniArchive, g_windowPersist, *this);
		iniArchive.Save("Settings.ini");
	}

	glfwDestroyWindow(m_window);
}

//--- Self-test (VVerse.exe --selftest) -----------------------------------------------------------
// See App.h's own runSelfTest()/selfTestTick() comments for the big picture. Image
// comparison follows EA SEED's Gigi (https://github.com/electronicarts/gigi,
// Techniques/UnitTests/TestLogic.py): exact pixel equality against a checked-in "gold" image, no
// tolerance/threshold, and a MISSING gold image is bootstrapped from the current capture instead of
// failing -- review the newly-created TestData/gold_*.png by hand once (it's what future runs will
// be held to), same as any other test fixture. This only works at all because the render is made as
// deterministic as this codebase reasonably allows first (see runSelfTest()); GPU/driver-level
// float nondeterminism is a known, accepted limitation of this exact-match approach (Gigi's own
// docs note the same tradeoff), not something this code tries to paper over with a fuzzy diff.
namespace
{
	struct SelfTestStep
	{
		enum class Kind { Setup1, Setup2, Setup3, Save, Load, Capture, SetRayTracing, Finish };
		Kind kind;
		const char* arg = nullptr; // Save/Load: g_sceneFileName to use; Capture: gold image name
		bool rayTracingOn = false; // SetRayTracing only
	};

	// The actual test: setup1 -> capture -> save -> setup2 -> capture -> save -> load each saved
	// file back and capture again (round-trip must reproduce the same gold image) -- then repeats
	// setup1 and the first load a SECOND time (catches state that only works once: a manager that's
	// cleared on the first call but not the second, an engine resource created lazily-but-not-
	// idempotently, etc.), matching what was asked for load/save/"using those features a second
	// time" -- then switches to raster (non-ray-tracing) mode and captures both scenes again, since
	// that's a separate shader (shader.rast.slang vs ray.slang) the steps above never exercise.
	// Kept as one flat, linear script instead of a more general framework -- there's only ever one
	// of these, and a human should be able to read the whole test top to bottom.
	const SelfTestStep kSelfTestScript[] = {
		{ SelfTestStep::Kind::Setup1 },
		{ SelfTestStep::Kind::Capture, "setup1" },
		{ SelfTestStep::Kind::Save, "TestData/test_setup1.aa" },
		{ SelfTestStep::Kind::Setup2 },
		{ SelfTestStep::Kind::Capture, "setup2" },
		{ SelfTestStep::Kind::Save, "TestData/test_setup2.aa" },
		{ SelfTestStep::Kind::Load, "TestData/test_setup1.aa" },
		{ SelfTestStep::Kind::Capture, "setup1" },
		{ SelfTestStep::Kind::Load, "TestData/test_setup2.aa" },
		{ SelfTestStep::Kind::Capture, "setup2" },
		{ SelfTestStep::Kind::Setup1 },
		{ SelfTestStep::Kind::Capture, "setup1" },
		{ SelfTestStep::Kind::Load, "TestData/test_setup1.aa" },
		{ SelfTestStep::Kind::Capture, "setup1" },
		// Raster (non-ray-tracing) pass: same two scenes, rasterizer path only. shader.rast.slang
		// is a completely separate shader from ray.slang -- the ray-traced captures above never
		// touch it, so they can't catch a raster-only regression. Confirmed live as exactly that:
		// shader.rast.slang's real per-light shading loop was left disabled from an earlier,
		// unrelated fix, rendering every non-lightmapped (setup1) material with no lighting at all
		// in raster mode while ray tracing looked completely correct.
		{ SelfTestStep::Kind::SetRayTracing, nullptr, false },
		{ SelfTestStep::Kind::Setup1 },
		{ SelfTestStep::Kind::Capture, "setup1_raster" },
		{ SelfTestStep::Kind::Setup2 },
		{ SelfTestStep::Kind::Capture, "setup2_raster" },
		{ SelfTestStep::Kind::SetRayTracing, nullptr, true },
		// setup3: SEED Pica Pica mini diorama (resources/pica-pica-mini-diorama-01.glb) -- a loaded
		// glTF asset with many submeshes/materials, a different coverage shape than setup1's hand-
		// authored couches or setup2's BSP level.
		{ SelfTestStep::Kind::Setup3 },
		{ SelfTestStep::Kind::Capture, "setup3" },
		{ SelfTestStep::Kind::Finish },
	};

	// Frames rendered (and discarded) before each Capture step actually screenshots -- lets the
	// G-buffer/motion-vector history (one frame behind, see ray.slang's own motion-vector comment)
	// settle after a scene change. Small and fixed on purpose, not "rendered until converged":
	// determinism (a fixed frame count reproduces exactly) matters more here than image quality --
	// see runSelfTest()'s determinism forcing (TAA/NRD/Reference Accumulation/animated noise all
	// disabled) for why the very first settled frame is already the final frame, not an early one
	// out of many that would otherwise still be converging.
	const int kSelfTestSettleFrames = 2;

	struct SelfTestGlobalState
	{
		bool active = false;
		tests::TestReport report;
		int step = 0;
		int framesLeft = kSelfTestSettleFrames;
	};
	SelfTestGlobalState g_selfTest;

	// See this whole block's own header comment for the Gigi-style exact-match/create-if-missing
	// approach this implements.
	void CompareOrCreateGoldImage(tests::TestReport& report, const std::string& capturedPath, const std::string& goldName)
	{
		const std::string goldPath = "TestData/gold_" + goldName + ".png";

		if (!std::filesystem::exists(goldPath))
		{
			std::filesystem::create_directories("TestData");
			std::error_code ec;
			std::filesystem::copy_file(capturedPath, goldPath, ec);
			LOGI("SelfTest: no gold image yet -- created %s from this run. Review it by hand, then "
				"re-run --selftest to actually check future runs against it.", goldPath.c_str());
			return; // nothing to compare against yet -- not a failure, matches Gigi's own "creating" case
		}

		int capW = 0, capH = 0, capN = 0, goldW = 0, goldH = 0, goldN = 0;
		stbi_uc* capPixels = stbi_load(capturedPath.c_str(), &capW, &capH, &capN, 4);
		stbi_uc* goldPixels = stbi_load(goldPath.c_str(), &goldW, &goldH, &goldN, 4);

		const bool decoded = capPixels && goldPixels;
		const bool sameSize = decoded && capW == goldW && capH == goldH;
		const bool samePixels = sameSize && memcmp(capPixels, goldPixels, size_t(capW) * capH * 4) == 0;

		report.Check("SelfTest", "image_matches_gold_" + goldName, samePixels,
			capturedPath + " must exactly match " + goldPath + " (" +
			(!decoded ? "failed to decode one of the two PNGs" : !sameSize ? "image dimensions differ" :
				samePixels ? "OK" : "pixels differ") + ")");

		if (capPixels) stbi_image_free(capPixels);
		if (goldPixels) stbi_image_free(goldPixels);
	}
}

void App::selfTestTick()
{
	if (g_selfTest.step >= (int)std::size(kSelfTestScript))
		return; // already finished; shouldn't still be called, but don't index out of bounds if so

	if (g_selfTest.framesLeft > 0)
	{
		--g_selfTest.framesLeft;
		return;
	}

	const SelfTestStep& s = kSelfTestScript[g_selfTest.step];
	switch (s.kind)
	{
	case SelfTestStep::Kind::Setup1:
		setupScene1();
		break;
	case SelfTestStep::Kind::Setup2:
		setupScene2();
		break;
	case SelfTestStep::Kind::Setup3:
		setupScene3();
		break;
	case SelfTestStep::Kind::Save:
		g_sceneFileName = s.arg;
		saveScene();
		break;
	case SelfTestStep::Kind::Load:
		g_sceneFileName = s.arg;
		loadScene();
		break;
	case SelfTestStep::Kind::SetRayTracing:
		g_rayTracing = s.rayTracingOn;
		// Same reset the UI checkbox itself triggers (Settings window: "rayTracing" ORs into
		// resetTemporal) -- selfTestTick() runs outside that normal ImGui flow, so it has to call
		// this directly to avoid the ray/raster switch leaking stale TAA/GI history into the very
		// next capture.
		ResetTemporalHistory();
		break;
	case SelfTestStep::Kind::Capture:
	{
		const std::string path = "TestData/captured_" + std::string(s.arg) + ".png";
		if (CaptureRenderTargetToPNG(m_renderTarget, path))
			CompareOrCreateGoldImage(g_selfTest.report, path, s.arg);
		else
			g_selfTest.report.Check("SelfTest", std::string("screenshot_succeeds_") + s.arg, false,
				"CaptureRenderTargetToPNG failed for " + path);
		break;
	}
	case SelfTestStep::Kind::Finish:
		glfwSetWindowShouldClose(m_window, GLFW_TRUE);
		return; // leave g_selfTest.step alone -- run()'s loop is about to exit on its own
	}

	++g_selfTest.step;
	g_selfTest.framesLeft = kSelfTestSettleFrames;
}

int App::runSelfTest()
{
	// Deterministic rendering, independent of whatever Settings.ini happens to have on this
	// particular machine (a fresh checkout, or a different scene/setting left configured from an
	// interactive session) -- see this whole block's own header comment for why exact-match
	// comparison needs this. g_animateNoise=false is the important one (see SceneInfo::
	// frameNoiseSeed's own comment, shader_io.h): it fixes ray.slang's per-pixel RNG seed instead
	// of varying it by frame, the one piece of genuine run-to-run randomness in the render itself.
	g_animateNoise = false;
	g_enableAA = false;
	g_referenceAccumulation = false;
	g_nrdRealtimeMode = false;
	// The script's own SetRayTracing steps switch this later on; start from a known value rather
	// than whatever Settings.ini happened to leave it at, same reasoning as every other global here.
	g_rayTracing = true;
	// Belt-and-suspenders alongside ResetTemporalHistory() (now called by setupScene1()/
	// setupScene2()/loadScene() themselves) -- the EMA reservoir's own historyWeight blending is
	// exactly the kind of "previous frame/scene leaks into this one" temporal state this whole
	// function is trying to eliminate, on top of it needing a valid reprojection every step would
	// break anyway (bounce loop -> GIConstants::historyValid resets each scene change already).
	g_enableRestirGI = false;

	g_selfTest = {};
	g_selfTest.active = true;
	g_selfTest.report.SetOutputFile("VVerseSelfTestReport.txt");

	printf("VVerse self-test mode -- running scripted setup1/setup2/save/load checks.\n\n");

	run(); // ticks the script via selfTestTick() (see run()'s own hook), closes the window itself when done

	g_selfTest.report.PrintSummary();
	return g_selfTest.report.AllPassed() ? 0 : 1;
}

void App::run()
{
	if (!GImGui)
		return;

	// Main rendering loop
	while (!glfwWindowShouldClose(m_window))
	{
		// Logs the actual measured frame rate once per second, so VSync Mode behavior (and the
		// startup "bounce" above) can be verified/diagnosed from VVerse.txt without needing to
		// watch the window.
		if(false)
		{
			static double fpsWindowStart = glfwGetTime();
			static uint32 fpsWindowFrameCount = 0;
			++fpsWindowFrameCount;
			const double now = glfwGetTime();
			const double elapsed = now - fpsWindowStart;
			if (elapsed >= 1.0)
			{
				LOGI("FPS: %.1f (vSyncMode=%d)", fpsWindowFrameCount / elapsed, (int)m_vSyncMode);
				fpsWindowStart = now;
				fpsWindowFrameCount = 0;
			}
		}

		// Pacer: this is the frame rate limiter, it will limit the frame rate to the minimum refresh rate of the monitors.
		m_framePacer.paceFrame((m_vSyncMode == EVSyncMode::VSync) ? utils::getMonitorsMinRefreshRate() : 10000.0);

		// The very first swapchain is always bootstrapped with VSync/FIFO (see init()'s
		// initResources() call) regardless of m_vSyncMode -- rebuild it once here into whatever
		// mode was actually configured, completing the VSync-then-TearFree/Off "bounce" that
		// confirmed fixes Windows/NVIDIA not otherwise establishing proper flip-model/MAILBOX-
		// vblank-sync behavior. Time-based (not frame-count-based): at high/uncapped frame
		// rates a handful of frames could elapse before the window's even been shown.
		{
			static bool warmupRebuildDone = false;
			if (!warmupRebuildDone && glfwGetTime() > 0.5)
			{
				warmupRebuildDone = true;
				m_swapchain.requestRebuild();
			}
		}

		glfwPollEvents();

		{
			extern CRelativeMouseInput g_MouseInput;
			CRelativeMouseInput::MouseInputButtons Buttons = g_MouseInput.GetMouseButtons();

			// right mouse button
			if (Buttons & 2)
			{
				g_MouseInput.SetUserCapture("RightMouseButton");
			}
			else
			{
				g_MouseInput.ResetUserCapture("RightMouseButton");
			}

			CRelativeMouseInput::MouseInputData data = g_MouseInput.ClaimMouseInputData("RightMouseButton");

			if (auto* p2 = g_scene.m_sceneObjects.GetElementPtr(g_currentCamera))
			if (auto *p = g_CameraManager.GetElementPtr(p2->hndCamera))
			{
				//				const float rotateSpeed = 2 * InCamera.GetVerticalFov() / g_Renderer.GetHeight();
				const float rotateSpeed = 1 * 0.002f;		// todo
				const float movementSpeed = 50.0f / 10.0f;

				if (data.IsValid() && (data.RelativeX != 0 || data.RelativeY != 0))
				{
					float fInvMouse = 1.0;
					p2->transform.Rotate(rotateSpeed * data.RelativeX, -rotateSpeed * data.RelativeY * fInvMouse);
					g_temporalAccumulator = 0;
				}

				float3 forward = p2->transform.GetForward();
				float3 left = glm::normalize(glm::cross(forward, p2->transform.GetUp()));

				float dt = (float)ImGui::GetIO().DeltaTime;
				forward *= movementSpeed * dt;
				left *= movementSpeed * dt;

				float3 move(0, 0, 0);

				// When GetAsyncKeyState('S') fires during a Print Screen press on your JLAB keyboard, you are dealing with a hardware/firmware shortcut shortcut bypass. [1] (https://www.bestbuy.com/site/questions/jlab-jbuds-wireless-keyboard-black/6521118/question/f0bb5529-bb1b-3c50-bca6-36911ce1d1a1)Because many JLAB keyboards lack a physical PrtScn key, JLAB hardcodes a macro into their firmware. When you trigger their custom screenshot shortcut, the keyboard digitally blasts the full native Windows shortcut: Windows Key + Shift + S. [1] (https://www.bestbuy.com/site/questions/jlab-jbuds-wireless-keyboard-black/6521118/question/f0bb5529-bb1b-3c50-bca6-36911ce1d1a1)
				bool winPressed = (GetAsyncKeyState(VK_LWIN) & 0x8000) || (GetAsyncKeyState(VK_RWIN) & 0x8000);

				if (glfwGetWindowAttrib(m_window, GLFW_FOCUSED) && !winPressed)
				{
					if (GetAsyncKeyState(VK_SHIFT) & 0x8000)
					{
						left *= 3;
						forward *= 3;
					}

					if (GetAsyncKeyState('A') & 0x8000)
					{
						move += left;
						g_temporalAccumulator = 0;
					}
					else if (GetAsyncKeyState('D') & 0x8000)
					{
						move -= left;
						g_temporalAccumulator = 0;
					}

					if (GetAsyncKeyState('W') & 0x8000)
					{
						move += forward;
						g_temporalAccumulator = 0;
					}
					else if (GetAsyncKeyState('S') & 0x8000)
					{
						move -= forward;
						g_temporalAccumulator = 0;
					}
				}
				p2->transform.SetPos(p2->transform.GetPos() + move);
			}
		}

		if (glfwGetWindowAttrib(m_window, GLFW_ICONIFIED) == GLFW_TRUE)
		{
			ImGui_ImplGlfw_Sleep(10);  // Do nothing when minimized
			continue;
		}
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGuizmo::BeginFrame();

		/*--
		 * IMGUI Docking
		 * Create a dockspace and dock the viewport and settings window.
		 * The central node is named "Viewport", which can be used later with Begin("Viewport")
		 * to render the final image.
		-*/
		const ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingInCentralNode;
		ImGuiID dockID = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), dockFlags);
		// Docking layout, must be done only if it doesn't exist
		if (!ImGui::DockBuilderGetNode(dockID)->IsSplitNode() && !ImGui::FindWindowByName("Viewport"))
		{
			ImGui::DockBuilderDockWindow("Viewport", dockID);  // Dock "Viewport" to  central node
			ImGui::DockBuilderGetCentralNode(dockID)->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;  // Remove "Tab" from the central node
			ImGuiID leftID = ImGui::DockBuilderSplitNode(dockID, ImGuiDir_Left, 0.2f, nullptr, &dockID);  // Split the central node

			ImGuiID dock_id_left_top = 0;
			ImGuiID dock_id_left_bottom = 0;
			ImGui::DockBuilderSplitNode(leftID, ImGuiDir_Up, 0.50f, &dock_id_left_top, &dock_id_left_bottom);

			ImGui::DockBuilderDockWindow("Properties", dock_id_left_top);
			ImGui::DockBuilderDockWindow("Settings", dock_id_left_bottom);
			ImGui::DockBuilderDockWindow("Materials", dock_id_left_bottom);
			ImGui::DockBuilderDockWindow("Environment", dock_id_left_bottom);
			ImGui::DockBuilderDockWindow("Info", dock_id_left_bottom);
		}

		bool bReloadShaders = false;

		// [optional] Show the menu bar
		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Reload Shaders (F5)"))
					bReloadShaders = true;
				ImGui::Separator();
				if (ImGui::MenuItem("Exit"))
					glfwSetWindowShouldClose(m_window, true);
				ImGui::EndMenu();
			}
			ImGui::EndMainMenuBar();
		}
		/* END Docking */

		// We define "viewport" with no padding an retrieve the rendering area
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("Viewport");
		ImVec2 windowSize = ImGui::GetContentRegionAvail();
		{
			// Mouse position for interaction.slang's pick ray: GetCursorScreenPos() here is the
			// content region's top-left in the same (global, OS) screen space as GetIO().MousePos,
			// since ImGui::Image() below (the actual rendered frame, drawn later in drawFrame()) is
			// the first thing drawn in this same window/content-region and doesn't move the cursor
			// before it. Same NDC convention as ray.slang's fragPos (-1..1, pre-flip).
			ImVec2 viewportScreenPos = ImGui::GetCursorScreenPos();
			ImVec2 mousePos = ImGui::GetIO().MousePos;
			float2 localPos(mousePos.x - viewportScreenPos.x, mousePos.y - viewportScreenPos.y);

			const bool inView = windowSize.x > 0 && windowSize.y > 0 && localPos.x >= 0.0f && localPos.y >= 0.0f && localPos.x < windowSize.x && localPos.y < windowSize.y;

			g_viewportScreenPos = viewportScreenPos;
			g_viewportSize = windowSize;

			// IsOver()/IsUsing() reflect last frame's Manipulate() call (this frame's hasn't run yet,
			// see drawFrame()'s "Viewport" block) -- a one-frame-stale read, but enough to stop a click
			// on a gizmo handle from also firing interaction.slang's object-pick ray underneath it.
			if(inView && !ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !ImGuizmo::IsOver() && !ImGuizmo::IsUsing())
				g_pxMousePos = localPos;
			else
				g_pxMousePos = float2(FLT_MAX, FLT_MAX);

			g_mouseButtons = 0;
			if(ImGui::GetIO().MouseDown[0])
				g_mouseButtons |= 0x1;
			if (ImGui::GetIO().KeyCtrl)
				g_mouseButtons |= 1 << 31;
		}
		ImGui::End();
		ImGui::PopStyleVar();

		ImGuiIO& io = ImGui::GetIO();
		if (!io.WantCaptureKeyboard)
		{
			if (ImGui::IsKeyPressed(ImGuiKey_F5))
				bReloadShaders = true;
		}

		if(bReloadShaders)
		{
			g_renderer.reloadShaders();
#if USE_RAY_TRACING
			// reloadShaders() already did a full vkDeviceWaitIdle (see Renderer::reloadShaders),
			// so it's safe to rewrite every slot's descriptor set here regardless of which one
			// the GPU last used.
			for (uint32 i = 0; i < Renderer::kMaxFramesInFlight; ++i)
				updateRayDescriptorSet(i);
#endif
			g_temporalAccumulator = 0;
		}

		// Verify if the viewport has a new size and resize the RenderTarget accordingly.
		const VkExtent2D viewportSize = { uint32(windowSize.x), uint32(windowSize.y) };
		if (m_viewportSize.width != viewportSize.width || m_viewportSize.height != viewportSize.height)
		{
			onViewportSizeChange(viewportSize);
		}

		// Extra ImGui windows can be added here, like the demo window.
		// ImGui::ShowDemoWindow();

		// Frame Resource Preparation - only render if preparation succeeds
		if (prepareFrameResources())
		{
#if USE_RAY_TRACING
			{
				// interaction.slang's pick ray reports InstanceID() (entity picking) or the hit
				// triangle's own material index (Ctrl+LMB, material picking) -- see PickResult's own
				// comment (App.h) for why the frame-in-flight readback slot is hidden here
				// instead of every caller needing to know about it. GetHandleByIndex() resolves
				// either raw index directly; no mesh/TLAS-instance indirection needed on the CPU side.
				PickResult pick = GetPickResult();
				if (pick.index != -1)
				{
					if (pick.isMaterial)
					{
						auto hnd = g_MaterialManager.GetHandleByIndex((uint32)pick.index);
						if (hnd.IsValid())
						{
							g_selection.clear();
							g_selection.m_materials.insert(hnd);
						}
					}
					else
					{
						auto hnd = g_scene.m_sceneObjects.GetHandleByIndex((uint32)pick.index);
						if (hnd.IsValid())
						{
							g_selection.clear();
							g_selection.m_sceneObjects.insert(hnd);
						}
					}
				}
			}
#endif

			// m_frameCounter (pre-increment) is the timeline value THIS frame's submission will
			// eventually signal (see endFrame() below); any Buffer::deinit() called while
			// recording this frame is stamped with it, so it's only actually freed once the GPU
			// has proven it reached that point -- see DeferredRelease.h.
			g_deferredRelease.BeginFrame(m_frameCounter);

			// Begin command buffer recording
			VkCommandBuffer cmd = beginCommandRecording();

			// Record rendering commands
			drawFrame(cmd);

			// drawFrame() may have created staging buffers this frame (e.g. dispatchRayTracing()'s
			// per-frame g_meshBuffer/g_entityMeshIndexBuffer rebuild, Buffer::initRaw() under the
			// hood) -- queue each for the same frame-safe deferred destruction every other Buffer
			// already gets (see DeferredRelease.h), stamped with this frame's value via the
			// g_deferredRelease.BeginFrame() call above, so they're only actually freed once the GPU
			// has proven it finished the vkCmdCopyBuffer that reads them. Previously these only ever
			// got swept once, at the very first frame (see prepareFrameResources()'s old one-shot
			// startupStagingBuffersFreed) -- an unbounded per-frame leak once dispatchRayTracing()
			// started rebuilding those two buffers every frame instead of just once.
			for (const utils::Buffer& stagingBuffer : g_renderer.m_allocator.takeStagingBuffers())
				g_deferredRelease.Release(stagingBuffer);

			// Ends recording of commands for the frame
			endCommandRecording(cmd);

			// End frame and present
			endFrame(cmd);

			// See runSelfTest()/selfTestTick()'s own comments -- a frame has now actually been
			// rendered (endFrame() submitted it; CaptureRenderTargetToPNG's own fence wait, not
			// anything here, is what guarantees the GPU has finished it by the time a Capture step
			// reads m_renderTarget), so this is the right point to advance the self-test script.
			if (g_selfTest.active)
				selfTestTick();
		}
		else
		{
			// If frame preparation failed (e.g., swapchain needs rebuild), just skip this frame
			// The next frame will try again
		}

		// Update and Render additional Platform Windows (floating windows)
		ImGui::EndFrame();
		if ((ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
	}
}

void App::init()
{
	/*-- Resource model -----------------------------------------------------------
	 *
	 * This sample picks a different binding mechanism per resource class. They
	 * coexist freely; understanding which goes where is the key to reading
	 * the rest of this file.
	 *
	 *   Graphics shaders --> Shader objects (VK_EXT_shader_object).
	 *     There is no graphics VkPipeline at all. Three VkShaderEXT objects
	 *     (one vertex + two fragment variants) are created up front; at draw
	 *     time vkCmdBindShadersEXT picks the (vert, frag) pair, and every
	 *     piece of state that used to live in a pipeline (rasterization,
	 *     depth/stencil, blend, vertex input, ...) is set with a vkCmdSet*EXT
	 *     call. See createGraphicsShaders() and setGraphicsDynamicState().
	 *
	 *   Textures / samplers --> Descriptor heap (VK_EXT_descriptor_heap).
	 *     Sampler and image descriptors live in two GPU buffers (sampler heap
	 *     + resource heap). Shaders index them by integer slot. The graphics
	 *     shader objects carry VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT and no
	 *     descriptor set layouts. See createDescriptorHeap() and
	 *     recordGraphicsCommands().
	 *
	 *   Storage / uniform buffers --> Buffer device address (BDA).
	 *     The shader holds a buffer_reference (a typed GPU pointer) and
	 *     dereferences it directly. No descriptor needed. See m_vertexBuffer,
	 *     m_sceneInfoBuffer.
	 *
	 *   Small per-draw data --> vkCmdPushDataEXT (graphics) or
	 *                            vkCmdPushConstants2 (compute).
	 *     Graphics has no pipeline layout, so traditional push constants are
	 *     unavailable; vkCmdPushDataEXT writes directly to the shader's
	 *     push_constant block. Compute keeps the legacy path with a real
	 *     VkPushConstantRange in its pipeline layout.
	 *
	 *   ImGui still uses traditional VkDescriptorSet/VkDescriptorPool and its
	 *   own VkPipeline internally (m_uiDescriptorPool); that path is
	 *   independent from the heap and from shader objects.
	 *
	 * -----------------------------------------------------------------------*/

	 // Vulkan feature structs - allocated on the stack
	VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unifiedImageLayoutsFeature{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR };
	// Descriptor heap replaces traditional descriptor sets/pools with GPU buffer-based bindless descriptors.
	// Samplers and images are written into heap buffers and accessed by index in the shaders.
	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptorHeapFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT };
	// Untyped pointers: required by descriptor heap
	VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR };
	// Shader objects: replace VkPipeline for graphics with linkable, reusable VkShaderEXT
	// objects bound via vkCmdBindShadersEXT. Pairs naturally with the layout=NULL design:
	// there is no graphics pipeline object at all, only shader objects + dynamic state.
	VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT };
	// Extended dynamic state 3: required by shader objects for blend/rasterization
	// state that no longer lives in a pipeline object.
	VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dynamicState3Features{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT };
	// Vertex input dynamic state: required by shader objects (vertex bindings/attributes
	// become a vkCmdSetVertexInputEXT call instead of pipeline state).
	VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputDynamicStateFeatures{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT };

	// Configure Vulkan context with required and optional extensions
	utils::ContextCreateInfo contextConfig;

	// g_enableValidationLayers' own default mirrors this struct's default, so this is a no-op
	// unless --validate=0/1 overrode it -- see g_enableValidationLayers' own comment.
	contextConfig.enableValidationLayers = g_enableValidationLayers;

	// Required extensions (with their feature struct pointers)
	contextConfig.deviceExtensions.push_back({ VK_KHR_SWAPCHAIN_EXTENSION_NAME, true, nullptr });
	contextConfig.deviceExtensions.push_back({ VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME, true, &unifiedImageLayoutsFeature });
	contextConfig.deviceExtensions.push_back({ VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, true, &descriptorHeapFeatures });  // Bindless descriptor heap for textures and samplers
	contextConfig.deviceExtensions.push_back({ VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, true, &untypedPtrFeatures });  // Required by bindless
	contextConfig.deviceExtensions.push_back({ VK_EXT_SHADER_OBJECT_EXTENSION_NAME, true, &shaderObjectFeatures });  // Graphics: shader objects instead of pipelines
	contextConfig.deviceExtensions.push_back({ VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, true, &dynamicState3Features });  // Required for shader-object blend/rasterization state
	contextConfig.deviceExtensions.push_back({ VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, true,&vertexInputDynamicStateFeatures });  // Required for shader-object vertex input

	// ray tracing
	{
		contextConfig.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, true, nullptr });

		VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
		asFeatures.accelerationStructure = VK_TRUE;
		contextConfig.deviceExtensions.push_back({ VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, true, &asFeatures });

		VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
		rtFeatures.rayTracingPipeline = VK_TRUE;
		contextConfig.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, true, &rtFeatures });

		// NRD (g_nrdDenoiser, NrdIntegration.h): REBLUR's Blur/PostBlur passes use SPIR-V's
		// DerivativeGroupQuadsKHR (quad-group screen-space derivatives in a compute shader) --
		// without this feature enabled, vkCreateShaderModule rejects those pipelines' pre-embedded
		// SPIR-V outright (VUID-VkShaderModuleCreateInfo-pCode-08740), confirmed live the first time
		// NRD's pipelines were created against a device that didn't have it on.
		VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR computeShaderDerivativesFeatures{
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR };
		computeShaderDerivativesFeatures.computeDerivativeGroupQuads = VK_TRUE;
		contextConfig.deviceExtensions.push_back({ VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, true, &computeShaderDerivativesFeatures });
	}

	// Optional: richer diagnostics for VK_ERROR_DEVICE_LOST -- see OnVulkanDeviceLost() below, hooked
	// into VK_CHECK (vk_framework.h). VK_EXT_device_fault gives a fault description plus GPU
	// addresses implicated in the fault; VK_NV_device_diagnostic_checkpoints (NVIDIA) reports which
	// named command-buffer checkpoint the GPU last reached before dying, so a hang can be localized
	// to a phase (e.g. "the ray tracing dispatch" vs. "the graphics pass") without re-deriving it by
	// hand. Both optional (false): not every driver supports them, and vkGetDeviceProcAddr is
	// guaranteed by spec to return null for an extension's functions when it wasn't enabled, so the
	// availability checks at the call sites are enough -- no need to assert here.
	VkPhysicalDeviceFaultFeaturesEXT faultFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT };
	contextConfig.deviceExtensions.push_back({ VK_EXT_DEVICE_FAULT_EXTENSION_NAME, false, &faultFeatures });
	contextConfig.deviceExtensions.push_back({ VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, false, nullptr });

	// Create the Vulkan context with configuration
	g_renderer.m_context.init(contextConfig);

	// After context.init(), vkGetPhysicalDeviceFeatures2 has populated every
	// feature struct in the pNext chain. Assert the bools we actually depend
	// on -- vkCreateDevice will already have failed if a required feature is
	// missing, but these asserts give a clearer diagnostic on non-conformant
	// drivers and document the hard dependencies at a glance.
	ASSERT(unifiedImageLayoutsFeature.unifiedImageLayouts, "unifiedImageLayouts required (GENERAL attachment usage)");
	ASSERT(descriptorHeapFeatures.descriptorHeap, "descriptorHeap required");
	ASSERT(untypedPtrFeatures.shaderUntypedPointers, "shaderUntypedPointers required (by descriptorHeap)");
	ASSERT(shaderObjectFeatures.shaderObject, "shaderObject required (graphics path)");
	ASSERT(dynamicState3Features.extendedDynamicState3ColorBlendEnable, "extendedDynamicState3 required (shader objects)");
	ASSERT(vertexInputDynamicStateFeatures.vertexInputDynamicState, "vertexInputDynamicState required (shader objects)");

	// Initialize the VMA allocator. Pass the *device* API version (not the instance loader
	// version) clamped to the highest version VMA understands in this build (1.4).
	g_renderer.m_allocator.init(VmaAllocatorCreateInfo{
		.physicalDevice = g_renderer.m_context.getPhysicalDevice(),
		.device = g_renderer.m_context.getDevice(),
		.instance = g_renderer.m_context.getInstance(),
		.vulkanApiVersion = std::min(g_renderer.m_context.getDeviceApiVersion(), VK_API_VERSION_1_4),
		});

	// Texture sampler pool
	m_samplerPool.init(g_renderer.m_context.getDevice());

	// Create the window surface
	VK_CHECK(glfwCreateWindowSurface(g_renderer.m_context.getInstance(), m_window, nullptr, static_cast<VkSurfaceKHR*>(&m_surface)));
	DBG_VK_NAME(m_surface);

	// Used for creating single-time command buffers
	g_renderer.createTransientCommandPool();

	// Create the swapchain
	m_swapchain.init(g_renderer.m_context.getPhysicalDevice(), g_renderer.m_context.getDevice(), g_renderer.m_context.getGraphicsQueue(), m_surface, g_renderer.m_transientCmdPool);
	// Bootstrap the very first swapchain with VSync (FIFO) regardless of m_vSyncMode -- confirmed
	// (see the run()-loop warm-up rebuild below) that Windows/NVIDIA only properly establishes
	// flip-model/MAILBOX-vblank-sync behavior if the window's swapchain has been through FIFO at
	// least once; going straight to MAILBOX/IMMEDIATE first silently behaves like Off (uncapped,
	// tearing) even when TearFree/VSync is requested, until rebuilt after a FIFO present.
	m_windowSize = m_swapchain.initResources(EVSyncMode::VSync);  // Update the window size to the actual size of the surface

	// Create what is needed to submit the scene for each frame in-flight
	// m_frameData is sized by frames-in-flight (CPU parallelism), NOT by
	// imageCount (GPU/presentation parallelism).
	createFrameSubmission(m_swapchain.getFramesInFlight());

	// Create a descriptor pool for ImGui (it still needs a traditional pool for its own textures)
	createUIDescriptorPool();

	// Initializing Dear ImGui
	initImGui();

	// Create the RenderTarget (offscreen color + depth)
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

		const VkFormat depthFormat = utils::findDepthFormat(g_renderer.m_context.getPhysicalDevice());
		utils::RenderTargetCreateInfo rtInit{
			.device = g_renderer.m_context.getDevice(),
			.alloc = &g_renderer.m_allocator,
			.size = m_windowSize,
			.color = {VK_FORMAT_R8G8B8A8_UNORM},  // Single color attachment
			.depth = depthFormat,
		};
		m_renderTarget.init(cmd, rtInit);
		m_imTexture.init(m_renderTarget.getUiImageView(0));  // ImGui texture wrapper for the RenderTarget's color image view

		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

#if USE_RAY_TRACING
	// if NV_ALLOW_RAYTRACING_VALIDATION env variable is set
	// see https://developer.nvidia.com/blog/ray-tracing-validation-at-the-driver-level/
	// Check for ray tracing validation support and enable the feature
	if (false)
	{
		VkPhysicalDeviceRayTracingValidationFeaturesNV validationFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV };
		VkPhysicalDeviceFeatures2 features = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
		features.pNext = &validationFeatures;
		vkGetPhysicalDeviceFeatures2(g_renderer.m_context.getPhysicalDevice(), &features);

		// Confirm that the ray tracing validation feature is supported
		ASSERT(validationFeatures.rayTracingValidation == (VkBool32)true);
	}

#endif

	//	const char* spGetBuildTagString();
	//	Log("Slang Version: %s\n", spGetBuildTagString());

	Log("\n");

	g_MeshManager.InitOnce(4096);
	g_MaterialManager.InitOnce(4096);
	g_scene.m_sceneObjects.InitOnce(4096);
	g_LightManager.InitOnce(4096);
	g_CameraManager.InitOnce(4096);

	extern CObjManager<CTexHandle, Texture> g_TextureManager;
	g_TextureManager.InitOnce(4096);

	// Create GPU buffers (SSBO) containing the vertex data and the point data, and the image (uploading data to GPU)
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

		g_renderer.init(g_renderer.m_context.getPhysicalDevice(), cmd);

		EventContext context{ g_renderer.m_context.getDevice(), g_renderer.m_allocator };
		// after g_renderer.init as this is where the shadercache is loaded
		g_EventManager.callAppInit(EF_Type_Mask, context);

#if USE_RAY_TRACING
		EnsureOutputBufferCapacity(cmd, VkDeviceSize(m_viewportSize.width) * m_viewportSize.height);
		LOGI("g_outputBuffer initial alloc: %ux%u, m_outputBufferCapacityPixels tracker = %llu",
			m_viewportSize.width, m_viewportSize.height, (unsigned long long)m_outputBufferCapacityPixels);
		EnsureRtxdiCapacity(m_viewportSize.width, m_viewportSize.height);
		EnsureGIReservoirCapacity(m_viewportSize.width, m_viewportSize.height);
		EnsureGBufferCapacity(m_viewportSize.width, m_viewportSize.height);

		g_nrdDenoiser.init();
		g_nrdDenoiser.ensureCapacity(m_viewportSize.width, m_viewportSize.height,
			g_gbufferNormalRoughness.view, g_gbufferViewZ.view, g_gbufferAlbedo.view);
#endif

		// Image-based lighting is NOT loaded here -- see setupScene1()/setupScene2()/loadScene()'s
		// own g_ibl.load() call sites (each guarded by !g_ibl.isValid()) for why loading it this
		// early, before any real scene texture, broke ray tracing entirely (it silently claimed
		// descriptor-heap index 0, which Shader::updateRayDescriptorSet() requires to always be a
		// real Texture2D).

		// Create the descriptor heap buffers (sampler + resource) and upload them to the GPU.
		// This must happen after every texture (test images + any glTF material textures) has
		// registered itself in g_renderer.m_descriptorHeap, since build() uploads whatever has
		// accumulated there in one pass.
		createDescriptorHeap(cmd);

		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}
	// Staging buffers created above are already sitting in g_renderer.m_allocator, no longer freed
	// here directly -- the main loop's run() now drains them into g_deferredRelease every frame
	// (see that call site's comment), the same frame-safe deferred destruction every other Buffer
	// already gets, instead of each caller managing its own staging buffers' lifetime by hand.

	// Create a buffer to store the scene information, updated once per frame via vkCmdUpdateBuffer.
	// The shader accesses it through its buffer device address (BDA) and also binds it as a
	// storage buffer descriptor in the ray-tracing pass.
	m_sceneInfoBuffer = g_renderer.m_allocator.createBuffer(sizeof(shaderio::SceneInfo),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT
		| VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
	DBG_VK_NAME(m_sceneInfoBuffer.buffer);

	// GPU->CPU selection readback -- see g_selectionReadback's own comment. HOST_ACCESS_RANDOM_BIT
	// (this codebase's "Readback" recipe, see ResourceAllocator::createBuffer's comment) +
	// MAPPED_BIT gives a persistently-mapped host-visible pointer, read directly with no per-frame
	// map/unmap.
	g_ReadbackBuffer = g_renderer.m_allocator.createBuffer(
		Renderer::kMaxFramesInFlight * 2 * sizeof(uint32),
		VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
		VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
	DBG_VK_NAME(g_ReadbackBuffer.buffer);
	{
		VmaAllocationInfo allocInfo{};
		vmaGetAllocationInfo(g_renderer.m_allocator, g_ReadbackBuffer.allocation, &allocInfo);
		g_ReadbackMapped = (uint32*)allocInfo.pMappedData;

		// index=-1 (asfloat/asint bit pattern), isMaterial=0 in every slot -- "nothing picked yet",
		// matching interaction.slang's own miss/no-mouse sentinel -- so GetPickResult()'s first call
		// has a defined value to compare against instead of whatever this memory happened to contain.
		for (uint32 i = 0; i < Renderer::kMaxFramesInFlight; ++i)
		{
			int32 invalid = -1;
			memcpy(&g_ReadbackMapped[i * 2 + 0], &invalid, sizeof(invalid));
			g_ReadbackMapped[i * 2 + 1] = 0;
		}
	}
}

void App::destroy()
{
	VkDevice device = g_renderer.m_context.getDevice();

	if (device)
		VK_CHECK(vkDeviceWaitIdle(device));

	m_swapchain.deinit();
	m_samplerPool.deinit();
	m_imTexture.deinit();  // Destroy the ImGui texture wrapper before shutting down ImGui itself
	g_taa.imTextureHistory[0].deinit();
	g_taa.imTextureHistory[1].deinit();

	if (device)
		g_renderer.deinit(device, g_renderer.m_allocator);

	if (GImGui)
	{
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	if (device)
	{
		vkDestroyPipeline(device, m_computePipeline, nullptr);
		vkDestroyPipelineLayout(device, m_computePipelineLayout, nullptr);

		// Each Mesh's BLAS handle (m_BottomLevelAS) is a plain VkAccelerationStructureKHR, not a
		// Buffer, so it isn't reached by the IAppInit/g_deInitListener mechanism below (which only
		// covers Buffer/Shader) -- destroy it explicitly here, before that call destroys the Buffer
		// (m_BLAS) backing its memory. Left undestroyed, this is exactly the leak the validation
		// layer reports at vkDestroyDevice ("N leaked objects... VkAccelerationStructureKHR").
		{
			const uint32 meshCount = g_MeshManager.GetElementCount();
			for (uint32 i = 0; i < meshCount; ++i)
			{
				RenderComponent& mesh = g_MeshManager.GetElementByIterator(i);
				if (mesh.m_BottomLevelAS != VK_NULL_HANDLE)
				{
					vkDestroyAccelerationStructureKHR(device, mesh.m_BottomLevelAS, nullptr);
					mesh.m_BottomLevelAS = VK_NULL_HANDLE;
				}
			}
		}

#if USE_RAY_TRACING
		// Same reasoning as the BLAS cleanup above: plain VkImage/VkImageView, not Buffer/Shader,
		// so never reached by the g_deInitListener mechanism below -- left undestroyed, this is
		// exactly the kind of leak the validation layer reports at vkDestroyDevice (observed as
		// an error on quit before this was added).
		destroyGBufferTarget(g_gbufferAlbedo);
		destroyGBufferTarget(g_gbufferNormalRoughness);
		destroyGBufferTarget(g_gbufferViewZ);
		destroyGBufferTarget(g_gbufferMotionVector);

		g_nrdDenoiser.deinit();
#endif

		// TAA history ping-pong buffers (see TaaResolveState::history, TaaResolve.h) -- not gated by
		// USE_RAY_TRACING like the g-buffers above, since TAA resolves both the raster and ray traced
		// paths. Same leak class as those: plain VkImage/VkImageView, never reached by the
		// IAppInit/g_deInitListener mechanism below. Left undestroyed, this is exactly the "UNFREED
		// ALLOCATION"/"Some allocations were not freed" VMA assert seen at quit before this was added.
		destroyGBufferTarget(g_taa.history[0]);
		destroyGBufferTarget(g_taa.history[1]);

		// call deinit() on all objects that listen to g_deInitListener
		// e.g. class Shader, class Buffer
		EventContext context{ device, g_renderer.m_allocator };
		g_EventManager.callAppDeInit(EF_Type_Mask, context);

		vkDestroyCommandPool(device, g_renderer.m_transientCmdPool, nullptr);
		vkDestroySurfaceKHR(g_renderer.m_context.getInstance(), m_surface, nullptr);
		vkDestroyDescriptorPool(device, m_uiDescriptorPool, nullptr);  // ImGui descriptor pool

		// Frame info
		for (size_t i = 0; i < m_frameData.size(); i++)
		{
			vkFreeCommandBuffers(device, m_frameData[i].cmdPool, 1, &m_frameData[i].cmdBuffer);
			vkDestroyCommandPool(device, m_frameData[i].cmdPool, nullptr);
		}
		vkDestroySemaphore(device, m_frameTimelineSemaphore, nullptr);
	}

	g_renderer.m_allocator.destroyBuffer(m_sceneInfoBuffer);     // Scene info GPU buffer (accessed via BDA)
	g_renderer.m_allocator.destroyBuffer(g_ReadbackBuffer);   // GPU->CPU selection-index readback
	g_renderer.m_allocator.destroyBuffer(m_samplerHeapBuffer);   // Descriptor heap: sampler GPU buffer

	// Every loaded texture (hand-authored test images + any glTF material textures) is
	// tracked in g_TextureManager, the same way meshes/materials are tracked in their own
	// managers -- destroy each one's GPU resources here.
	{
		extern CObjManager<CTexHandle, Texture> g_TextureManager;
		const uint32 textureCount = g_TextureManager.GetElementCount();
		for (uint32 i = 0; i < textureCount; ++i)
		{
			Texture& tex = g_TextureManager.GetElementByIterator(i);
			tex.deinit(device, g_renderer.m_allocator);
		}
	}

	g_ibl.deinit(device, g_renderer.m_allocator);

	m_renderTarget.deinit();

	// device is confirmed idle (VK_CHECK(vkDeviceWaitIdle(device)) above), and every
	// Buffer::deinit() call from this shutdown sequence -- including g_EventManager's
	// callAppDeInit() above, which reaches every still-alive Buffer (mesh vertex/index/BLAS,
	// g_meshBuffer, g_materialBuffer, each ray tracing Shader's own m_sbt, g_outputBuffer, ...) --
	// has already run by this
	// point, so it's safe to free everything g_deferredRelease is still holding
	// unconditionally, ahead of destroying the allocator that owns it.
	g_deferredRelease.deinit(g_renderer.m_allocator);

	g_renderer.m_allocator.deinit();
	g_renderer.m_context.deinit();
}

void App::createFrameSubmission(uint32 numFrames)
{
	VkDevice device = g_renderer.m_context.getDevice();

	m_frameData.resize(numFrames);

	// Initialize timeline semaphore at 0. We'll use a monotonic counter (m_frameCounter) starting at 1.
	const uint64 initialValue = 0;

	VkSemaphoreTypeCreateInfo timelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
		.pNext = nullptr,
		.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
		.initialValue = initialValue,
	};

	/*--
	 * Create timeline semaphore for GPU-CPU synchronization
	 * This ensures resources aren't overwritten while still in use by the GPU
	-*/
	const VkSemaphoreCreateInfo semaphoreCreateInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timelineCreateInfo };
	VK_CHECK(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &m_frameTimelineSemaphore));
	DBG_VK_NAME(m_frameTimelineSemaphore);

	/*--
	 * Create command pools and buffers for each frame
	 * Each frame gets its own command pool to allow parallel command recording while previous frames may still be executing on the GPU
	-*/
	const uint32 queueFamily = g_renderer.m_context.getGraphicsQueue().familyIndex;

	for (uint32 i = 0; i < numFrames; i++)
	{
		m_frameData[i].lastSignalValue = initialValue;  // Initialize to timeline semaphore's initial value

		// Separate pools allow independent reset/recording of commands while other frames are still in-flight
		m_frameData[i].cmdPool = utils::createCommandPool(device, queueFamily);
		DBG_VK_NAME(m_frameData[i].cmdPool);

		const VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = m_frameData[i].cmdPool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VK_CHECK(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &m_frameData[i].cmdBuffer));
		DBG_VK_NAME(m_frameData[i].cmdBuffer);
	}
}

bool SliderTexHandle(const char* label, CTexHandle& hnd)
{
	int count = (int)g_TextureManager.GetElementCount();

	uint32 it = g_TextureManager.GetIteratorFromHandle(hnd);
	char str[80];
	sprintf_s(str, sizeof(str), "#%d.%d", (uint32)hnd.Index, (uint32)hnd.Seed);
	bool ret = ImGui::SliderInt(label, (int*)&it, 0, count - 1, str);

	if(ret)
		hnd = g_TextureManager.GetHandleByIterator(it);

	return ret;
}

bool ImGuiMaterial(CMatHandle hnd)
{
	bool resetTemporal = false;

	if(auto* p = g_MaterialManager.GetElementPtr(hnd))
	{
		auto &el = *p;
		ImGui::Text("Material #%d.%d", (uint32)hnd.Index, (uint32)hnd.Seed);
		resetTemporal |= ImGui::Combo("Type", (int*)&el.m_type, Material::typeUI, 2);
		resetTemporal |= ImGui::InputText("Name", &el.m_name, ImGuiInputTextFlags_ReadOnly);
//		resetTemporal |= ImGui::InputText("Name", (char*)el.m_name.c_str(), el.m_name.size(), ImGuiInputTextFlags_ReadOnly);
		resetTemporal |= ImGui::ColorEdit3("Albedo Color", (float*)&el.m_albedoColor);
		resetTemporal |= ImGui::ColorEdit3("Emissive Color", (float*)&el.m_emissiveColor);
		resetTemporal |= ImGui::SliderFloat("EmissiveIntensity", &el.m_emissiveIntensity, 0.0f, 16.0f);
		resetTemporal |= ImGui::SliderFloat("Roughness", &el.m_roughness, 0.0f, 1.0f);
		resetTemporal |= ImGui::SliderFloat("Metallic", &el.m_metallic, 0.0f, 1.0f);
		resetTemporal |= ImGui::SliderFloat("AlbedoTextureFade", &el.m_albedoTextureFade, 0.0f, 1.0f);
		resetTemporal |= ImGui::SliderFloat("SpecularOcclusionFromAlbedo (0=off)", &el.m_specOcclusionFromAlbedo, 0.0f, 1.0f);
		resetTemporal |= SliderTexHandle("Albedo texture", el.m_hndTexture);
		resetTemporal |= SliderTexHandle("MetallicRoughness texture", el.m_hndMetallicRoughnessTexture);
		resetTemporal |= SliderTexHandle("Lightmap texture", el.m_hndLightmap);
		resetTemporal |= SliderTexHandle("Normal map texture", el.m_hndNormalMap);
	}

	return resetTemporal;
}

bool ImGuiCamera(CCameraHandle hnd)
{
	bool resetTemporal = false;

	if (auto* p = g_CameraManager.GetElementPtr(hnd))
	{
		auto& el = *p;
		resetTemporal |= ImGui::SliderFloat("Field of View", &el.fieldOfView, 10.0f, 120.0f);
	}

	return resetTemporal;
}

bool ImGuiLight(CLightHandle hnd)
{
	bool resetTemporal = false;

	if (auto* p = g_LightManager.GetElementPtr(hnd))
	{
		auto& el = *p;
		//	resetTemporal |= ImGui::SliderFloat3("lightPos", (float*)&m_lightPos, -3, 6);
		resetTemporal |= ImGui::SliderFloat("lightLength", &el.m_lightLength, 0, 6);
		resetTemporal |= ImGui::ColorEdit3("lightColor", (float*)&el.m_lightColor);
		resetTemporal |= ImGui::SliderFloat("lightIntensity", &el.m_lightIntensity, 0, 10);
	}

	return resetTemporal;
}

void App::drawFrame(VkCommandBuffer cmd)
{
	// Custom TAA resolve (taa_resolve.slang): decide this frame's ping-pong write slot once, up
	// front -- ImGui::Image() below needs to already reference the right descriptor (its content
	// gets filled in later this same frame, by the resolve dispatch in recordGraphicsCommands'
	// call site further down; ImGui only needs the handle now, execution order handles the rest),
	// and updateSceneBuffer()/the resolve dispatch both need to agree on the same slot too. Flipped
	// exactly once per frame, here, and read (not re-flipped) everywhere else this frame.
	g_taa.historyIndex = 1 - g_taa.historyIndex;

	// ImGui

	/*--
	 * The rendering of the scene is done using dynamic rendering into the RenderTarget (see recordGraphicsCommands).
	 * The target image will be rendered/displayed using ImGui.
	 * Its placement will cover the entire viewport (ImGui draws a quad with the texture we provide),
	 * and the image will be displayed in the viewport.
	 * There are multiple ways to display the image, but this method is the most flexible.
	 * Other methods include:
	 *  - Blitting the image to the swapchain image, with the UI drawn on top. However, this makes it harder
	 *    to fit the image within a specific area of the window.
	 *  - Using the image as a texture in a quad and rendering it to the swapchain image. This is what ImGui
	 *    does, but we don't need to add a quad to the scene, as ImGui handles it for us.
	-*/
	// Using the dock "Viewport", this sets the window to cover the entire central viewport
	if (ImGui::Begin("Viewport"))
	{
		// !!! This is where the RenderTarget image is displayed !!!
		// When AA is on, this frame's resolve dispatch (see recordGraphicsCommands' call site,
		// further down) fills g_taa.imTextureHistory[g_taa.historyIndex] instead of m_renderTarget's
		// own raw (still-jittered) color -- see TaaResolveState::history's comment (TaaResolve.h).
		if (g_enableAA && g_taa.renderTargetColorHeapIndex != shaderio::kInvalidTextureId)
		{
			// g_taa.history is grow-only (EnsureTaaHistoryCapacity), so it's sized to the largest
			// viewport ever requested (g_taa.capacityWidth/Height), not necessarily today's
			// m_viewportSize -- unlike m_renderTarget's own color image, which RenderTarget::update()
			// destroys/recreates to the exact current size on every resize. taa_resolve.slang only
			// ever writes the top-left m_viewportSize.width x m_viewportSize.height region of it (see
			// setTaaResolveDynamicState's viewport/scissor, this frame's resolve dispatch); the rest
			// still holds whatever an earlier, larger viewport last wrote there. Displaying the whole
			// texture unscaled-UV (ImGui::Image's implicit uv1=(1,1)) after shrinking the window back
			// down showed that leftover region below/right of the freshly-written area -- exactly the
			// "top is only updated" artifact reported after a resize. Crop to the actually-written
			// sub-rect instead.
			const ImVec2 uv1(float(m_viewportSize.width) / float(g_taa.capacityWidth), float(m_viewportSize.height) / float(g_taa.capacityHeight));
			ImGui::Image(g_taa.imTextureHistory[g_taa.historyIndex], ImGui::GetContentRegionAvail(), ImVec2(0, 0), uv1);
		}
		else
			ImGui::Image(m_imTexture, ImGui::GetContentRegionAvail());

		// Adding overlay text on the upper left corner
		ImGui::SetCursorPos(ImVec2(0, 0));
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

#if USE_RAY_TRACING
		if(g_rayTracing)
			ImGui::Text("TemporalFrames: %d", g_temporalAccumulator);
#endif
		for(const auto &el : g_UIErrors)
		{
			ImGui::TextColored(ImVec4(1,0,0,1), "Error: %s", el.c_str());
		}
		g_UIErrors.clear();

		// Move/rotate gizmo for the single selected entity -- "selected" here is exactly what the
		// Properties panel (below) shows, i.e. g_selection.m_sceneObjects. Only handles a single
		// selection; with several entities selected there's no one Transform to manipulate.
		if (g_selection.m_sceneObjects.size() == 1 && m_viewportSize.width > 0 && m_viewportSize.height > 0)
		{
			CEntityHandle hnd = *g_selection.m_sceneObjects.begin();
			if (Entity* entity = g_scene.m_sceneObjects.GetElementPtr(hnd))
			{
				// Same camera lookup as updateSceneBuffer() (recordGraphicsCommands(), further down
				// in this same function) -- duplicated here since that runs later this frame and this
				// gizmo needs the view/projection matrices now, to draw and manipulate immediately.
				float fieldOfView = 45.0f;
				Transform cameraTransform;
				if (auto* p2 = g_scene.m_sceneObjects.GetElementPtr(g_currentCamera))
					if (auto* p = g_CameraManager.GetElementPtr(p2->hndCamera))
					{
						fieldOfView = p->fieldOfView;
						cameraTransform = p2->transform;
					}

				const glm::mat4 view = cameraTransform.GetViewMatrix();
				const glm::mat4 proj = glm::perspectiveLH_ZO(glm::radians(fieldOfView), m_viewportSize.width / (float)m_viewportSize.height, 0.1f, 100000.0f);
				glm::mat4 model = entity->transform.GetObjectMatrix();

				ImGuizmo::SetOrthographic(false);
				ImGuizmo::SetDrawlist();
				ImGuizmo::SetRect(g_viewportScreenPos.x, g_viewportScreenPos.y, g_viewportSize.x, g_viewportSize.y);

				// default => translate
				ImGuizmo::OPERATION gizmoOperation = ImGuizmo::TRANSLATE;

				if (glfwGetWindowAttrib(m_window, GLFW_FOCUSED))
				{
					if (GetAsyncKeyState(VK_SHIFT) & 0x8000)	// SHIFT => rotate
						gizmoOperation = ImGuizmo::ROTATE;
					if (GetAsyncKeyState(VK_MENU) & 0x8000)		// ALT => scale
						gizmoOperation = ImGuizmo::SCALE;
				}

				if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj), gizmoOperation, ImGuizmo::LOCAL, glm::value_ptr(model)))
				{
					// Only touch the field the active operation actually edits, instead of
					// decomposing the whole matrix (position+rotation+scale together) and writing
					// all three back every frame -- a pure translate mathematically can't change
					// rotation, but round-tripping rotation through a full decompose/recompose every
					// frame of the drag (a generic decomposition, ambiguous up to a sign/axis flip)
					// let it drift or flip anyway, which is what made translating visibly rotate the
					// object.
					if (gizmoOperation == ImGuizmo::TRANSLATE)
					{
						entity->transform.position = float3(model[3]);
					}
					else if (gizmoOperation == ImGuizmo::ROTATE)
					{
						// model's upper-left 3x3 is scale*rotation (GetObjectMatrix() applies scale
						// before rotation, so each column is a rotation axis stretched by that axis'
						// entity->transform.scale) -- normalize each column to unit length to discard
						// the scale (unchanged by a rotate op anyway) and recover a pure rotation
						// matrix exactly, rather than depending on glm::decompose()'s general (and,
						// per the above, unstable-in-practice) polar decomposition.
						const glm::mat3 rot3(glm::normalize(float3(model[0])), glm::normalize(float3(model[1])), glm::normalize(float3(model[2])));
						entity->transform.rot = glm::quat_cast(rot3);
					}
					else if (gizmoOperation == ImGuizmo::SCALE)
					{
						entity->transform.scale = float3(length(model[0]), length(model[1]), length(model[2]));
					}
					// The below (drawFrame()'s own) resetTemporal isn't declared until after this
					// block -- set g_temporalAccumulator directly, same as every other place outside
					// that flag's scope (e.g. loadScene(), onViewportSizeChange()). Otherwise the
					// moved/rotated object only fades in over many accumulated frames instead of
					// updating immediately.
					g_temporalAccumulator = 0;
				}
			}
		}
	}
	ImGui::End();

	bool resetTemporal = false;

	if (ImGui::Begin("Materials"))
	{
		uint32 materialCount = g_MaterialManager.GetElementCount();

		if (materialCount)
		{
			static uint32 materialIndex = 0;	// todo
			ImGui::SliderInt("MaterialIndex", (int*)&materialIndex, 0, materialCount - 1);
			{
				ImGui::PushID(materialIndex);
				auto hnd = g_MaterialManager.GetHandleByIterator(materialIndex);
				auto& el = g_MaterialManager.GetElementByIterator(materialIndex);
				{
					resetTemporal |= ImGuiMaterial(hnd);
					ImGui::Separator();
				}
				ImGui::PopID();
			}
		}
	}
	ImGui::End();

	if (ImGui::Begin("Info"))
	{
		ImGui::Text("Entities: %u", (uint32)g_scene.m_sceneObjects.size());
		ImGui::Text("Materials: %u", (uint32)g_MaterialManager.size());
		ImGui::Text("Textures: %u", (uint32)g_TextureManager.size());
//		ImGui::Text("Shaders: %u", (uint32)g_renderer.size());
	}
	ImGui::End();

	if (ImGui::Begin("Properties"))
	{
		CRenderCompHandle selectMesh;
		CMatHandle selectMat;
		CLightHandle selectLight;

		for (auto hnd : g_selection.m_sceneObjects)
		{
			if(auto* p = g_scene.m_sceneObjects.GetElementPtr(hnd))
			{
				auto &el = *p;
				ImGui::PushID(hnd.Index);
				ImGui::Text("Entity #%d.%d", (uint32)hnd.Index, (uint32)hnd.Seed);

				if(el.hndRenderComp.IsValid())
				{
					ImGui::PushID("Mesh");
					if (ImGui::SmallButton("p"))	// p for properties
						selectMesh = el.hndRenderComp;
					ImGui::SameLine();
					ImGui::Text("RenderComp #%d.%d", (uint32)el.hndRenderComp.Index, (uint32)el.hndRenderComp.Seed);
					ImGui::PopID();
				}

				if (el.hndLight.IsValid())
				{
					ImGui::PushID("Light");
					if (ImGui::SmallButton("p"))	// p for properties
						selectLight = el.hndLight;
					ImGui::SameLine();
					ImGui::Text("Light #%d.%d", (uint32)el.hndLight.Index, (uint32)el.hndLight.Seed);
					ImGui::PopID();
				}
				ImGui::PopID();
			}
		}
		for (auto hnd : g_selection.m_meshes)
		{
			if (auto* p = g_MeshManager.GetElementPtr(hnd))
			{
				auto& el = *p;
				ImGui::PushID(hnd.Index);
				ImGui::Text("Mesh #%d.%d", (uint32)hnd.Index, (uint32)hnd.Seed);
				uint32 index = 0;
				for(auto& sub : el.m_subMeshes)
				{
					ImGui::PushID(index);
					if (auto* p2 = g_MaterialManager.GetElementPtr(sub.hndMaterial))
					{
						auto& el2 = *p2;
						if (ImGui::SmallButton("p"))	// p for properties
							selectMat = sub.hndMaterial;
						ImGui::SameLine();
						ImGui::Text("Material #%d.%d %s", (uint32)sub.hndMaterial.Index, (uint32)sub.hndMaterial.Seed, el2.m_name.c_str());
					}
					ImGui::PopID();
					++index;
				}
				ImGui::PopID();
			}
		}
		for (auto hnd : g_selection.m_materials)
		{
			ImGui::PushID(hnd.Index);
			resetTemporal |= ImGuiMaterial(hnd);
			ImGui::PopID();
		}
		for (auto hnd : g_selection.m_lights)
		{
			ImGui::PushID(hnd.Index);
			resetTemporal |= ImGuiLight(hnd);
			ImGui::PopID();
		}
		for (auto hnd : g_selection.m_cameras)
		{
			ImGui::PushID(hnd.Index);
			resetTemporal |= ImGuiCamera(hnd);
			ImGui::PopID();
		}

		if(selectMesh.IsValid())
		{
			g_selection.clear();
			g_selection.m_meshes.insert(selectMesh);
		}
		else if (selectMat.IsValid())
		{
			g_selection.clear();
			g_selection.m_materials.insert(selectMat);
		}
		else if (selectLight.IsValid())
		{
			g_selection.clear();
			g_selection.m_lights.insert(selectLight);
		}
	}
	ImGui::End();

	if (ImGui::Begin("Environment"))
	{
		resetTemporal |= ImGui::Checkbox("Flat Sky Color (otherwise IBL)", &g_flatSkyColor);

		{
			ImEnable_RAII enableColor(g_flatSkyColor);
			resetTemporal |= ImGui::ColorEdit3("skyRGB", (float*)&g_skyRGB);
			resetTemporal |= ImGui::SliderFloat("skyIntensity", &g_skyIntensity, 0, 10);
		}
		ImGui::Separator();
		// Basic homogeneous fog (ray.slang) -- see shader_io.h's SceneInfo::fogDensity/fogColor/
		// fogAsymmetry comment.
		resetTemporal |= ImGui::SliderFloat("Fog Density", &g_fogDensity, 0.0f, 1.0f);
		resetTemporal |= ImGui::ColorEdit3("Fog Color", (float*)&g_fogColor);
		resetTemporal |= ImGui::SliderFloat("Fog Asymmetry (g)", &g_fogAsymmetry, -0.99f, 0.99f);
		ImGui::Separator();
		resetTemporal |= ImGui::SliderFloat3("laserPos", (float*)&g_laserPos, -3, 6);
		resetTemporal |= ImGui::SliderFloat("laserAngle", &g_laserAngle, 0, 360);
		resetTemporal |= ImGui::SliderFloat("laserIntensity", &g_laserIntensity, 0, 300);

		if (auto* p2 = g_scene.m_sceneObjects.GetElementPtr(g_currentCamera))
		if (auto* p = g_CameraManager.GetElementPtr(p2->hndCamera))
		{
			ImGui::Separator();
			resetTemporal |= ImGui::SliderFloat("FieldOfView", &p->fieldOfView, 10, 120);
		}
	}
	ImGui::End();

	if (ImGui::Begin("Settings"))
	{
		resetTemporal |= ImGui::SliderFloat("ExposureLog", &g_ExposureLog, -5, 5);
		resetTemporal |= ImGui::Checkbox("Tonemapper", &g_enableTonemapper);
		resetTemporal |= ImGui::SliderFloat("Lightmap Intensity", &g_lightmapIntensity, 0.0f, 4.0f);

		ImGui::Separator();

		resetTemporal |= ImGui::Combo("VisualizeMode", (int*)&g_visualizeMode, shaderio::VisualizeMode::typeUI, 20);
		if(ImGui::Combo("VSync Mode", (int*)&m_vSyncMode, "Off\0" "TearFree\0" "VSync\0"))
			m_swapchain.requestRebuild();  // Recreate the swapchain with the new vSync setting

#if USE_RAY_TRACING
		resetTemporal |= ImGui::Checkbox("rayTracing", &g_rayTracing);

		// Realtime: NRD denoises a fresh single-sample frame every frame (temporalAccumulator
		// forced to 0 below, see updateSceneBuffer()). Off: TAA + ReSTIR DI handle stability instead
		// (ordinary interactive use), unless Reference Accumulation below is also on.
		resetTemporal |= ImGui::Checkbox("NRD Realtime Denoise", &g_nrdRealtimeMode);
		if (g_nrdRealtimeMode)
		{
			ImGui::SliderFloat("NRD Max Blur Radius", &g_nrdDenoiser.maxBlurRadius, 0.0f, 60.0f);
			ImGui::SliderFloat("NRD Min Blur Radius", &g_nrdDenoiser.minBlurRadius, 0.0f, 10.0f);
			ImGui::SliderInt("NRD History Frames", (int*)&g_nrdDenoiser.maxAccumulatedFrameNum, 0, 60);
		}

		// Old freeze-and-accumulate scheme (see g_referenceAccumulation's own comment): noisy while
		// the camera moves, converges once it stops. Off by default -- an explicit opt-in reference
		// mode now that TAA/NRD/ReSTIR DI cover ordinary use, not the silent fallback whenever NRD
		// Realtime happens to be off. NRD Realtime (above) still wins if both are on, same as before.
		resetTemporal |= ImGui::Checkbox("Reference Accumulation", &g_referenceAccumulation);

		// Diagnostic: replaces the denoised image with NRD's own REBLUR_VALIDATION debug overlay
		// (only meaningful with VisualizeMode = Default and NRD Realtime Denoise on) -- doesn't
		// itself need a temporal reset, but flipping it usually accompanies changes that do.
		ImGui::Checkbox("NRD Validation Overlay", &g_nrdShowValidation);

		// RTXDI (ReSTIR DI) is no longer wired into rayGenMain -- direct lighting at the primary hit
		// is now handled uniformly by the same per-vertex path (light-picking + optional temporal
		// resampling below) as every other vertex, not a separate RTXDI-gated special case. The
		// RTXDI SDK plumbing itself (g_restirDIContext, g_rtxdiReservoirs/Constants,
		// EnsureRtxdiCapacity) is left allocated but unused rather than torn out in the same pass --
		// see this comment's own history if reviving it is ever worth revisiting.

		// Basic path tracing (ray.slang rayGenMain): one path per pixel, no extra samples, no
		// denoiser -- see shader_io.h's SceneInfo::pathTracingBounces comment. 0 = direct lighting
		// only at the primary hit, N>0 = also trace N further indirect bounces.
		{
			static const uint32 kBounceOptions[] = { 0, 1, 2, 4, 8, 16 };
			int selected = 0;
			for (int i = 0; i < (int)std::size(kBounceOptions); ++i)
				if (kBounceOptions[i] == g_pathTracingBounces)
					selected = i;
			if (ImGui::Combo("Path Tracing Bounces", &selected,
				"0 Bounces\0" "1 Bounce\0" "2 Bounces\0" "4 Bounces\0" "8 Bounces\0" "16 Bounces\0"))
			{
				g_pathTracingBounces = kBounceOptions[selected];
				resetTemporal = true;
			}
		}

		// Temporal resampling of per-vertex path-tracing contributions (see gi_io.h's header
		// comment) -- applies uniformly to direct lighting at the primary hit AND the first
		// indirect bounce, the same way emissive already applies at every vertex; no longer gated
		// by bounce count the way RTXDI DI used to be.
		resetTemporal |= ImGui::Checkbox("Temporal Resampling (direct + indirect)", &g_enableRestirGI);
		if (g_enableRestirGI)
		{
			ImGui::SliderFloat("GI Depth Threshold", &g_giDepthThreshold, 0.0f, 1.0f);
			ImGui::SliderFloat("GI Normal Threshold", &g_giNormalThreshold, 0.0f, 1.0f);
			ImGui::SliderFloat("GI History Weight", &g_giHistoryWeight, 0.0f, 0.98f);
		}

		resetTemporal |= ImGui::Checkbox("Frame Noise Randomization", &g_animateNoise);
#endif
		resetTemporal |= ImGui::Checkbox("Enable Lights", &g_enableLights);
		resetTemporal |= ImGui::Checkbox("Enable Textures", &g_enableTextures);
		resetTemporal |= ImGui::Checkbox("Anti Aliasing (TAA)", &g_enableAA);
		if (g_enableAA)
		{
			ImGui::SliderFloat("TAA History Weight", &g_taa.historyWeight, 0.0f, 0.98f);
			ImGui::SliderFloat("TAA Sharpen", &g_taa.sharpenAmount, 0.0f, 2.0f);
		}
		resetTemporal |= ImGui::Checkbox("Draw Lights", &g_drawLights);
		ImGui::Checkbox("Info HUD", &g_InfoHUD);

		ImGui::Separator();

		if (ImGui::Button("Screenshot"))
		{
			// Scanned lazily, once, from screenshots/ itself (rather than kept in Settings.ini or
			// reset to 0 every launch) so a fresh session's screenshots number on from wherever the
			// last session left off instead of overwriting them -- see readme.md's own "increasing
			// index, screenshot folder, create if needed" todo this satisfies.
			static int nextIndex = -1;
			if (nextIndex < 0)
			{
				nextIndex = 0;
				std::error_code ec;
				for (const auto& entry : std::filesystem::directory_iterator("screenshots", ec))
				{
					int idx = 0;
					if (sscanf_s(entry.path().filename().string().c_str(), "screenshot_%d.png", &idx) == 1)
						nextIndex = std::max(nextIndex, idx + 1);
				}
			}

			char path[64];
			snprintf(path, sizeof(path), "screenshots/screenshot_%04d.png", nextIndex);
			if (CaptureRenderTargetToPNG(m_renderTarget, path))
			{
				LOGI("Screenshot saved: %s", path);
				++nextIndex;
			}
			else
			{
				LOGE("Screenshot failed: %s", path);
			}
		}

		ImGui::Separator();

		if (ImGui::Button("setup #1"))
		{
			resetTemporal |= true;
			setupScene1();
		}
		ImGui::SameLine();
		if (ImGui::Button("setup #2"))
		{
			resetTemporal |= true;
			setupScene2();
		}
		ImGui::SameLine();
		if (ImGui::Button("setup #3"))
		{
			resetTemporal |= true;
			setupScene3();
		}
		ImGui::Separator();
		if(ImGui::Button("load"))
		{
			resetTemporal |= true;
			loadScene();
		}
		ImGui::SameLine();
		if (ImGui::Button("save"))
			saveScene();
		ImGui::SameLine();
		ImGui::InputText("##scene fileName", &g_sceneFileName);
	}
	ImGui::End();

	// setupScene1()/setupScene2()/loadScene() now call this themselves (see ResetTemporalHistory's
	// own comment) -- this covers every OTHER thing that sets resetTemporal (settings sliders,
	// material/texture edits, ...), which still need the same reset but don't go through one of
	// those three functions.
	if (resetTemporal)
		ResetTemporalHistory();

	ImGui::Render();  // This is creating the data to draw the UI (not on GPU yet)

#if USE_RAY_TRACING
	{
		// Rebuilt every frame (not just the first) so a moving/animated scene's instance
		// transforms stay current -- see Renderer::createTopLevelAS's comment for why this is
		// safe to do every frame without stalling the GPU (per-frame-slot buffers, reusing the
		// existing frame-in-flight wait instead of any new fence).
		VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);
		const uint32 frameSlot = m_swapchain.getFrameResourceIndex();

		SetCheckpoint(cmd, "before createTopLevelAS");
		g_renderer.createTopLevelAS(vulkanContext, g_scene, frameSlot);
		updateRayDescriptorSet(frameSlot);
		SetCheckpoint(cmd, "after createTopLevelAS");
	}
#endif

	/*--
	 * - A compute shader is modifying the vertex position
	 * - Draw commands for the triangles are recorded to the command buffer, with target image in the RenderTarget
	-*/
	recordGraphicsCommands(cmd);  // Record the rendering commands for the triangles

	// Make the RenderTarget color writes visible to ImGui's fragment shader read.
	const VkMemoryBarrier2 rtToImGui{
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
	};
	const VkDependencyInfo rtToImGuiDep{
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &rtToImGui,
	};
	vkCmdPipelineBarrier2(cmd, &rtToImGuiDep);

	// Custom TAA resolve (taa_resolve.slang) -- see TaaResolveState (TaaResolve.h). Runs now that
	// m_renderTarget's color is readable (the barrier just above) so it can sample this frame's raw
	// jittered image; its own output (g_taa.history[g_taa.historyIndex]) is what ImGui actually
	// displays this frame (see the ImGui::Image() call, top of this function) and becomes next
	// frame's history. m_shader[0]/[1] != VK_NULL_HANDLE guards against a failed compile (e.g. a
	// syntax error in taa_resolve.slang) the same way every other bind site in this codebase does --
	// binding a null shader object is exactly the class of bug the createGfx() guard fix earlier this
	// session was about.
	if (g_enableAA && g_taa.resolveShader.m_shader[0] != VK_NULL_HANDLE && g_taa.resolveShader.m_shader[1] != VK_NULL_HANDLE
		&& g_taa.renderTargetColorHeapIndex != shaderio::kInvalidTextureId
		&& g_taa.historyHeapIndex[g_taa.historyIndex] != shaderio::kInvalidTextureId)
	{
		SetCheckpoint(cmd, "before taa_resolve");

		shaderio::GraphicsPushData taaPushData{};
		taaPushData.sceneInfoAddress = m_sceneInfoBuffer.address;
		const VkPushDataInfoEXT taaPushDataInfo{
			.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
			.offset = 0,
			.data = {.address = &taaPushData, .size = sizeof(shaderio::GraphicsPushData)},
		};

		const VkViewport taaViewport{0.0f, 0.0f, float(m_viewportSize.width), float(m_viewportSize.height), 0.0f, 1.0f};
		const VkRect2D taaScissor{{0, 0}, m_viewportSize};
		setTaaResolveDynamicState(cmd, taaViewport, taaScissor);

		// Re-bind both heaps: recordGraphicsCommands() (just above) already bound them once for its
		// own draws, but its later dispatchRayTracing() call binds g_ray/g_laser/g_interaction's
		// traditional VkDescriptorSet (vkCmdBindDescriptorSets) in between -- switching the command
		// buffer into "Classic Descriptor" mode, which invalidates whatever heaps were bound under
		// "Descriptor Heap" mode. vkCmdPushDataEXT below switches the command buffer back into heap
		// mode, but only the mode -- it doesn't restore the actual heap buffer bindings that
		// switching away from it dropped, so this draw's fragment shader (SamplerState.Handle(...) in
		// taa_resolve.slang) would otherwise sample through nothing (confirmed live as vkCmdDraw's
		// VUID-vkCmdDraw-None-11308, "a sampler heap was not bound").
		const VkBindHeapInfoEXT taaSamplerHeapBind{
			.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
			.heapRange = {m_samplerHeapBuffer.address, m_samplerHeapSize},
			.reservedRangeOffset = m_samplerReservedOffset,
			.reservedRangeSize = m_samplerReservedSize,
		};
		vkCmdBindSamplerHeapEXT(cmd, &taaSamplerHeapBind);
		const VkBindHeapInfoEXT taaResourceHeapBind{
			.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
			.heapRange = {g_renderer.m_descriptorHeap.m_resourceHeapBuffer.address, g_renderer.m_descriptorHeap.m_resourceHeapSize},
			.reservedRangeOffset = g_renderer.m_descriptorHeap.m_resourceReservedOffset,
			.reservedRangeSize = g_renderer.m_descriptorHeap.m_resourceReservedSize,
		};
		vkCmdBindResourceHeapEXT(cmd, &taaResourceHeapBind);

		// No color/depth attachment at all -- taa_resolve.slang writes via a bindless
		// RWTexture2D.Handle(...) store, not SV_TARGET (see its own header comment). This "draw" only
		// exists to invoke the fragment shader once per pixel.
		const VkRenderingInfo taaRenderingInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = {{0, 0}, m_viewportSize},
			.layerCount = 1,
			.colorAttachmentCount = 0,
			.pColorAttachments = nullptr,
		};
		vkCmdBeginRendering(cmd, &taaRenderingInfo);
		utils::cmdBindGraphicsShaders(cmd, g_taa.resolveShader.m_shader[0], g_taa.resolveShader.m_shader[1]);
		vkCmdPushDataEXT(cmd, &taaPushDataInfo);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		vkCmdEndRendering(cmd);

		// Make this frame's resolved history write visible to ImGui's fragment shader read.
		const VkMemoryBarrier2 taaToImGui{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		};
		const VkDependencyInfo taaToImGuiDep{
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &taaToImGui,
		};
		vkCmdPipelineBarrier2(cmd, &taaToImGuiDep);

		g_taa.historyValid = true;
		SetCheckpoint(cmd, "after taa_resolve");
	}

	// Start rendering to the swapchain
	SetCheckpoint(cmd, "before beginDynamicRenderingToSwapchain");
	beginDynamicRenderingToSwapchain(cmd);
	{
		// The ImGui draw commands are recorded to the command buffer, which includes the display of our RenderTarget image
		SetCheckpoint(cmd, "before ImGui_ImplVulkan_RenderDrawData");
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		SetCheckpoint(cmd, "after ImGui_ImplVulkan_RenderDrawData");
	}
	endDynamicRenderingToSwapchain(cmd);
	SetCheckpoint(cmd, "after endDynamicRenderingToSwapchain");

	SetCheckpoint(cmd, "drawFrame end (recording complete)");
}


void App::endFrame(VkCommandBuffer cmd)
{
	/*--
	 * Prepare to submit the current frame for rendering
	 * First add the swapchain semaphore to wait for the image to be available.
	-*/
	std::vector<VkSemaphoreSubmitInfo> waitSemaphores;
	std::vector<VkSemaphoreSubmitInfo> signalSemaphores;
	waitSemaphores.push_back({
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = m_swapchain.getAcquireSemaphore(),
		.stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		});
	signalSemaphores.push_back({
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = m_swapchain.getPresentSemaphore(),
		.stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		});

	// Get the frame data for the current in-flight slot. The Swapchain owns the
	// frame-resource index; we use it here so both stay in lockstep by construction.
	const uint32 frameSlot = m_swapchain.getFrameResourceIndex();
	auto& frame = m_frameData[frameSlot];

	/*--
	 * Calculate the signal value for when this frame completes
	 * Use monotonic counter that increments by 1 each frame: 1, 2, 3, 4...
	-*/
	const uint64 signalFrameValue = m_frameCounter++;
	frame.lastSignalValue = signalFrameValue;  // Store for next time this frame buffer is used
#ifdef NVVK_SEMAPHORE_DEBUG
	LOGI("SubmitFrame: \t\t slot=%u signalValue=%llu", frameSlot, static_cast<unsigned long long>(signalFrameValue));
#endif

	/*--
	 * Add timeline semaphore to signal when GPU completes this frame
	 * The color attachment output stage is used since that's when the frame is fully rendered
	-*/
	signalSemaphores.push_back({
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = m_frameTimelineSemaphore,
		.value = signalFrameValue,
		.stageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		});

	// Note : in this sample, we only have one command buffer per frame.
	const std::array<VkCommandBufferSubmitInfo, 1> cmdBufferInfo{ {{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.commandBuffer = cmd,
	}} };

	// Populate the submit info to synchronize rendering and send the command buffer
	const std::array<VkSubmitInfo2, 1> submitInfo{ {{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.waitSemaphoreInfoCount = uint32(waitSemaphores.size()),    //
		.pWaitSemaphoreInfos = waitSemaphores.data(),              // Wait for the image to be available
		.commandBufferInfoCount = uint32(cmdBufferInfo.size()),     //
		.pCommandBufferInfos = cmdBufferInfo.data(),               // Command buffer to submit
		.signalSemaphoreInfoCount = uint32(signalSemaphores.size()),  //
		.pSignalSemaphoreInfos = signalSemaphores.data(),            // Signal when rendering is finished
	}} };

	// Submit the command buffer to the GPU and signal when it's done
	VK_CHECK(vkQueueSubmit2(g_renderer.m_context.getGraphicsQueue().queue, uint32(submitInfo.size()), submitInfo.data(), nullptr));

	// Present the image. presentFrame() advances the swapchain's frame-resource
	// index for us, so the next call to prepareFrameResources() will pick up
	// the next slot.
	m_swapchain.presentFrame(g_renderer.m_context.getGraphicsQueue().queue);
}


#if USE_RAY_TRACING
void App::EnsureOutputBufferCapacity(VkCommandBuffer cmd, VkDeviceSize pixelCount)
{
	if (pixelCount <= m_outputBufferCapacityPixels)
		return;

	const uint32 extraData = 1024 * 1024;	// *float4 => 16MB

	EventContext context{ g_renderer.m_context.getDevice(), g_renderer.m_allocator };

	for (Buffer& buf : g_outputBuffer)
	{
		if (buf.m_data.buffer != VK_NULL_HANDLE)
			buf.deinit(context);

		buf.init(g_renderer.m_allocator,
			(pixelCount + extraData) * sizeof(float4),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
		vkCmdFillBuffer(cmd, buf.m_data.buffer, 0, VK_WHOLE_SIZE, 0);
	}

	// See g_outputBufferSpecular's own comment -- ping-ponged like g_outputBuffer above, but plain
	// per-pixel storage only (no +extraData trailing region: nothing ever indexes into this one for
	// the interaction-pick/laser bookkeeping g_outputBuffer's own trailing region holds).
	for (Buffer& buf : g_outputBufferSpecular)
	{
		if (buf.m_data.buffer != VK_NULL_HANDLE)
			buf.deinit(context);

		buf.init(g_renderer.m_allocator,
			pixelCount * sizeof(float4),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
		vkCmdFillBuffer(cmd, buf.m_data.buffer, 0, VK_WHOLE_SIZE, 0);
	}

	m_outputBufferCapacityPixels = pixelCount;
}

App::PickResult App::GetPickResult() const
{
	if (!g_ReadbackMapped)
		return {};

	// Hides the frame-in-flight slot indexing (see g_ReadbackMapped's own comment) -- callers just
	// get "what's under the cursor right now" and don't need to know this is actually a couple
	// frames old or which of Renderer::kMaxFramesInFlight slots it came from.
	const uint32 slot = m_swapchain.getFrameResourceIndex();
	PickResult result;
	memcpy(&result.index, &g_ReadbackMapped[slot * 2 + 0], sizeof(result.index));
	result.isMaterial = g_ReadbackMapped[slot * 2 + 1] != 0;
	return result;
}

void App::EnsureRtxdiCapacity(uint32 width, uint32 height)
{
	if (g_restirDIContext && width <= g_rtxdiCapacityWidth && height <= g_rtxdiCapacityHeight)
		return;

	g_rtxdiCapacityWidth = glm::max(width, g_rtxdiCapacityWidth);
	g_rtxdiCapacityHeight = glm::max(height, g_rtxdiCapacityHeight);

	rtxdi::ReSTIRDIStaticParameters staticParams{};
	staticParams.RenderWidth = g_rtxdiCapacityWidth;
	staticParams.RenderHeight = g_rtxdiCapacityHeight;
	g_restirDIContext = std::make_unique<rtxdi::ReSTIRDIContext>(staticParams);

	// Uniform local-light selection (RTXDI_ENABLE_PRESAMPLING 0 in ray.slang -- our light counts
	// are small, no RIS/ReGIR needed), no BRDF-sample resampling (our point lights aren't
	// traceable geometry a BRDF ray could ever hit, see RAB_TraceRayForLocalLight in
	// RtxdiBridge.slang), pairwise MIS for the fused spatiotemporal resampling pass.
	RTXDI_DIInitialSamplingParameters initialSamplingParams = g_restirDIContext->GetInitialSamplingParameters();
	initialSamplingParams.localLightSamplingMode = ReSTIRDI_LocalLightSamplingMode::Uniform;
	initialSamplingParams.numBrdfSamples = 0;
	g_restirDIContext->SetInitialSamplingParameters(initialSamplingParams);

	// RTXDI_DISpatioTemporalResamplingParameters's own default (GetDefaultReSTIRDISpatioTemporalResamplingParams,
	// ReSTIRDI.cpp) is already Basic; left explicit here as a reminder not to switch to Pairwise
	// without further testing -- Pairwise MIS's code path (RTXDI_DISpatioTemporalResamplingWithPairwiseMIS)
	// reliably caused VK_ERROR_DEVICE_LOST (GPU address fault) in this integration, bisected down to
	// that specific function; Basic gives correct temporal+spatial reuse without it.
	RTXDI_DISpatioTemporalResamplingParameters spatioTemporalParams = g_restirDIContext->GetSpatioTemporalResamplingParameters();
	spatioTemporalParams.biasCorrectionMode = ReSTIRDI_SpatioTemporalBiasCorrectionMode::Basic;
	g_restirDIContext->SetSpatioTemporalResamplingParameters(spatioTemporalParams);

	EventContext context{ g_renderer.m_context.getDevice(), g_renderer.m_allocator };
	VkCommandBuffer rtxdiCmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

	const RTXDI_ReservoirBufferParameters reservoirParams = g_restirDIContext->GetReservoirBufferParameters();
	const VkDeviceSize reservoirBytes = VkDeviceSize(sizeof(RTXDI_PackedDIReservoir)) * reservoirParams.reservoirArrayPitch * rtxdi::c_NumReSTIRDIReservoirBuffers;

	if (g_rtxdiReservoirs.m_data.buffer != VK_NULL_HANDLE)
		g_rtxdiReservoirs.deinit(context);
	g_rtxdiReservoirs.init(g_renderer.m_allocator, reservoirBytes,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	vkCmdFillBuffer(rtxdiCmd, g_rtxdiReservoirs.m_data.buffer, 0, VK_WHOLE_SIZE, 0);

	const VkDeviceSize normalBytes = VkDeviceSize(g_rtxdiCapacityWidth) * g_rtxdiCapacityHeight * sizeof(uint32);
	for (Buffer& buf : g_rtxdiNormals)
	{
		if (buf.m_data.buffer != VK_NULL_HANDLE)
			buf.deinit(context);
		buf.init(g_renderer.m_allocator, normalBytes,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
		vkCmdFillBuffer(rtxdiCmd, buf.m_data.buffer, 0, VK_WHOLE_SIZE, 0);
	}

	if (g_rtxdiConstants.m_data.buffer == VK_NULL_HANDLE)
	{
		g_rtxdiConstants.init(g_renderer.m_allocator, sizeof(shaderio::RtxdiConstants),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	}

	if (g_rtxdiNeighborOffsets.m_data.buffer == VK_NULL_HANDLE)
	{
		const uint32 neighborOffsetCount = g_restirDIContext->GetStaticParameters().NeighborOffsetCount;
		std::vector<uint8_t> rawOffsets(neighborOffsetCount * 2);
		rtxdi::FillNeighborOffsetBuffer(rawOffsets.data(), neighborOffsetCount);

		// RTXDI ships these as signed bytes (Buffer<float2>/RG8_SNORM in the reference samples);
		// decoded to plain float2 here so the shader can read them via the same buffer-device-
		// address raw-pointer idiom as everything else in this codebase (see
		// RTXDI_NEIGHBOR_OFFSETS_BUFFER, ray.slang) instead of a texel buffer view.
		std::vector<float2> offsets(neighborOffsetCount);
		for (uint32 i = 0; i < neighborOffsetCount; ++i)
			offsets[i] = float2((float)(int8_t)rawOffsets[2 * i], (float)(int8_t)rawOffsets[2 * i + 1]) / 127.0f;

		g_rtxdiNeighborOffsets.initRaw(g_renderer.m_allocator, rtxdiCmd, offsets.data(), offsets.size(), sizeof(float2), Buffer::eBT_ShaderBindingTable);
	}

	utils::endSingleTimeCommands(rtxdiCmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
}

void App::EnsureGIReservoirCapacity(uint32 width, uint32 height)
{
	if (g_giReservoirs.m_data.buffer != VK_NULL_HANDLE && width <= g_giCapacityWidth && height <= g_giCapacityHeight)
		return;

	g_giCapacityWidth = glm::max(width, g_giCapacityWidth);
	g_giCapacityHeight = glm::max(height, g_giCapacityHeight);

	EventContext context{ g_renderer.m_context.getDevice(), g_renderer.m_allocator };
	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

	// One buffer, two pitched ping-pong halves -- see g_giReservoirs' own declaration comment for
	// why (mirrors g_rtxdiReservoirs' array-pitch convention rather than g_rtxdiNormals' two-
	// separate-buffers one). Each pixel gets shaderio::GI_RESERVOIR_SLOTS_PER_PIXEL consecutive
	// slots within a half (direct lighting at the primary hit, and the first indirect bounce's own
	// contribution, are different quantities at different vertices -- see gi_io.h's own comment).
	const VkDeviceSize reservoirPitch = VkDeviceSize(g_giCapacityWidth) * g_giCapacityHeight * shaderio::GI_RESERVOIR_SLOTS_PER_PIXEL;
	const VkDeviceSize reservoirBytes = 2 * reservoirPitch * sizeof(shaderio::GIReservoir);

	if (g_giReservoirs.m_data.buffer != VK_NULL_HANDLE)
		g_giReservoirs.deinit(context);
	g_giReservoirs.init(g_renderer.m_allocator, reservoirBytes,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	// Zero-init: GIReservoir::valid == 0 (IsValidGIReservoir() == false) is exactly "nothing here
	// yet", handling the first-ever frame and post-resize growth for free, same as g_rtxdiReservoirs.
	vkCmdFillBuffer(cmd, g_giReservoirs.m_data.buffer, 0, VK_WHOLE_SIZE, 0);

	if (g_giConstants.m_data.buffer == VK_NULL_HANDLE)
	{
		g_giConstants.init(g_renderer.m_allocator, sizeof(shaderio::GIConstants),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	}

	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
}

// Creates one G-buffer render target: a VkImage (STORAGE, for ray.slang's RWTexture2D writes;
// SAMPLED too, for a future denoiser pass to read) + its view, transitioned to GENERAL and cleared
// to black -- same idiom RenderTarget::create() uses for its own color attachments (see
// extern/vk_framework.h), just without RenderTarget's attachment-specific usage bits since these
// targets are never used as render-pass color attachments, only written via compute-style image
// stores from the ray tracing raygen shader.
static void createGBufferTarget(VkCommandBuffer cmd, GBufferTarget& target, VkFormat format, VkExtent2D size, const char* name)
{
	const VkImageCreateInfo imageInfo
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { size.width, size.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	target.image = g_renderer.m_allocator.createImage(imageInfo);
	utils::DebugUtil::getInstance().setObjectName(target.image.image, name);

	const VkImageViewCreateInfo viewInfo
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = target.image.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
	};
	VK_CHECK(vkCreateImageView(g_renderer.m_context.getDevice(), &viewInfo, nullptr, &target.view));

	utils::cmdInitImageLayout(cmd, target.image.image);
	const VkClearColorValue clearValue{ { 0.0f, 0.0f, 0.0f, 0.0f } };
	const VkImageSubresourceRange range{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 };
	vkCmdClearColorImage(cmd, target.image.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
}

static void destroyGBufferTarget(GBufferTarget& target)
{
	if (target.view != VK_NULL_HANDLE)
	{
		vkDestroyImageView(g_renderer.m_context.getDevice(), target.view, nullptr);
		target.view = VK_NULL_HANDLE;
	}
	if (target.image.image != VK_NULL_HANDLE)
	{
		g_renderer.m_allocator.destroyImage(target.image);
		target.image = {};
	}
}

// (Re)creates the two TAA history ping-pong images at grow-only capacity, same philosophy/safety
// requirement as EnsureGBufferCapacity (only ever called from onViewportSizeChange(), which already
// vkDeviceWaitIdle's before this runs, so no in-flight resolve dispatch can still be reading the old
// images when they're destroyed here). Registers both in the bindless descriptor heap as storage
// images (taa_resolve.slang writes via RWTexture2D.Handle(...), not a color attachment -- see
// TaaResolveState::history's own comment, TaaResolve.h) and (re)allocates their ImGui descriptor sets, but does
// NOT call descriptorHeap.build() -- caller's responsibility, since this always runs alongside other
// heap registrations (g_gbufferMotionVector, m_renderTarget's color/depth) that need the same single
// build() call to actually take effect.
static void EnsureTaaHistoryCapacity(uint32 width, uint32 height)
{
	if (g_taa.history[0].image.image != VK_NULL_HANDLE && width <= g_taa.capacityWidth && height <= g_taa.capacityHeight)
		return;

	g_taa.capacityWidth = glm::max(width, g_taa.capacityWidth);
	g_taa.capacityHeight = glm::max(height, g_taa.capacityHeight);
	const VkExtent2D size{ g_taa.capacityWidth, g_taa.capacityHeight };

	for (uint32 i = 0; i < 2; ++i)
	{
		g_taa.imTextureHistory[i].deinit();
		destroyGBufferTarget(g_taa.history[i]);
	}

	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	for (uint32 i = 0; i < 2; ++i)
	{
		createGBufferTarget(cmd, g_taa.history[i], VK_FORMAT_R8G8B8A8_UNORM, size, i == 0 ? "TAA-History-0" : "TAA-History-1");
		// Registered twice under the same VkImage (already both STORAGE_BIT and SAMPLED_BIT, see
		// createGBufferTarget): once as a storage image for taa_resolve.slang's RWTexture2D.Handle(...)
		// write, once as a sampled image so reading back *last* frame's history (the other ping-pong
		// slot, reprojected by a fractional motion vector) can bilinear-filter -- a storage image only
		// supports point Load(), which would look noticeably worse under sub-pixel motion.
		//
		// Same slot kept across recreations via updateTexture()/updateStorageImage() (add* only the
		// first time) as m_renderTarget's color/depth registration (onViewportSizeChange) -- this can
		// also run more than once in a row (ImGui's docking layout settling growing the viewport
		// repeatedly), and add*() again each time would leave a stale entry behind pointing at the
		// image this same iteration just destroyed above -- see DescriptorHeap.h's updateTexture()
		// comment for the crash that caused.
		if (g_taa.historyHeapIndex[i] == shaderio::kInvalidTextureId)
		{
			g_taa.historyHeapIndex[i] = g_renderer.m_descriptorHeap.addStorageImage(g_taa.history[i].image.image, VK_FORMAT_R8G8B8A8_UNORM);
			g_taa.historySampledHeapIndex[i] = g_renderer.m_descriptorHeap.addTexture(g_taa.history[i].image.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL);
		}
		else
		{
			g_renderer.m_descriptorHeap.updateStorageImage(g_taa.historyHeapIndex[i], g_taa.history[i].image.image, VK_FORMAT_R8G8B8A8_UNORM);
			g_renderer.m_descriptorHeap.updateTexture(g_taa.historySampledHeapIndex[i], g_taa.history[i].image.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL);
		}
		g_taa.imTextureHistory[i].init(g_taa.history[i].view, VK_IMAGE_LAYOUT_GENERAL);
	}
	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);

	// Freshly-cleared images, no meaningful history in either slot yet.
	g_taa.historyValid = false;
}

// (Re)creates the 4 G-buffer targets at grow-only capacity, same philosophy/safety requirement as
// EnsureRtxdiCapacity (only ever called from init()/onViewportSizeChange(), both of which the
// caller already knows are safe points to free a resource an in-flight ray dispatch might still
// read -- see onViewportSizeChange's vkDeviceWaitIdle).
void App::EnsureGBufferCapacity(uint32 width, uint32 height)
{
	if (g_gbufferAlbedo.image.image != VK_NULL_HANDLE && width <= g_gbufferCapacityWidth && height <= g_gbufferCapacityHeight)
		return;

	g_gbufferCapacityWidth = glm::max(width, g_gbufferCapacityWidth);
	g_gbufferCapacityHeight = glm::max(height, g_gbufferCapacityHeight);
	const VkExtent2D size{ g_gbufferCapacityWidth, g_gbufferCapacityHeight };

	destroyGBufferTarget(g_gbufferAlbedo);
	destroyGBufferTarget(g_gbufferNormalRoughness);
	destroyGBufferTarget(g_gbufferViewZ);
	destroyGBufferTarget(g_gbufferMotionVector);

	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	createGBufferTarget(cmd, g_gbufferAlbedo, VK_FORMAT_R8G8B8A8_UNORM, size, "GBuffer-Albedo");
	createGBufferTarget(cmd, g_gbufferNormalRoughness, VK_FORMAT_R16G16B16A16_SNORM, size, "GBuffer-NormalRoughness");
	createGBufferTarget(cmd, g_gbufferViewZ, VK_FORMAT_R32_SFLOAT, size, "GBuffer-ViewZ");
	createGBufferTarget(cmd, g_gbufferMotionVector, VK_FORMAT_R16G16_SFLOAT, size, "GBuffer-MotionVector");
	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);

	// Bindless registration for taa_resolve.slang's own read of this (ray.slang writes it via its
	// separate traditional RT descriptor binding, unrelated to this heap index) -- kept at the same
	// heap slot across recreations via updateTexture() (addTexture() only the first time): this
	// function only runs when growing past capacity, but that can still happen more than once in a
	// row (e.g. ImGui's docking layout settling), and addTexture() again each time would leave a
	// stale entry behind pointing at the just-destroyed previous image -- see DescriptorHeap.h's own
	// comment on updateTexture() for the crash that caused (this exact pattern, discovered via
	// m_renderTarget's color/depth registration below in onViewportSizeChange). Caller
	// (onViewportSizeChange) still owns the actual descriptorHeap.build() call, alongside its own
	// m_renderTarget color/depth registration.
	if (g_taa.gbufferMotionVectorHeapIndex == shaderio::kInvalidTextureId)
		g_taa.gbufferMotionVectorHeapIndex = g_renderer.m_descriptorHeap.addTexture(g_gbufferMotionVector.image.image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL);
	else
		g_renderer.m_descriptorHeap.updateTexture(g_taa.gbufferMotionVectorHeapIndex, g_gbufferMotionVector.image.image, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL);
}
#endif

void App::onViewportSizeChange(VkExtent2D size)
{
	g_temporalAccumulator = 0;

	m_viewportSize = size;
	// Recreate the RenderTarget to the size of the viewport.
	// Full device wait, not just this queue: we're about to free g_outputBuffer below, and a
	// queue-only wait doesn't guarantee no other in-flight work (e.g. deferred host operations
	// for ray tracing) still references it.
	VK_CHECK(vkDeviceWaitIdle(g_renderer.m_context.getDevice()));

	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
		m_renderTarget.update(cmd, m_viewportSize);
		m_imTexture.update(m_renderTarget.getUiImageView());
		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

#if USE_RAY_TRACING
	// g_outputBuffer is sized for the viewport (one float4 per pixel) but was only ever allocated
	// once at startup. Without resizing it here, the ray gen shader's output[index] = color writes
	// past the end of the buffer whenever the viewport grows -- an out-of-bounds GPU write.
	//
	// Grow-only: during startup ImGui's docking layout can report a few different viewport sizes
	// in quick succession as it settles, and each resize here means a full device-wide wait plus a
	// buffer destroy/recreate -- expensive GPU-resource churn concentrated in the least stable part
	// of the frame sequence. Only reallocate when the new size actually exceeds current capacity;
	// the ray gen shader indexes with the *current* m_viewportSize each frame, so writes stay in
	// bounds as long as width*height fits in whatever was last allocated, regardless of aspect ratio.
	const VkDeviceSize neededPixels = VkDeviceSize(m_viewportSize.width) * m_viewportSize.height;
	// DIAGNOSTIC: log every viewport resize so a real run shows the sequence of sizes
	// ImGui's docking layout reports right after startup (see comment above), and whether
	// g_outputBuffer gets reallocated smaller than the true initial window size before
	// growing back -- that gap would leave a brief window where m_outputBufferCapacityPixels
	// understates what was really needed.
	if(false)
	LOGI("onViewportSizeChange: new size %ux%u (%llu px), current capacity %llu px, will %sreallocate",
		size.width, size.height, (unsigned long long)neededPixels,
		(unsigned long long)m_outputBufferCapacityPixels, (neededPixels > m_outputBufferCapacityPixels) ? "" : "NOT ");
	if (neededPixels > m_outputBufferCapacityPixels)
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
		EnsureOutputBufferCapacity(cmd, neededPixels);
		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}
	// Safe here (unlike inside updateSceneBuffer(), where this used to live): vkDeviceWaitIdle at
	// the top of this function guarantees no in-flight ray dispatch still references the old
	// g_rtxdiReservoirs/g_rtxdiNormals before EnsureRtxdiCapacity frees/reallocates them.
	EnsureRtxdiCapacity(m_viewportSize.width, m_viewportSize.height);
	EnsureGIReservoirCapacity(m_viewportSize.width, m_viewportSize.height);
	EnsureGBufferCapacity(m_viewportSize.width, m_viewportSize.height);
	g_nrdDenoiser.ensureCapacity(m_viewportSize.width, m_viewportSize.height,
		g_gbufferNormalRoughness.view, g_gbufferViewZ.view, g_gbufferAlbedo.view);
#endif

	// Custom TAA resolve (taa_resolve.slang) -- outside USE_RAY_TRACING since it applies to the
	// rasterizer too. History images are grow-only (see EnsureTaaHistoryCapacity); m_renderTarget's
	// color/depth heap registration below runs on every call because RenderTarget::update() (above,
	// unlike the grow-only resources here) recreates on any size change, invalidating whatever views
	// were previously registered -- updateTexture() (not addTexture() again) keeps this from leaving
	// a stale entry behind every time, see its own comment (DescriptorHeap.h) for why that mattered
	// in practice (confirmed live: onViewportSizeChange firing more than once during ImGui's docking
	// layout settling crashed build() on the second call).
	EnsureTaaHistoryCapacity(m_viewportSize.width, m_viewportSize.height);
	if (g_taa.renderTargetColorHeapIndex == shaderio::kInvalidTextureId)
		g_taa.renderTargetColorHeapIndex = g_renderer.m_descriptorHeap.addTexture(
			m_renderTarget.getColorImage(), m_renderTarget.getColorFormat(), VK_IMAGE_LAYOUT_GENERAL);
	else
		g_renderer.m_descriptorHeap.updateTexture(g_taa.renderTargetColorHeapIndex,
			m_renderTarget.getColorImage(), m_renderTarget.getColorFormat(), VK_IMAGE_LAYOUT_GENERAL);
	if (g_taa.renderTargetDepthHeapIndex == shaderio::kInvalidTextureId)
		g_taa.renderTargetDepthHeapIndex = g_renderer.m_descriptorHeap.addTexture(
			m_renderTarget.getDepthImage(), m_renderTarget.getDepthFormat(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_DEPTH_BIT);
	else
		g_renderer.m_descriptorHeap.updateTexture(g_taa.renderTargetDepthHeapIndex,
			m_renderTarget.getDepthImage(), m_renderTarget.getDepthFormat(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_DEPTH_BIT);
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
		g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);
		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}
	g_taa.historyValid = false;
	// Same reasoning as g_taa.historyValid just above, for ReSTIR GI's own reservoir history (see
	// GIConstants::historyValid's comment, gi_io.h) -- a resize changes the pixel layout even when
	// the underlying buffer didn't need to grow.
	g_giHistoryValid = false;
}

void App::beginDynamicRenderingToSwapchain(VkCommandBuffer cmd) const
{
	// Image to render to
	const std::array<VkRenderingAttachmentInfo, 1> colorAttachment{ {{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_swapchain.getImageView(),
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,   // Clear the image (see clearValue)
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,  // Store the image (keep the image)
		.clearValue = {{{0.0f, 0.0f, 0.0f, 1.0f}}},
	}} };

	// Details of the dynamic rendering
	const VkRenderingInfo renderingInfo{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {{0, 0}, m_windowSize},
		.layerCount = 1,
		.colorAttachmentCount = uint32(colorAttachment.size()),
		.pColorAttachments = colorAttachment.data(),
	};

	// Transition the swapchain image to general layout for use as a render target in dynamic rendering
	utils::cmdTransitionSwapchainToRenderingLayout(cmd, m_swapchain.getImage());

	vkCmdBeginRendering(cmd, &renderingInfo);
}

void App::endDynamicRenderingToSwapchain(VkCommandBuffer cmd)
{
	vkCmdEndRendering(cmd);

	// Transition the swapchain image back to the present layout
	utils::cmdTransitionSwapchainToPresentLayout(cmd, m_swapchain.getImage());
}

// [0] is always the single global light; the rest are every point light the loaded BSP level's
// "light" entities parsed into (see BSPEntityParser::ProcessEntity). Rebuilt from scratch every
// frame -- this list is small, and the global light can move/change color live via the UI, so
// there's nothing worth caching. Sets g_lightCount; App::updateSceneBuffer() reads it
// (and g_lights.m_data.address) into SceneInfo right after this runs.
static void updateLights(VkCommandBuffer cmd)
{
	std::vector<shaderio::Light> lights;
	lights.reserve(1 + g_BSPLevel.GetLightCount());

	// The BSP level's own transform state lives entirely on its Entity (g_scene.m_sceneObjects,
	// found below by matching hndRenderComp against g_BSPLevel's own handle) -- no separate cached
	// copy of it (there used to be one, g_BSPTransform, kept in sync by hand at the one place the
	// entity was created; this reads the live value instead, so it can't ever go stale, e.g. if the
	// entity is ever moved/rescaled via the gizmo). Defaults to identity if the BSP entity doesn't
	// exist this frame (nothing loaded it yet) -- GetLightCount() is 0 in that case too, so the
	// bsp-lights loop below is a no-op regardless of what this defaults to.
	Transform bspTransform;

	// todo: optimize
	for (uint32 it = 0, count = g_scene.m_sceneObjects.size(); it < count; ++it)
	{
		const auto& entity = g_scene.m_sceneObjects.GetElementByIterator(it);

		if(const auto* p = g_LightManager.GetElementPtr(entity.hndLight))
		{
			const auto& ref = *p;
			shaderio::Light& global = lights.emplace_back();
			global.position = entity.transform.GetPos();
			global.color = ref.m_lightColor * ref.m_lightIntensity;
			// The global light has no natural "radius" of its own (it was flat/unattenuated before g_lights
			// existed) -- generous enough to stay close to its old unattenuated look across the small demo
			// scene, while still actually falling off (see EvalLightFalloff, mathlib.slang) instead of
			// contributing at full strength from arbitrarily far away like every light used to.
			global.radius = 50.0f;
			// Reuses the existing "lightLength" UI slider (g_lightLength, already meant as this light's
			// physical extent for the old tube-light debug visualization) as its soft-shadow source size.
			global.sourceRadius = ref.m_lightLength * 0.5f;
		}

		// IsLoaded() guard first: getHandle() asserts if Load() never ran (e.g. app startup, or
		// setup#1's non-BSP demo scene) -- confirmed live as an assert firing every frame whenever
		// no BSP level is loaded, since this loop runs regardless of which scene setup is active.
		if (g_BSPLevel.IsLoaded() && entity.hndRenderComp == g_BSPLevel.getHandle())
			bspTransform = entity.transform;
	}

	// g_BSPLevel.m_Lights are in the level file's own local space; bspTransform (above) is the same
	// transform (position/rotation/scale) applied to the level's Entity (see setupScene2()), so its
	// full object matrix -- not Transform::TransformPosition(), which drops scale -- is what places
	// these lights where the (rotated, 0.01-scaled) level geometry actually ended up in the world.
	const glm::mat4 bspObjectMatrix = bspTransform.GetObjectMatrix();
	// A light's radius is a distance, so it has to shrink by the same factor positions do, or every
	// light ends up reaching far more of the (now much smaller) level than it should -- abs() because
	// setupScene2()'s SetScale mirrors X (negative), and a negative radius would make EvalLightFalloff's
	// max(radius, 0.001f) floor collapse every BSP light's falloff window to ~1mm, i.e. effectively
	// zero everywhere (this was mistaken for a shadow-ray bug: the only unaffected light was the
	// global one, whose radius is a hardcoded positive constant, not derived from this scale).
	const float bspScale = glm::abs(bspTransform.GetScale().x);

	for (uint32 i = 0; i < g_BSPLevel.GetLightCount(); ++i)
	{
		const SBSPLight* bspLight = g_BSPLevel.GetLightNo(i);
		shaderio::Light& dst = lights.emplace_back();
		dst.position = float3(bspObjectMatrix * float4(bspLight->m_vOrigin, 1.0f));
		// m_LightColor is a normalized 0..1 color -- scaled up a bit so a light is clearly visible
		// right at its own position (falloff below takes it to 0 by m_fAttenuationRadius anyway).
		dst.color = bspLight->m_LightColor * 3.0f;
		// Quake's "light" entity value is really an attenuation radius already -- use it directly
		// (scaled, see above) instead of folding it into color, so a level's worth of lights
		// (dozens to hundreds) don't all contribute at full strength everywhere.
		dst.radius = bspLight->m_fAttenuationRadius * bspScale;
		// Physical light-source size for soft shadows (see SBSPLight::m_fSourceRadius) -- a
		// distance, like m_fAttenuationRadius above, so it's scaled the same way.
		dst.sourceRadius = bspLight->m_fSourceRadius * bspScale;
	}

	if (lights.size() > shaderio::kMaxLights)
	{
		LOGE("g_lights has %zu lights but shaderio::kMaxLights is %u -- truncating", lights.size(), shaderio::kMaxLights);
		lights.resize(shaderio::kMaxLights);
	}

	// Light picking used to need a CPU-built power CDF here (PickImportanceSampledLight, ray.slang)
	// so the fixed shadow-ray budget could favor brighter lights; RTXDI (ReSTIR DI) now does that
	// job on the GPU via spatiotemporal resampling instead, so g_lights only ever needs to carry
	// the plain Light array.
	if (g_lights.m_data.buffer == VK_NULL_HANDLE)
	{
		g_lights.init(g_renderer.m_allocator,
			sizeof(shaderio::Light) * shaderio::kMaxLights,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	}

	const VkDeviceSize lightsSize = lights.size() * sizeof(shaderio::Light);

	// Same barrier/update/barrier pattern as updateSceneBuffer() below, which runs right after this
	// and reads g_lights.m_data.address into SceneInfo. vkCmdUpdateBuffer() requires dataSize > 0
	// (VUID-vkCmdUpdateBuffer-dataSize-00033) -- skip entirely with 0 lights (e.g. g_LightManager
	// is empty and no BSP level is loaded either). g_lightCount below still correctly ends up 0,
	// and every shader-side use already guards on lightCount > 0 (see ray.slang), so nothing reads
	// g_lights' buffer contents in that case.
	if (lightsSize > 0)
	{
		utils::cmdBufferMemoryBarrier(cmd, g_lights.m_data.buffer,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		vkCmdUpdateBuffer(cmd, g_lights.m_data.buffer, 0, lightsSize, lights.data());
		utils::cmdBufferMemoryBarrier(cmd, g_lights.m_data.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
	}

	g_lightCount = (uint32)lights.size();
}

// Calculate the radical inverse of an index in a given base
// @return 0..1
static float HaltonSequence(uint32 index, uint32 base)
{
	float result = 0.0f;
	float f = 1.0f / (float)base;
	uint32 i = index;
	while (i > 0)
	{
		result += f * (float)(i % base);
		i /= base;
		f /= (float)base;
	}
	return result;
}
// Generate 2D jitter offset using Halton(2, 3)
// Pass frameIndex (e.g., totalFrames % 8 or 16) and screen dimensions
// @return 0..1
static float2 ComputeTemporalJitter(uint32 frameIndex)
{
	return float2(HaltonSequence(frameIndex + 1, 2), HaltonSequence(frameIndex + 1, 3));
}

void App::updateSceneBuffer(VkCommandBuffer cmd) const
{
	// Updating the data for the frame
	shaderio::SceneInfo sceneInfo{};
	float time = static_cast<float>(ImGui::GetTime());
	float sineValue = std::sin(time);    // Get the sine of the current time
	float mappedValue = sineValue;
#if USE_RAY_TRACING
	// Ping-pong by frame parity: this frame writes g_outputBuffer[currentIndex] (outputAddress)
	// while reading last frame's result from the other slot (prevOutputAddress) for temporal
	// blending -- see rayGenMain (ray.slang). m_frameCounter is stable for the whole frame (it
	// only changes at submit time, in endFrame()), so both addresses stay consistent throughout
	// this frame's recording.
	{
		const uint32 currentIndex = (uint32)(m_frameCounter & 1);
		sceneInfo.outputAddress = g_outputBuffer[currentIndex].m_data.address;
		sceneInfo.prevOutputAddress = g_outputBuffer[1 - currentIndex].m_data.address;
		sceneInfo.outputSpecularAddress = g_outputBufferSpecular[currentIndex].m_data.address;
		sceneInfo.prevOutputSpecularAddress = g_outputBufferSpecular[1 - currentIndex].m_data.address;
	}
#endif
	sceneInfo.animValue = mappedValue;
	sceneInfo.resolution = float2(m_viewportSize.width, m_viewportSize.height);
	sceneInfo.numData = 0; // todo remove
	sceneInfo.texId = m_imageID;
	sceneInfo.visualizeMode = (uint32)g_visualizeMode;
	sceneInfo.clearColor = g_skyRGB * g_skyIntensity;
	sceneInfo.fogColor = g_fogColor;
	sceneInfo.fogDensity = g_fogDensity;
	sceneInfo.fogAsymmetry = g_fogAsymmetry;
	sceneInfo.laserPos = g_laserPos;
	sceneInfo.laserAngle = g_laserAngle * PI / 180;
	sceneInfo.laserIntensity = g_laserIntensity;
	sceneInfo.exposureMul = powf(2.0f, g_ExposureLog);
	sceneInfo.enableTonemapper = g_enableTonemapper ? 1 : 0;
	sceneInfo.lightmapIntensity = g_lightmapIntensity;
	sceneInfo.pxMousePos = g_pxMousePos;
	sceneInfo.mouseButtons = g_mouseButtons;

	sceneInfo.enableIBL = (!g_flatSkyColor && g_ibl.isValid()) ? 1 : 0;
	sceneInfo.iblSpecularHeapIndex = g_ibl.getSpecularHeapIndex();
	sceneInfo.iblDiffuseHeapIndex = g_ibl.getDiffuseHeapIndex();
	sceneInfo.iblSpecularMipCount = g_ibl.getSpecularMipCount();

	float zNear = 0.1f;
	float zFar = 100000.0f;

	// in degrees
	float fieldOfView = 45.0f;
	Transform transform;

	if (auto* p2 = g_scene.m_sceneObjects.GetElementPtr(g_currentCamera))
		if (auto* p = g_CameraManager.GetElementPtr(p2->hndCamera))
		{
			fieldOfView = p->fieldOfView;
			transform = p2->transform;
		}

	sceneInfo.clipFromView = glm::perspectiveLH_ZO(glm::radians(fieldOfView), m_viewportSize.width / (float)m_viewportSize.height, zNear, zFar);	// LH
	sceneInfo.fieldOfViewDeg = fieldOfView;

	sceneInfo.worldFromView = transform.GetViewMatrix();
	sceneInfo.viewFromWorld = glm::transpose(sceneInfo.worldFromView);
	sceneInfo.viewFromClip = glm::inverse(sceneInfo.clipFromView);
	// worldFromView actually holds the *view* matrix (world-to-view; see its assignment above --
	// name is misleading but ray-gen's camera-ray reconstruction already depends on viewFromWorld
	// being transpose(that value) exactly as-is, so it can't be repointed to a true inverse here).
	// worldFromClip needs the real view-to-world matrix, so invert it separately, then transpose
	// to match this buffer's other matrices (read via explicit row indexing / MatrixVectorMultiply,
	// which needs the transpose to line up with GLM's column-major storage).
	// TEMP DIAGNOSTIC: shader.rast.slang is temporarily reading this via MatrixVectorMultiply
	// instead of raw mul(), to isolate whether this C++-side matrix is correct independent of the
	// mul()/majorness mismatch. Revert together with the shader-side change once confirmed.
	sceneInfo.worldFromClip = glm::transpose(glm::inverse(sceneInfo.worldFromView) * sceneInfo.viewFromClip);

	sceneInfo.wsCameraPos = transform.GetPos();
#if USE_RAY_TRACING
	// Cache raw (non-transposed) current/previous view+proj matrices for g_nrdDenoiser.dispatch()
	// (dispatchRayTracing()'s call site, further down this file) -- see g_nrdWorldToView's own
	// comment for why this can't just reuse RTXDI's g_rtxdiPrevViewFromWorld/g_rtxdiPrevClipFromView.
	g_nrdPrevWorldToView = g_nrdWorldToView;
	g_nrdPrevViewToClip = g_nrdViewToClip;
	g_nrdWorldToView = sceneInfo.worldFromView;
	g_nrdViewToClip = sceneInfo.clipFromView;

	// Previous frame's camera matrices for taa_resolve.slang's own rasterizer-mode reprojection (see
	// SceneInfo::prevViewFromWorld's comment, shader_io.h) -- g_nrdPrevWorldToView/g_nrdPrevViewToClip
	// just above already hold exactly this (raw, non-transposed), cached the same way for NRD; row-
	// major transpose to match every other matrix in this buffer.
	sceneInfo.prevViewFromWorld = glm::transpose(g_nrdPrevWorldToView);
	sceneInfo.prevClipFromView = glm::transpose(g_nrdPrevViewToClip);

	// Custom TAA resolve (taa_resolve.slang) -- see TaaResolveState (TaaResolve.h). Heap
	// indices are stable across frames (only re-registered on resize, see onViewportSizeChange/
	// EnsureGBufferCapacity), so this just copies the cached globals; only which history slot is
	// "read" vs "write" actually flips, by g_taa.historyIndex (decided once, at the top of drawFrame(),
	// before ImGui::Image() needed to already know it).
	sceneInfo.taaCurrentColorHeapIndex = g_taa.renderTargetColorHeapIndex;
	sceneInfo.taaDepthHeapIndex = g_taa.renderTargetDepthHeapIndex;
	sceneInfo.taaMotionVectorHeapIndex = g_taa.gbufferMotionVectorHeapIndex;
	sceneInfo.taaHistoryWriteHeapIndex = g_taa.historyHeapIndex[g_taa.historyIndex];
	// Sampled (filtered), not storage -- see g_taa.historySampledHeapIndex's own comment.
	sceneInfo.taaHistoryReadHeapIndex = g_taa.historySampledHeapIndex[1 - g_taa.historyIndex];
	sceneInfo.taaHistoryValid = g_taa.historyValid ? 1 : 0;
	sceneInfo.taaHistoryCapacityWidth = g_taa.capacityWidth;
	sceneInfo.taaHistoryCapacityHeight = g_taa.capacityHeight;
	sceneInfo.taaHistoryWeight = g_taa.historyWeight;
	sceneInfo.taaSharpenAmount = g_taa.sharpenAmount;

//	uint32 frameSeed = (uint32)m_frameCounter;
	uint32 frameSeed = (uint32)m_frameCounter % 32;	// more stable noise

	sceneInfo.rayTracing = g_rayTracing;
	sceneInfo.pathTracingBounces = g_pathTracingBounces;
	// frame-varying when animation is on (0 is a valid frame counter value, but multiplying by
	// a nonzero odd constant only maps 0 -> 0 -- harmless, just one frame indistinguishable
	// from "off", which self-corrects the very next frame), always 0 when off
	sceneInfo.frameNoiseSeed = g_animateNoise ? frameSeed * 0x9e3779b9u : 0u;

	// NRD wants a fresh, single-sample-per-pixel noisy signal every frame (its own temporal
	// accumulation replaces this scheme's running-sum-while-static one) -- force a permanent
	// "just reset" state instead of ever incrementing g_temporalAccumulator while realtime mode is
	// on, so rayGenMain (ray.slang) always writes this frame's raw trace result, never a blended sum.
	// Same when Reference Accumulation itself is off (g_referenceAccumulation's own comment): TAA +
	// ReSTIR DI are the ones responsible for stability then, not this scheme.
	sceneInfo.temporalAccumulator = (g_nrdRealtimeMode || !g_referenceAccumulation) ? 0 : g_temporalAccumulator;
	sceneInfo.temporalMixIn = (sceneInfo.temporalAccumulator == MAX_TEMPORAL_ACCUM) ? 0.0f : 1.0f;
	sceneInfo.invTemporalAccumulator = 1.0f / (sceneInfo.temporalAccumulator + 1);
	sceneInfo.frameIndex = g_frameIndex;

	{
		// AA jitter is intentionally on in every mode (Reference accumulation and NRD Realtime
		// alike), per-visualize-mode debug views excepted. It used to be excluded from NRD Realtime
		// mode: this engine's own freeze-and-accumulate "Reference" scheme resolves jitter by
		// averaging many jittered samples over time, and Realtime mode forces temporalAccumulator to
		// 0 (a single fresh sample every frame, see above) so NRD denoises a fresh signal each frame
		// -- with nothing here to resolve the jitter's own per-frame sub-pixel geometric aliasing
		// into, that combination previously made edges visibly flicker between sub-pixel samples
		// (confirmed live, previously masked by a separate REBLUR bug that blurred the whole signal
		// including albedo). The fix now is to let NRD do that resolving instead: it's a temporal
		// denoiser built to accept camera jitter as an explicit input (nrdParams.cameraJitterX/Y
		// below) precisely so its own motion-vector-based reprojection/history can integrate jittered
		// samples the same way this engine's Reference accumulation does. If jitter flicker
		// reappears in Realtime mode, that means NRD isn't actually consuming this jitter signal
		// correctly (e.g. history reset, motion vectors, or reprojection settings) -- the fix belongs
		// there, not in reintroducing this exclusion.
		const bool doJitter = (g_visualizeMode == 0) && g_enableAA;
		sceneInfo.temporalJitter = doJitter ? ComputeTemporalJitter(g_frameIndex) : float2(0.5f);
		sceneInfo.temporalJitterPrev = (doJitter && g_frameIndex > 0) ? ComputeTemporalJitter(g_frameIndex - 1) : float2(0.5f);
	}

	if (!g_nrdRealtimeMode && g_referenceAccumulation)
	{
		++g_temporalAccumulator;
		if (g_temporalAccumulator > MAX_TEMPORAL_ACCUM)
			g_temporalAccumulator = MAX_TEMPORAL_ACCUM;
		if (g_visualizeMode != 0)
			g_temporalAccumulator = 0;
	}
	else
	{
		g_temporalAccumulator = 0;
	}
#endif
	sceneInfo.infoHUD = g_InfoHUD;
	sceneInfo.drawLights = g_drawLights;
	sceneInfo.zNear = zNear;
	sceneInfo.zFar = zFar;
	sceneInfo.lightsAddress = g_lights.m_data.address;
	sceneInfo.lightCount = g_enableLights ? g_lightCount : 0;
	sceneInfo.enableTextures = g_enableTextures;
#if USE_RAY_TRACING
	sceneInfo.enableRtxdi = g_enableRtxdi;
	sceneInfo.enableRestirGI = g_enableRestirGI;
#endif

#if USE_RAY_TRACING
	// RTXDI (ReSTIR DI): fill and upload this frame's RtxdiConstants (shaders/rtxdi_io.h) -- read
	// by ray.slang via RayPushData::rtxdiConstantsAddress (set in dispatchRayTracing()). Buffer
	// (re)allocation happens in EnsureRtxdiCapacity(), called from onViewportSizeChange()/init()
	// only -- NOT here, since those are the only call sites where it's safe to free/reallocate
	// g_rtxdiReservoirs/g_rtxdiNormals without racing a still-executing previous frame's ray
	// dispatch that might still be reading them (see EnsureRtxdiCapacity's own comment).
	{
		ASSERT(g_restirDIContext); // EnsureRtxdiCapacity() must have run at least once by now

		g_restirDIContext->SetFrameIndex(sceneInfo.frameIndex);

		// Live-tunable reuse thresholds (Settings window / g_restirDepthThreshold and friends, see
		// their own comment) -- applied every frame rather than only where the context itself gets
		// (re)created (EnsureRtxdiCapacity), since a live slider needs to take effect immediately,
		// not just on the next resize. Cheap: this is a plain struct write, no GPU resource churn.
		{
			RTXDI_DISpatioTemporalResamplingParameters spatioTemporalParams = g_restirDIContext->GetSpatioTemporalResamplingParameters();
			spatioTemporalParams.depthThreshold = g_restirDepthThreshold;
			spatioTemporalParams.normalThreshold = g_restirNormalThreshold;
			spatioTemporalParams.maxHistoryLength = (uint32_t)g_restirMaxHistoryLength;
			g_restirDIContext->SetSpatioTemporalResamplingParameters(spatioTemporalParams);
		}

		shaderio::RtxdiConstants rtxdiConstants{};
		rtxdiConstants.runtimeParams = g_restirDIContext->GetRuntimeParams();
		rtxdiConstants.reservoirBufferParams = g_restirDIContext->GetReservoirBufferParameters();
		rtxdiConstants.initialSamplingParams = g_restirDIContext->GetInitialSamplingParameters();
		rtxdiConstants.spatioTemporalParams = g_restirDIContext->GetSpatioTemporalResamplingParameters();
		rtxdiConstants.reservoirBufferAddress = g_rtxdiReservoirs.m_data.address;
		rtxdiConstants.neighborOffsetsAddress = g_rtxdiNeighborOffsets.m_data.address;

		// Ping-pong by frame parity, same as g_outputBuffer: this frame writes the "current" slot
		// (normals and the reservoir array index alike) while reading last frame's from the other.
		const uint32 rtxdiCurrentIndex = (uint32)(m_frameCounter & 1);
		rtxdiConstants.normalAddress = g_rtxdiNormals[rtxdiCurrentIndex].m_data.address;
		rtxdiConstants.prevNormalAddress = g_rtxdiNormals[1 - rtxdiCurrentIndex].m_data.address;
		rtxdiConstants.inputBufferIndex = 1 - rtxdiCurrentIndex;
		rtxdiConstants.outputBufferIndex = rtxdiCurrentIndex;

		rtxdiConstants.prevViewFromWorld = g_rtxdiPrevViewFromWorld;
		rtxdiConstants.prevClipFromView = g_rtxdiPrevClipFromView;
		g_rtxdiPrevViewFromWorld = sceneInfo.viewFromWorld;
		g_rtxdiPrevClipFromView = sceneInfo.clipFromView;

		utils::cmdBufferMemoryBarrier(cmd, g_rtxdiConstants.m_data.buffer,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		vkCmdUpdateBuffer(cmd, g_rtxdiConstants.m_data.buffer, 0, sizeof(shaderio::RtxdiConstants), &rtxdiConstants);
		utils::cmdBufferMemoryBarrier(cmd, g_rtxdiConstants.m_data.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
	}

	// ReSTIR GI (temporal-only indirect-bounce resampling): fill and upload this frame's
	// GIConstants (shaders/gi_io.h) -- read by ray.slang via RayPushData::giConstantsAddress (set
	// in dispatchRayTracing()). Buffer (re)allocation happens in EnsureGIReservoirCapacity(),
	// called from onViewportSizeChange()/init() only -- same in-flight-safety reasoning as
	// EnsureRtxdiCapacity's own comment. Reuses the SAME frame-parity bit RTXDI's own
	// rtxdiCurrentIndex above derives from (m_frameCounter & 1) -- not RTXDI-specific data, just
	// frame parity, so this stays decoupled from RTXDI DI's own state rather than reading it back.
	{
		ASSERT(g_giReservoirs.m_data.buffer != VK_NULL_HANDLE); // EnsureGIReservoirCapacity() must have run at least once by now

		shaderio::GIConstants giConstants{};
		giConstants.reservoirBufferAddress = g_giReservoirs.m_data.address;
		giConstants.reservoirPitch = g_giCapacityWidth * g_giCapacityHeight * shaderio::GI_RESERVOIR_SLOTS_PER_PIXEL;
		const uint32 giCurrentIndex = (uint32)(m_frameCounter & 1);
		giConstants.inputBufferIndex = 1 - giCurrentIndex;
		giConstants.outputBufferIndex = giCurrentIndex;
		giConstants.historyWeight = g_giHistoryWeight;
		giConstants.depthThreshold = g_giDepthThreshold;
		giConstants.normalThreshold = g_giNormalThreshold;
		// Valid from the frame AFTER this one onward -- this frame's own GI dispatch (if any) is
		// about to write real data into the reservoir buffer for the first time since the last
		// reset, so by the time next frame reads it back, it's safe to reproject.
		giConstants.historyValid = g_giHistoryValid ? 1 : 0;
		g_giHistoryValid = true;

		utils::cmdBufferMemoryBarrier(cmd, g_giConstants.m_data.buffer,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		vkCmdUpdateBuffer(cmd, g_giConstants.m_data.buffer, 0, sizeof(shaderio::GIConstants), &giConstants);
		utils::cmdBufferMemoryBarrier(cmd, g_giConstants.m_data.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
	}
#endif

	// Add a barrier to make sure nothing was writing to it, before updating its content
	utils::cmdBufferMemoryBarrier(cmd, m_sceneInfoBuffer.buffer,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	// Update the buffer with the new data
	vkCmdUpdateBuffer(cmd, m_sceneInfoBuffer.buffer, 0, sizeof(shaderio::SceneInfo), &sceneInfo);
	// Add barrier to make sure the buffer is updated before the shaders use it
	utils::cmdBufferMemoryBarrier(cmd, m_sceneInfoBuffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

	g_frameIndex++;
}

#if USE_RAY_TRACING
void App::updateRayDescriptorSet(uint32 frameSlot)
{
	g_ray.updateRayDescriptorSet(g_renderer.m_context.getDevice(), g_renderer.m_tlasSlots[frameSlot].tlas,
		m_renderTarget.getColorImageView(), m_sceneInfoBuffer.buffer, g_renderer.m_rtTextureViews,
		shaderio::kMaxRayTracingTextures, m_rtSampler, frameSlot,
		g_gbufferAlbedo.view, g_gbufferNormalRoughness.view, g_gbufferViewZ.view, g_gbufferMotionVector.view,
		g_ibl.getSpecularCubeView());

	g_laser.updateRayDescriptorSet(g_renderer.m_context.getDevice(), g_renderer.m_tlasSlots[frameSlot].tlas,
		m_renderTarget.getColorImageView(), m_sceneInfoBuffer.buffer, g_renderer.m_rtTextureViews,
		shaderio::kMaxRayTracingTextures, m_rtSampler, frameSlot,
		g_gbufferAlbedo.view, g_gbufferNormalRoughness.view, g_gbufferViewZ.view, g_gbufferMotionVector.view,
		g_ibl.getSpecularCubeView());

	g_interaction.updateRayDescriptorSet(g_renderer.m_context.getDevice(), g_renderer.m_tlasSlots[frameSlot].tlas,
		m_renderTarget.getColorImageView(), m_sceneInfoBuffer.buffer, g_renderer.m_rtTextureViews,
		shaderio::kMaxRayTracingTextures, m_rtSampler, frameSlot,
		g_gbufferAlbedo.view, g_gbufferNormalRoughness.view, g_gbufferViewZ.view, g_gbufferMotionVector.view,
		g_ibl.getSpecularCubeView());
}

void App::dispatchRayTracing(VkCommandBuffer cmd)
{
	// Nothing to trace with an empty scene (e.g. before "setup #1" is ever clicked) -- matching
	// Renderer::createTopLevelAS()'s own early-out, which then never builds a TLAS this frame.
	// Dispatching anyway would use g_ray/g_laser's descriptor set with its g_tlas binding never
	// written (updateRayDescriptorSet() also skips on a null TLAS), which the validation layer
	// catches as VUID-vkCmdTraceRaysKHR-None-08114 ("descriptor... has never been updated").
	if (g_scene.m_sceneObjects.empty())
		return;

	// Scene-level, shared by every ray tracing shader dispatched below: the per-mesh
	// {IB, VB, primitiveMaterials} address buffer only depends on the loaded meshes, not on any
	// one shader, so it's built once here rather than once per shader -- once per FRAME, that is:
	// rebuilt every frame (like g_materialBuffer below), not gated on "already built once".
	// g_MeshManager's element count can grow after the first build (e.g. clicking "setup #1"
	// again, which doesn't reset the scene) -- a stale, too-small g_meshBuffer left over from an
	// earlier, smaller mesh count would leave newer meshes' TLAS instances indexing past its end
	// in the ray tracing shader, an out-of-bounds GPU read (observed as VK_ERROR_DEVICE_LOST).
	{
		uint32 count = g_MeshManager.GetElementCount();

		// Nothing to build with an empty scene (e.g. before "setup #1" is ever clicked) -- leave
		// g_meshBuffer null rather than calling initRaw() with an empty data() (null pointer,
		// asserted against in Buffer.cpp). Left null here, it's simply left out of this frame's
		// rayPushData below; there's nothing for a ray tracing dispatch to read it for anyway with
		// 0 meshes (0 TLAS instances, see Renderer::createTopLevelAS's own empty-scene early-out).
		if (count)
		{
			VkCommandBuffer meshCmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

			// uint64 for buffer device address, *3 for IB, VB, and the per-triangle material
			// override buffer (0 if the mesh has no submeshes, e.g. g_plane -- see
			// Mesh::m_primitiveMaterials and rayClosestHitMain in ray.slang)
			std::vector<uint64> data(count * 3);

			for (uint32 i = 0; i < count; ++i)
			{
				const auto& el = g_MeshManager.GetElementByIterator(i);
				data[i * 3 + 0] = el.m_indices.m_data.address;
				data[i * 3 + 1] = el.m_vertices.m_data.address;
				data[i * 3 + 2] = el.m_primitiveMaterials.m_data.address;
			}

			// need to be outside vkCmdBeginRendering/vkCmdEndRendering
			g_meshBuffer.initRaw(g_renderer.m_allocator, meshCmd, data.data(), data.size(), sizeof(*data.data()), Buffer::eBT_ShaderBindingTable);

			utils::endSingleTimeCommands(meshCmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
		}
	}

	// g_entityMeshIndexBuffer: entityIndex -> meshIndex, the indirection a closest-hit shader needs
	// since InstanceID() (Renderer::createTopLevelAS's asInstance.instanceCustomIndex) is now the
	// entity's index, not the mesh's -- see RayPushData::entityMeshIndexAddress's comment. Packed by
	// entity index (scene.m_sceneObjects' Elements[] position, i.e. GetHandleByIterator(i).Index) --
	// createTopLevelAS (Renderer.cpp) sets instanceCustomIndex to that exact same Index, so this
	// buffer must be sized/addressed by it too, NOT by iteration order (an earlier version of this
	// code did `data[i]`, matching the wrong, unused 2-argument createTopLevelAS overload still
	// sitting in Renderer.cpp -- the real one, taking a frameSlot, is what App.cpp
	// actually calls, and uses entityHnd.Index).
	{
		extern CObjManager<CRenderCompHandle, RenderComponent> g_MeshManager;

		// entityHnd.Index only equals its iteration position when g_scene.m_sceneObjects has never
		// had Clear() called on a non-empty instance (see GetHandleByIterator()'s own comment) --
		// both setupScene1() and setupScene2() do exactly that on entry, so switching between them
		// (or loading a smaller scene right after a larger live one) recycles slots with Index values
		// past GetElementCount(). Sized by GetCapacity() (Elements.size(), fixed at InitOnce(4096))
		// instead, so any valid Index -- not just ones below the *current* entity count -- fits;
		// confirmed live as an out-of-bounds write/VK_ERROR_DEVICE_LOST when this was GetElementCount().
		uint32 capacity = g_scene.m_sceneObjects.GetCapacity();

		// Nothing to build with an empty scene, same reasoning as g_meshBuffer above.
		if (!g_scene.m_sceneObjects.empty())
		{
			VkCommandBuffer entityCmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

			std::vector<uint32> data(capacity, ~0u);
			for (uint32 i = 0, count = g_scene.m_sceneObjects.GetElementCount(); i < count; ++i)
			{
				const CEntityHandle hnd = g_scene.m_sceneObjects.GetHandleByIterator(i);
				const Entity& el = g_scene.m_sceneObjects.GetElementByIterator(i);

				// Must mirror Renderer::createTopLevelAS's own mesh resolution exactly (including its
				// g_lightMesh fallback for a light with no mesh of its own, e.g. setupScene1()'s hand-
				// placed light) -- that function still builds a real TLAS instance (instanceCustomIndex
				// = this entity's index) for such an entity once g_lightMesh becomes valid, but this
				// buffer left it at the ~0u sentinel, so the closest-hit shader's g_meshBuffer[~0u * 3]
				// lookup read wildly out of bounds. Confirmed live as an assert/VK_ERROR_DEVICE_LOST
				// specifically when clicking "setup #2" (the only thing that loads g_lightMesh) after
				// "setup #1" (the only thing that creates a mesh-less light entity) -- neither alone
				// triggers it.
				CRenderCompHandle effectiveMesh = el.hndRenderComp;
				if (!effectiveMesh.IsValid() && g_LightManager.GetElementPtr(el.hndLight))
					effectiveMesh = g_lightMesh;

				if (effectiveMesh.IsValid())
				{
					// Same invariant Renderer::createTopLevelAS used to assert directly (mesh index
					// must equal that mesh's position in g_MeshManager's own iteration order) -- now
					// checked here instead, where the mesh index is actually looked up.
					ASSERT(effectiveMesh.Index < g_MeshManager.GetElementCount());
					ASSERT(hnd.Index < capacity);
					data[hnd.Index] = effectiveMesh.Index;
				}
			}

			g_entityMeshIndexBuffer.initRaw(g_renderer.m_allocator, entityCmd, data.data(), data.size(), sizeof(*data.data()), Buffer::eBT_ShaderBindingTable);

			utils::endSingleTimeCommands(entityCmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
		}
	}

	// Scene-level push data (sceneInfo/mesh/material buffer addresses) shared by every ray tracing
	// shader dispatch below. outputAddress/prevOutputAddress are per-dispatch though: a shader other
	// than g_ray can copy this struct and override those two fields to target a different buffer
	// while still sharing sceneInfoAddress/meshBufferAddress/materialBufferAddress. g_ray itself
	// ignores these two (it reads SceneInfo::outputAddress/prevOutputAddress instead, set in
	// updateSceneBuffer()); they're populated here so they're ready for a shader that doesn't.
	shaderio::RayPushData rayPushData{};
	rayPushData.sceneInfoAddress = m_sceneInfoBuffer.address;  // Points to the GPU buffer updated above
	rayPushData.meshBufferAddress = g_meshBuffer.m_data.address;
	rayPushData.entityMeshIndexAddress = g_entityMeshIndexBuffer.m_data.address;
	rayPushData.materialBufferAddress = g_materialBuffer.m_data.address;
	rayPushData.rtxdiConstantsAddress = g_rtxdiConstants.m_data.address;
	rayPushData.giConstantsAddress = g_giConstants.m_data.address;
	// 0 (null) here just means an empty scene (no meshes yet, e.g. before "setup #1" is ever
	// clicked) -- fine, since there's then nothing for a ray tracing dispatch to read it for.
	{
		const uint32 currentIndex = (uint32)(m_frameCounter & 1);
		rayPushData.outputAddress = g_outputBuffer[currentIndex].m_data.address;
		rayPushData.prevOutputAddress = g_outputBuffer[1 - currentIndex].m_data.address;
	}

	// DIAGNOSTIC: every ray tracing shader's ray gen writes output[pixel.y * dispatchWidth + pixel.x]
	// into g_outputBuffer, which only has room for m_outputBufferCapacityPixels pixels (see
	// onViewportSizeChange). If a dispatch ever exceeds that, it's an out-of-bounds GPU write --
	// log it loudly so it shows up in <exe>.txt even when the process dies from the resulting
	// VK_ERROR_DEVICE_LOST before this frame's LOGI calls would otherwise be noticed.
	{
		const VkDeviceSize dispatchPixels = VkDeviceSize(m_viewportSize.width) * m_viewportSize.height;
		if (dispatchPixels > m_outputBufferCapacityPixels)
		{
			LOGE("vkCmdTraceRaysKHR about to dispatch %ux%u (%llu px) but g_outputBuffer only has capacity for %llu px -- out-of-bounds write incoming",
				m_viewportSize.width, m_viewportSize.height, (unsigned long long)dispatchPixels, (unsigned long long)m_outputBufferCapacityPixels);
		}
	}

	const uint32 frameSlot = m_swapchain.getFrameResourceIndex();

	if (!g_laser.dispatchRayTracing(cmd, frameSlot, g_renderer.m_transientCmdPool, {4, 4}, rayPushData))
		g_UIErrors.push_back("g_laser failed to compile");

	if (!g_ray.dispatchRayTracing(cmd, frameSlot, g_renderer.m_transientCmdPool, m_viewportSize, rayPushData))
		g_UIErrors.push_back("g_ray failed to compile");

	// Mouse-picking: a single ray, dispatched at a fixed {1,1} size regardless of viewport
	// resolution -- see interaction.slang. Shares rayPushData as-is (including
	// outputAddress/prevOutputAddress) since its result is written into the same buffer g_ray
	// writes its own pixel output to, just past it (kInteractionResultOffset, shader_io.h).
	if (!g_interaction.dispatchRayTracing(cmd, frameSlot, g_renderer.m_transientCmdPool, {1, 1}, rayPushData))
		g_UIErrors.push_back("g_interaction failed to compile");

	// Copy the picked-object index + isMaterial flag toward the CPU -- see g_ReadbackMapped's own
	// comment for why this is safe to read back a couple frames from now without ever stalling the
	// GPU. Barrier from the ray tracing write to this transfer read: srcAccessMask is passed
	// explicitly since utils::inferAccessMaskFromStage doesn't recognize the ray tracing shader
	// stage (dst is left to infer -- TRANSFER_BIT is a stage it does know).
	if (g_ReadbackMapped)
	{
		const uint32 currentIndex = (uint32)(m_frameCounter & 1);
		VkBuffer srcBuffer = g_outputBuffer[currentIndex].m_data.buffer;

		// interactionResult[0].w (the 4th float, byte 12, of the float4 at kInteractionResultOffset)
		// immediately followed by interactionResult[1].x (byte 0 of the NEXT float4, byte 16 overall)
		// -- contiguous, see interaction.slang's own result[0]/result[1] writes, so one 8-byte copy
		// grabs both.
		const VkDeviceSize srcOffset =
			(VkDeviceSize(m_viewportSize.width) * m_viewportSize.height + shaderio::kInteractionResultOffset) * sizeof(float4)
			+ 3 * sizeof(float);
		const VkDeviceSize dstOffset = frameSlot * 2 * sizeof(uint32);
		const VkDeviceSize copySize = 2 * sizeof(uint32);

		utils::cmdBufferMemoryBarrier(cmd, srcBuffer,
			VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_ACCESS_2_SHADER_WRITE_BIT, 0, srcOffset, copySize);

		const VkBufferCopy copyRegion{ .srcOffset = srcOffset, .dstOffset = dstOffset, .size = copySize };
		vkCmdCopyBuffer(cmd, srcBuffer, g_ReadbackBuffer.buffer, 1, &copyRegion);
	}

	// future: e.g.
	// shaderio::RayPushData ray2PushData = rayPushData;
	// ray2PushData.outputAddress = ...;  // a different buffer than g_outputBuffer
	// ray2PushData.prevOutputAddress = ...;
	// if (!g_ray2.dispatchRayTracing(cmd, frameSlot, m_transientCmdPool, m_viewportSize, ray2PushData))
	//     g_UIErrors.push_back("g_ray2 failed to compile");
}
#endif

void App::recordGraphicsCommands(VkCommandBuffer cmd)
{
	DBG_VK_SCOPE(cmd);  // <-- Helps to debug in Nsight

	SetCheckpoint(cmd, "recordGraphicsCommands start");

	// Update g_lights before the scene buffer, which points SceneInfo::lightsAddress/lightCount at
	// it (cannot be done inside dynamic rendering)
	updateLights(cmd);
	updateSceneBuffer(cmd);

	const VkDeviceSize offsets[] = { 0 };
	const VkDeviceSize sizes[] = { VK_WHOLE_SIZE };
	const VkViewport   viewport{ 0.0F, 0.0F, float(m_viewportSize.width), float(m_viewportSize.height), 0.0F, 1.0F };
	const VkRect2D     scissor{ {0, 0}, m_viewportSize };

	/*--
	 * Prepare push data: with descriptor heap, the pipeline layout is VK_NULL_HANDLE so we cannot
	 * use traditional push constants or push descriptors. Instead, vkCmdPushDataEXT sends small
	 * per-draw data to the shader's push_constant block. Push data carries only:
	 *   - The buffer device address of the SceneInfo buffer (updated once per frame above)
	 *   - The per-draw triangle color (changes between draw calls)
	 * The shader reads SceneInfo from GPU memory via buffer reference, keeping push data small.
	-*/
	shaderio::GraphicsPushData pushData{};
	pushData.sceneInfoAddress = m_sceneInfoBuffer.address;  // Points to the GPU buffer updated above
	// materialBufferAddress is filled in below, after g_materialBuffer's lazy init (~30 lines down)
	// runs: on the very first call to this function g_materialBuffer doesn't exist yet, so reading
	// its address here would capture 0 -- which the fragment shader then dereferences
	// ((MaterialData*)pushData.materialBufferAddress + materialID in shader.rast.slang), causing a
	// null-pointer read fault on frame 1 only (frame 2+ self-heals since the buffer already exists
	// by then). This was a real, reproduced VK_ERROR_DEVICE_LOST right after startup.
	pushData.modelMatrix = glm::mat4(1.0f);

	// VkPushDataInfoEXT describes where in push_constant space to write and the CPU data to upload
	VkPushDataInfoEXT pushDataInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
		.offset = 0,
		.data = {.address = &pushData, .size = sizeof(shaderio::GraphicsPushData)},
	};

	VkClearColorValue clearColor;
	clearColor.float32[0] = g_skyRGB.r;
	clearColor.float32[1] = g_skyRGB.g;
	clearColor.float32[2] = g_skyRGB.b;
	clearColor.float32[3] = 1.0f;

	// Image to render to
	const std::array<VkRenderingAttachmentInfo, 1> colorAttachment{ {{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_renderTarget.getColorImageView(),
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,   // Clear the image (see clearValue)
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,  // Store the image (keep the image)
		.clearValue = {{clearColor}},
	}} };

	// Depth buffer to use
	const VkRenderingAttachmentInfo depthAttachment{
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = m_renderTarget.getDepthImageView(),
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,   // Clear depth buffer
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,  // Store depth buffer
		//    .clearValue  = {{{1.0f, 0}}},
			.clearValue = {{{0.0f, 0}}},
	};

	// Details of the dynamic rendering
	const VkRenderingInfo renderingInfo{
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {{0, 0}, m_renderTarget.getSize()},
		.layerCount = 1,
		.colorAttachmentCount = uint32(colorAttachment.size()),
		.pColorAttachments = colorAttachment.data(),
		.pDepthAttachment = &depthAttachment,
	};

	// Bind the descriptor heap buffers so shaders can access textures and samplers by index.
	// This replaces the traditional vkCmdBindDescriptorSets2 call that was used for the texture descriptor set.
	// The sampler heap provides sampler descriptors (linear, nearest, etc.) and the resource heap provides image descriptors (loaded textures).
	// Each VkBindHeapInfoEXT describes the heap buffer address, total size, and where the mandatory reserved range sits within the buffer.
		const VkBindHeapInfoEXT samplerHeapBind{
			.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
			.heapRange = {m_samplerHeapBuffer.address, m_samplerHeapSize},
			.reservedRangeOffset = m_samplerReservedOffset,
			.reservedRangeSize = m_samplerReservedSize,
		};
		vkCmdBindSamplerHeapEXT(cmd, &samplerHeapBind);

		const VkBindHeapInfoEXT resourceHeapBind{
			.sType = VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT,
			.heapRange = {g_renderer.m_descriptorHeap.m_resourceHeapBuffer.address, g_renderer.m_descriptorHeap.m_resourceHeapSize},
			.reservedRangeOffset = g_renderer.m_descriptorHeap.m_resourceReservedOffset,
			.reservedRangeSize = g_renderer.m_descriptorHeap.m_resourceReservedSize,
		};
		vkCmdBindResourceHeapEXT(cmd, &resourceHeapBind);

//		if (g_materialBuffer.m_data.buffer == VK_NULL_HANDLE)
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

		uint32 count = g_MaterialManager.GetElementCount();

		// uint64 for buffer device address, *2 for IB and VB
		// at least 1 entry even with 0 materials (empty scene) -- an empty vector's data() can be
		// null, and Buffer::initRaw() asserts its data pointer is non-null (Buffer.cpp:24); nothing
		// ever indexes into this dummy entry since there's nothing to draw with 0 materials either.
		std::vector<shaderio::MaterialData> data(std::max<uint32>(count, 1));

		for (uint32 i = 0; i < count; ++i)
		{
			shaderio::MaterialData& dst = data[i];
			const auto src = g_MaterialManager.GetElementByIterator(i);

			// ray.slang's g_textures[] (bound from g_renderer.m_rtTextureViews, see
			// updateRayDescriptorSet()) only actually holds a live view for indices this session's
			// Texture::uploadPixels() has (re)written since the last time loadScene() cleared the
			// whole array -- a heap index whose view is still VK_NULL_HANDLE here would read a
			// destroyed/never-written slot on the GPU (silent corruption or VK_ERROR_DEVICE_LOST)
			// instead of this clean assert.
			Texture* tex = g_TextureManager.GetElementPtr(src.m_hndTexture);
			dst.textureId = tex ? tex->getHeapIndex() : shaderio::kInvalidTextureId;
			ASSERT(dst.textureId == shaderio::kInvalidTextureId || dst.textureId >= shaderio::kMaxRayTracingTextures
				|| g_renderer.m_rtTextureViews[dst.textureId] != VK_NULL_HANDLE);


			Texture* mrTex = g_TextureManager.GetElementPtr(src.m_hndMetallicRoughnessTexture);
			dst.metallicRoughnessTextureId = mrTex ? mrTex->getHeapIndex() : shaderio::kInvalidTextureId;
			ASSERT(dst.metallicRoughnessTextureId == shaderio::kInvalidTextureId || dst.metallicRoughnessTextureId >= shaderio::kMaxRayTracingTextures
				|| g_renderer.m_rtTextureViews[dst.metallicRoughnessTextureId] != VK_NULL_HANDLE);

			Texture* lmTex = g_TextureManager.GetElementPtr(src.m_hndLightmap);
			dst.lightmapTextureId = lmTex ? lmTex->getHeapIndex() : shaderio::kInvalidTextureId;
			ASSERT(dst.lightmapTextureId == shaderio::kInvalidTextureId || dst.lightmapTextureId >= shaderio::kMaxRayTracingTextures
				|| g_renderer.m_rtTextureViews[dst.lightmapTextureId] != VK_NULL_HANDLE);

			// See g_hndDefaultNormalMap's own comment -- unlike the textures above (which fall back
			// to kInvalidTextureId, a sentinel the shader branches on), a material with no normal map
			// of its own resolves straight to the flat (0,0,1) engine default's heap index instead,
			// so the shader can always sample this field unconditionally.
			Texture* normalTex = g_TextureManager.GetElementPtr(src.m_hndNormalMap);
			Texture* defaultNormalTex = g_TextureManager.GetElementPtr(g_hndDefaultNormalMap);
			ASSERT(defaultNormalTex); // created once in init(), before any material is ever filled
			dst.normalMapTextureId = normalTex ? normalTex->getHeapIndex() : defaultNormalTex->getHeapIndex();
			ASSERT(dst.normalMapTextureId >= shaderio::kMaxRayTracingTextures
				|| g_renderer.m_rtTextureViews[dst.normalMapTextureId] != VK_NULL_HANDLE);

			dst.albedoColor = src.m_albedoColor;
			dst.emissiveColor = src.m_emissiveColor;
			dst.emissiveIntensity = src.m_emissiveIntensity;
			dst.roughness = src.m_roughness;
			dst.metallic = src.m_metallic;
			dst.albedoTextureFade = src.m_albedoTextureFade;
			dst.specularOcclusionFromAlbedo = src.m_specOcclusionFromAlbedo;
		}

		// need to be outside vkCmdBeginRendering/vkCmdEndRendering
		g_materialBuffer.initRaw(g_renderer.m_allocator, cmd, data.data(), data.size(), sizeof(*data.data()), Buffer::eBT_ShaderBindingTable);

		utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

	// g_materialBuffer is guaranteed valid from here on (just-created above on frame 1, or already
	// existing on later frames); see the comment where pushData was constructed.
	pushData.materialBufferAddress = g_materialBuffer.m_data.address;
	ASSERT(pushData.materialBufferAddress);

#if USE_RAY_TRACING
	// Gates ray tracing as a whole; dispatchRayTracing() explicitly dispatches each ray tracing
	// shader by name and surfaces its own compile-readiness failures individually.
	if (g_rayTracing)
	{
		dispatchRayTracing(cmd);

		// NRD (g_nrdDenoiser, NrdIntegration.h) denoises this frame's fresh, single-sample
		// g_outputBuffer slot in place -- skip on an empty scene the same way dispatchRayTracing()
		// itself already does (nothing was written this frame to denoise). Also skip outside
		// EVM_Shaded: every other visualize mode repurposes g_outputBuffer to hold its own raw
		// debug value (albedo, normals, IDs, ...) for that pixel, not noisy lighting -- running NRD's
		// spatial/temporal filter over that data blurs/smears already-deterministic debug output for
		// no benefit (confirmed live: EVM_AlbedoColor visibly softened/ghosted under camera motion).
		if (g_nrdRealtimeMode && !g_scene.m_sceneObjects.empty() && g_visualizeMode == 0)
		{
			NrdDenoiser::DispatchParams nrdParams{};
			nrdParams.viewportSize = m_viewportSize;
			nrdParams.frameSlot = m_swapchain.getFrameResourceIndex();
			nrdParams.frameIndex = g_frameIndex - 1;  // updateSceneBuffer() already incremented it for this frame
			nrdParams.gbufferMotionVectorView = g_gbufferMotionVector.view;
			nrdParams.gbufferNormalRoughnessView = g_gbufferNormalRoughness.view;
			nrdParams.gbufferViewZView = g_gbufferViewZ.view;
			nrdParams.gbufferAlbedoView = g_gbufferAlbedo.view;
			nrdParams.worldToView = glm::value_ptr(g_nrdWorldToView);
			nrdParams.worldToViewPrev = glm::value_ptr(g_nrdPrevWorldToView);
			nrdParams.viewToClip = glm::value_ptr(g_nrdViewToClip);
			nrdParams.viewToClipPrev = glm::value_ptr(g_nrdPrevViewToClip);

			// Matches updateSceneBuffer()'s own sceneInfo.temporalJitter formula exactly (see its
			// comment) so NRD is told the same jitter that was actually applied to this frame's
			// primary ray -- it needs the real, nonzero value now that jitter is on in Realtime mode
			// too, to reproject/accumulate jittered samples via its own motion-vector-based history.
			const bool doJitter = (g_visualizeMode == 0) && g_enableAA;
			const float2 jitter01 = doJitter ? ComputeTemporalJitter(nrdParams.frameIndex) : float2(0.5f);
			const float2 jitter01Prev = (doJitter && nrdParams.frameIndex > 0)
				? ComputeTemporalJitter(nrdParams.frameIndex - 1) : float2(0.5f);
			nrdParams.cameraJitterX = jitter01.x - 0.5f;
			nrdParams.cameraJitterY = jitter01.y - 0.5f;
			nrdParams.cameraJitterPrevX = jitter01Prev.x - 0.5f;
			nrdParams.cameraJitterPrevY = jitter01Prev.y - 0.5f;

			const uint32 currentIndex = (uint32)(m_frameCounter & 1);
			nrdParams.outputBufferAddress = g_outputBuffer[currentIndex].m_data.address;
			nrdParams.outputSpecularBufferAddress = g_outputBufferSpecular[currentIndex].m_data.address;
			nrdParams.showValidation = g_nrdShowValidation;

			g_nrdDenoiser.dispatch(cmd, nrdParams);
		}
	}
#endif

	// Begin dynamic rendering
	SetCheckpoint(cmd, "before vkCmdBeginRendering (offscreen graphics pass)");
	vkCmdBeginRendering(cmd, &renderingInfo);

	// ---- Dynamic state ----------------------------------------------------
	// With shader objects, every piece of state that used to live in the
	// graphics pipeline must be set explicitly with a vkCmdSet*EXT call before
	// each draw. We set everything once here since none of it changes between
	// the two draws below.
	setGraphicsDynamicState(cmd, viewport, scissor);

	bool bGfx = g_gfx.m_shader[0] && g_gfx.m_shader[1];

	if(!bGfx)
		g_UIErrors.push_back("g_Gfx failed to compile");

	// todo
#if USE_RAY_TRACING
	if (g_rayTracing)
		bGfx = false;
#endif

	if(bGfx)
	{
		SetCheckpoint(cmd, "before main scene draw loop");
		ASSERT(g_gfx.m_shader[0]);
		ASSERT(g_gfx.m_shader[1]);
		// Bind shader objects: vertex stage (shared) + the no-texture fragment stage.
	//    utils::cmdBindGraphicsShaders(cmd, m_vertShader, m_fragShaderNoTexture);
		utils::cmdBindGraphicsShaders(cmd, g_gfx.m_shader[0], g_gfx.m_shader[1]);

//		for (auto it = g_scene.m_sceneObjects.begin(), end = g_scene.m_sceneObjects.end(); it != end; ++it)
		for(uint32 i = 0, count = g_scene.m_sceneObjects.size(); i < count; ++i)
		{
			const Entity& ref = g_scene.m_sceneObjects.GetElementByIterator(i);
//			const Entity& ref = *it;

			// Bind the vertex buffer. vkCmdBindVertexBuffers2 (Vulkan 1.3 core) extends the
			// older vkCmdBindVertexBuffers with optional pSizes and pStrides arrays. With
			// shader objects, vertex input layout is dynamic (set via vkCmdSetVertexInputEXT
			// above), so passing explicit sizes/strides here is fine.
			//
			// A stale handle here is expected after a scene reload: saveMeshes() only serializes
			// file-loaded meshes (see Mesh::m_loadFromFile), so a scene object referencing a
			// procedural mesh (e.g. g_plane) has nothing to resolve to once g_MeshManager is reset
			// -- matching Renderer::createTopLevelAS()'s equivalent lookup/skip for the ray tracing
			// path (Renderer.cpp), instead of dereferencing null.
			//
			// GetElementPtr() only compares Seed, not whether the slot is actually in use -- a
			// stale handle whose Seed happens to match a never-used slot's default Seed (0, see
			// CObjManager::InitOnce) still "resolves" to a non-null Mesh**, but the Mesh* it points
			// to is the slot's default value, nullptr (g_MeshManager's T is Mesh*, unlike the
			// value-typed managers) -- so the pointee must be checked too, not just the pointer.
			RenderComponent* mesh = g_MeshManager.GetElementPtr(ref.hndRenderComp);
			if (!mesh)
				continue;
			if (!mesh->m_vertices.m_data.buffer)
				continue;

			vkCmdBindVertexBuffers2(cmd, 0, 1, &mesh->m_vertices.m_data.buffer, offsets, sizes, nullptr);
			vkCmdBindIndexBuffer(cmd, mesh->m_indices.m_data.buffer, 0, VK_INDEX_TYPE_UINT32);

			pushData.modelMatrix = glm::transpose(ref.transform.GetObjectMatrix());

			// One draw call per submesh range, each with its own material (see Mesh::SubMesh) --
			// every mesh always has at least one submesh now (createPlane()/Mesh::load()/
			// 3DBSPLevel::Load() all populate it), so there's no more single-material-per-instance
			// fallback path.
			ASSERT(!mesh->m_subMeshes.empty());
			for (auto subIt = mesh->m_subMeshes.begin(), subEnd = mesh->m_subMeshes.end(); subIt != subEnd; ++subIt)
			{
				const RenderComponent::SubMesh& sub = *subIt;

				pushData.materialAndMeshID = short2Uint(sub.hndMaterial.Index, ref.hndRenderComp.Index);
				vkCmdPushDataEXT(cmd, &pushDataInfo);
				vkCmdDrawIndexed(cmd, sub.indexCount, 1, sub.indexStart, 0, 0);
			}
		}
	}

	vkCmdPushDataEXT(cmd, &pushDataInfo);

	// s2h_3d #################################

//	utils::cmdBindGraphicsShaders(cmd, g_s2h_3d.m_shader[0], g_s2h_3d.m_shader[1]);
//	vkCmdDraw(cmd, 3, 1, 0, 0);  // 3 vertices

	// post #################################

	SetCheckpoint(cmd, "before post.slang draw");

	if(g_post.m_shader[0] && g_post.m_shader[1])
	{
		utils::cmdBindGraphicsShaders(cmd, g_post.m_shader[0], g_post.m_shader[1]);
		vkCmdDraw(cmd, 3, 1, 0, 0);  // 3 vertices
	}
	else g_UIErrors.push_back("g_post failed to compile");

	SetCheckpoint(cmd, "after post.slang draw");

	vkCmdEndRendering(cmd);
	SetCheckpoint(cmd, "after vkCmdEndRendering (offscreen pass)");
}

void App::initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(m_window, true);
	static VkFormat           imageFormats[] = { m_swapchain.getImageFormat() };
	ImGui_ImplVulkan_InitInfo initInfo = {
			  .Instance = g_renderer.m_context.getInstance(),
			  .PhysicalDevice = g_renderer.m_context.getPhysicalDevice(),
			  .Device = g_renderer.m_context.getDevice(),
			  .QueueFamily = g_renderer.m_context.getGraphicsQueue().familyIndex,
			  .Queue = g_renderer.m_context.getGraphicsQueue().queue,
			  .DescriptorPool = m_uiDescriptorPool,
			  .MinImageCount = 2,
			  .ImageCount = m_swapchain.getImageCount(),
			  .PipelineInfoMain =
			{
					  .PipelineRenderingCreateInfo =  // Dynamic rendering
				{
						  .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
						  .colorAttachmentCount = 1,
						  .pColorAttachmentFormats = imageFormats,
				},
			},
			  .PipelineInfoForViewports = initInfo.PipelineInfoMain,
			  .UseDynamicRendering = true,
	};

	ImGui_ImplVulkan_Init(&initInfo);

	// ViewportsEnable (not DockingEnable -- a different, easily-confused flag) lets popups/
	// windows escape the main OS window and become separate platform windows, each rendered
	// through its own swapchain via ImGui's *default* Vulkan backend
	// (extern/imgui/backends/imgui_impl_vulkan.cpp's ImGui_ImplVulkan_RenderWindow) -- a
	// legacy VkRenderPass-based path that's entirely separate from this app's custom dynamic-
	// rendering/shader-object main renderer. That mismatch is what caused
	// "vkQueueSubmit(): ... performs a layout transition on presentable VkImage ... but the
	// image has not been acquired" whenever a popup (e.g. a combo dropdown) got positioned
	// outside the main window and ImGui spun up a secondary viewport/swapchain for it -- more
	// likely whenever the Settings panel was tall enough to push the popup out of bounds
	// (e.g. with many per-material "Albedo Color" pickers). Docking itself (panels within the
	// one main window) has no extra swapchains involved and isn't affected by removing this.
	ImGui::GetIO().ConfigFlags = ImGuiConfigFlags_DockingEnable;
}


/*--
 * The heap is split into two parts:
 *   1. Sampler heap  -- holds sampler descriptors (linear, nearest, etc.)
 *   2. Resource heap  -- holds image descriptors (texture2D views)
 *
 * Each heap buffer has a "reserved range" at the end (required by the spec), followed by the
 * user descriptors at the beginning. The host staging data is written with
 * vkWriteSamplerDescriptorsEXT / vkWriteResourceDescriptorsEXT, then uploaded to GPU buffers.
 *
 * At draw time, vkCmdBindSamplerHeapEXT / vkCmdBindResourceHeapEXT bind these buffers.
-*/
void App::createDescriptorHeap(VkCommandBuffer cmd)
{
	// Every texture (hand-authored test images + any glTF-loaded material textures) has
	// already registered itself into g_renderer.m_descriptorHeap and, at the same index,
	// g_renderer.m_rtTextureViews -- see Texture::uploadPixels. build() uploads whatever has
	// accumulated there to the GPU in one pass.
	g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

	// Query the descriptor heap properties: descriptor sizes, alignment, and maximum heap capacities
	VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT };
	VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &heapProps };
	vkGetPhysicalDeviceProperties2(g_renderer.m_context.getPhysicalDevice(), &props2);

	// ---- Sampler Heap ----

	// Compute how many descriptors the hardware can hold, accounting for the mandatory reserved range
	uint32 maxSamplerCapacity = static_cast<uint32>(
		(heapProps.maxSamplerHeapSize - heapProps.minSamplerHeapReservedRange) / heapProps.samplerDescriptorSize);

	// We only need one sampler in this sample (linear filtering), but the heap can hold more
	uint32 maxSamplers = std::min(1U, maxSamplerCapacity);
	VkDeviceSize samplerHeapSize = heapProps.samplerDescriptorSize * maxSamplers + heapProps.minSamplerHeapReservedRange;
	samplerHeapSize = utils::alignUp(samplerHeapSize, heapProps.samplerHeapAlignment);
	std::vector<uint8> samplerHeapData(samplerHeapSize, 0);  // CPU staging buffer, zero-initialized

	// Write one sampler descriptor (linear filter, repeat addressing)
	VkSamplerCreateInfo samplerCI{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		// Negative bias shifts every implicit-LOD Sample() call (shader.rast.slang's albedo/
		// metallic-roughness/lightmap lookups -- the only ones using automatic screen-space-
		// derivative LOD, not an explicit SampleLevel/Load) toward sharper, lower mip levels --
		// standard practice paired with TAA (taa_resolve.slang), which softens fine detail over
		// several accumulated jittered samples; the extra aliasing/shimmer this reintroduces at full
		// sharpness is exactly what TAA's neighborhood-clamped temporal blend is there to hide.
		// ComputeTemporalJitter's Halton(2,3) sequence effectively supersamples each pixel across
		// ~16 distinct sub-pixel positions once accumulated -- a 4x4 grid equivalent -- so the mip
		// bias should be log2(1/4) = -2 (two full levels sharper), not -1 (which only accounts for a
		// 2x2/4-sample equivalent).
		.mipLodBias = -2.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
	};

	// Same sampler, but as a plain VkSampler for the ray tracing pipeline's traditional binding
	// (see Shader::updateRayDescriptorSet's comment). acquireSampler deduplicates against the
	// heap-registered one below, so this doesn't create a second physical sampler object.
	m_rtSampler = m_samplerPool.acquireSampler(samplerCI);

	// Using acquireSamplerDescriptor avoid any duplicate samplers in the heap.
	uint32 samplerIndex = m_samplerPool.acquireSamplerDescriptor(samplerCI);

	// Point into the staging buffer at the correct offset for this sampler index
	VkHostAddressRangeEXT samplerDst{
		.address = samplerHeapData.data() + static_cast<size_t>(samplerIndex) * static_cast<size_t>(heapProps.samplerDescriptorSize),
		.size = heapProps.samplerDescriptorSize,
	};
	// Write the sampler descriptor into host memory (CPU staging)
	vkWriteSamplerDescriptorsEXT(g_renderer.m_context.getDevice(), 1, &samplerCI, &samplerDst);

	// All heap buffers need these usage flags: device address for binding, transfer dst for upload,
	// and the descriptor heap bit to mark them as heap storage
	VkBufferUsageFlags2 heapUsage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
		| VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;


	// Create the GPU buffer for the sampler heap and upload the staging data
	m_samplerHeapBuffer = g_renderer.m_allocator.createBufferAndUploadData(cmd, std::span(samplerHeapData), heapUsage, {}, heapProps.samplerHeapAlignment);
	DBG_VK_NAME(m_samplerHeapBuffer.buffer);

	m_samplerHeapSize = samplerHeapSize;
	m_samplerReservedOffset = heapProps.samplerDescriptorSize * maxSamplers;
	m_samplerReservedSize = heapProps.minSamplerHeapReservedRange;
}

void App::createUIDescriptorPool()
{
	VkPhysicalDeviceProperties2 deviceProperties2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	vkGetPhysicalDeviceProperties2(g_renderer.m_context.getPhysicalDevice(), &deviceProperties2);
	const auto& deviceProperties = deviceProperties2.properties;

	// ImGui's Vulkan backend (1.92+) uses separate SAMPLED_IMAGE + SAMPLER descriptors:
	//   * One sampler set per built-in sampler (linear + nearest)
	//   * One sampled-image set per texture: font atlas (which may grow dynamically),
	//     RenderTarget display, and every ImGui_ImplVulkan_AddTexture caller.
	// IMGUI_IMPL_VULKAN_MINIMUM_*_POOL_SIZE define the absolute floors; we pick a
	// generous multiple so the sample has comfortable headroom for additional
	// ImTexture instances. All sizes are clamped to device limits to stay safe on
	// minimum-spec hardware.
	constexpr uint32 kUiTextureBudget = 128;  // Max simultaneous ImGui textures
	const uint32     samplerSize =
		std::min<uint32>(IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE, deviceProperties.limits.maxDescriptorSetSamplers);
	const uint32 sampledImageSize =
		std::min<uint32>(std::max<uint32>(IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE, kUiTextureBudget),
			deviceProperties.limits.maxDescriptorSetSampledImages);
	const uint32 maxDescriptorSets = samplerSize + sampledImageSize;

	const std::array<VkDescriptorPoolSize, 2> poolSizes = { {
		{VK_DESCRIPTOR_TYPE_SAMPLER, samplerSize},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledImageSize},
	} };
	const VkDescriptorPoolCreateInfo          poolInfo = {
				  .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				  .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
				  .maxSets = maxDescriptorSets,
				  .poolSizeCount = uint32(poolSizes.size()),
				  .pPoolSizes = poolSizes.data(),
	};

	VK_CHECK(vkCreateDescriptorPool(g_renderer.m_context.getDevice(), &poolInfo, nullptr, &m_uiDescriptorPool));
	DBG_VK_NAME(m_uiDescriptorPool);
	LOGI("Created UI descriptor pool: %u sets (sampler=%u, sampledImage=%u)", maxDescriptorSets, samplerSize, sampledImageSize);
}


bool App::prepareFrameResources()
{
	// Check if swapchain needs rebuilding (this internally calls vkQueueWaitIdle())
	if (m_swapchain.needRebuilding())
	{
		m_windowSize = m_swapchain.reinitResources(m_vSyncMode);
	}

	// Wait first, *then* acquire. Waiting on the timeline semaphore guarantees the
	// GPU has released this slot's resources (command buffer, in-flight data) before
	// we start reusing them. Acquiring first would mean we hold a swapchain image
	// while still potentially racing the GPU on per-frame resources -- and in
	// out-of-order presentation, the wrong slot's wait value would be in scope.
	auto& frame = m_frameData[m_swapchain.getFrameResourceIndex()];

	// Wait until GPU has finished processing the frame that was using these resources previously
	// Note: If swapchain was rebuilt above, this wait is essentially a no-op since vkQueueWaitIdle() was already called
	const VkSemaphoreWaitInfo waitInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
		.semaphoreCount = 1,
		.pSemaphores = &m_frameTimelineSemaphore,
		.pValues = &frame.lastSignalValue,
	};
	VK_CHECK(vkWaitSemaphores(g_renderer.m_context.getDevice(), &waitInfo, std::numeric_limits<uint64>::max()));
#ifdef NVVK_SEMAPHORE_DEBUG
	LOGI("WaitFrame: \t\t slot=%u waitValue=%llu", m_swapchain.getFrameResourceIndex(),
		static_cast<unsigned long long>(frame.lastSignalValue));
#endif

	// The wait above just proved the GPU has completed everything up through
	// frame.lastSignalValue -- free anything Buffer::deinit() queued that's covered by that
	// (see DeferredRelease.h / g_deferredRelease.BeginFrame in the render loop).
	g_deferredRelease.Update(g_renderer.m_allocator, frame.lastSignalValue);

	VkResult result = m_swapchain.acquireNextImage(g_renderer.m_context.getDevice());
	return (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);  // Continue only if we got a valid image
}

VkCommandBuffer App::beginCommandRecording()
{
	VkDevice device = g_renderer.m_context.getDevice();

	// Get the frame data for the current in-flight slot (owned by Swapchain).
	auto& frame = m_frameData[m_swapchain.getFrameResourceIndex()];

	/*--
	 * Reset the whole command pool to reuse its command buffer for recording
	 * the current frame. An equivalent alternative is to create the pool with
	 * VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT and call
	 * vkResetCommandBuffer() per buffer; whole-pool reset is simpler when each
	 * pool only contains one buffer (as here).
	-*/
	VK_CHECK(vkResetCommandPool(device, frame.cmdPool, 0));
	VkCommandBuffer cmd = frame.cmdBuffer;

	// Begin the command buffer recording for the frame
	const VkCommandBufferBeginInfo beginInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
											 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	return cmd;
}

//--- Main ---------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	bool hasConsole = AttachParentConsole();
	g_suppressAssertDialogs = hasConsole;

	printf("VVerse\n");


/*
	// The linker builds this as a Windows-subsystem (GUI) exe (see VVerse.vcxproj's <SubSystem>),
	// deliberately -- so double-clicking it doesn't flash an empty console window. But that also
	// means LOGI/std::cout/printf go nowhere when launched from an existing console (cmd/PowerShell
	// don't attach a GUI-subsystem child's stdio to themselves), which silently swallows every log
	// line AND the crash-hook's own fputs() below -- confirmed live: --test/--selftest looked like
	// they produced no output at all when run this way. AttachConsole(ATTACH_PARENT_PROCESS) hooks
	// this process's stdio up to whichever console actually launched it, when there is one; when
	// there isn't (double-clicked from Explorer, or launched detached), it just fails and this is a
	// no-op, preserving the original silent-GUI-app behavior.
	if (AttachConsole(ATTACH_PARENT_PROCESS))
	{
		printf("1\n");
		// Only take over a stream that isn't already pointed somewhere real. A GUI-subsystem
		// process launched with no redirection gets an unusable handle for each of these
		// (GetFileType() == FILE_TYPE_UNKNOWN) -- but `> file`/`| pipe` redirection (or an
		// already-attached console) gives it a real one (FILE_TYPE_DISK/PIPE/CHAR), and
		// unconditionally overwriting that with CONOUT$ would silently discard the caller's own
		// redirection instead of adding console output alongside it. (Confirmed live: without
		// this check, `VVerse.exe --test > log.txt` under a console-attached launcher wrote
		// nothing to log.txt because stdout got redirected to the console instead.)
		bool attachedOutput = false;
		if (GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_UNKNOWN)
		{
			FILE* dummy = nullptr;
			freopen_s(&dummy, "CONOUT$", "w", stdout);
			attachedOutput = true;
		}
		if (GetFileType(GetStdHandle(STD_ERROR_HANDLE)) == FILE_TYPE_UNKNOWN)
		{
			FILE* dummy = nullptr;
			freopen_s(&dummy, "CONOUT$", "w", stderr);
		}
		if (GetFileType(GetStdHandle(STD_INPUT_HANDLE)) == FILE_TYPE_UNKNOWN)
		{
			FILE* dummy = nullptr;
			freopen_s(&dummy, "CONIN$", "r", stdin);
		}
		std::ios::sync_with_stdio(true);
		if (attachedOutput)
		{
			// The console's prompt has already redrawn itself on its own line the instant this
			// (asynchronous, GUI-subsystem) process launched -- without this, our first real
			// output line lands appended to that stale prompt line instead of starting fresh
			// below it.
			std::cout << std::endl;
		}
	}
	else printf("2\n");
*/

	bool isDebuggerPresent = IsDebuggerPresent();
	// Gated on hasConsole (launched from an existing console, see AttachParentConsole/Platform.cpp),
	// not IsDebuggerPresent() -- confirmed live that IsDebuggerPresent()-gated suppression (hook,
	// then _CrtSetReportMode, then both together) had no effect no matter how the app was launched,
	// which points at IsDebuggerPresent() itself spuriously returning true (a known false-positive
	// caused by overlay/injection software -- GPU overlays, Discord, RGB tools, etc. -- attaching to
	// the process). The actual requirement is simpler anyway: suppress the dialog specifically for a
	// console-driven run (--test/--selftest, or any unattended launch), not "whenever no debugger",
	// so key it on the thing we can directly observe instead of a signal that's proving unreliable.
	if (hasConsole)
	{
		// ASSERT (extern/vk_framework.h) uses plain ASSERT() in Debug builds, whose default
		// MSVC CRT behavior is an interactive "Debug Error! Abort/Retry/Ignore" dialog, and an
		// unhandled exception would otherwise hit Windows Error Reporting's "close program /
		// debug" dialog too -- both block forever with no attached desktop/human to click
		// them, which defeats unsupervised runs (--test, or any scripted/automated launch).
		//
		// Both a report hook and _CrtSetReportMode/_CrtSetReportFile are set below -- belt and
		// braces, since which one (if either) a given ASSERT() call site actually consults has
		// proven inconsistent in practice.
		_CrtSetReportHook([](int reportType, char* message, int*) -> int
			{
				// Raw fopen/fprintf, deliberately bypassing stdout/stderr's own redirection state
				// entirely -- a diagnostic to answer definitively whether this hook is even being
				// invoked for a given ASSERT() call site, independent of every console/stdio
				// uncertainty elsewhere in main(). Remove once that's confirmed either way.
				FILE* f = nullptr;
				fopen_s(&f, "assert_hook_fired.txt", "a");
				if (f)
				{
					fprintf(f, "reportType=%d message=%s\n", reportType, message ? message : "(null)");
					fclose(f);
				}
				if (message)
					fputs(message, stderr);
				if (reportType == _CRT_ASSERT)
					abort();
				return TRUE;  // handled; skip the default (dialog) reporting path
			});
		for (int reportType : { _CRT_ASSERT, _CRT_ERROR, _CRT_WARN })
		{
			_CrtSetReportMode(reportType, _CRTDBG_MODE_FILE);
			_CrtSetReportFile(reportType, _CRTDBG_FILE_STDERR);
		}
		// abort() itself (called just above) has its OWN "Debug Error! abort() has been called"
		// dialog in the Debug CRT, entirely separate from _CrtSetReportHook (which only covers
		// _CRT_ASSERT/_CRT_WARN/_CRT_ERROR reporting) -- without this, that second dialog still
		// blocks an unattended run even though the assert dialog above is suppressed.
		_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
		SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
	}

	// unsupervised test mode (VVerse.exe --test): runs all test suites and exits before
	// touching GLFW/Vulkan, see TestMode.h
	int testExitCode = 0;
	if (tests::RunTestModeIfRequested(argc, argv, testExitCode))
		return testExitCode;

	// OBS registers its Vulkan capture layer as a system-wide implicit layer
	// (HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers -> obs-vulkan64.json), so the Vulkan loader loads
	// it into every Vulkan process regardless of whether OBS itself is running. That hook has caused
	// a genuine ~2s GPU stall here (VK_ERROR_DEVICE_LOST from a driver TDR, confirmed by disabling
	// every app-side workload variable -- shadow rays, validation layers, resolution -- with no
	// change, and reproducing only with the hook loaded). OBS documents DISABLE_VULKAN_OBS_CAPTURE as
	// the supported opt-out for exactly this; must be set before any Vulkan call.
	SetEnvironmentVariableA("DISABLE_VULKAN_OBS_CAPTURE", "1");

	// Get the logger instance
	utils::Logger& logger = utils::Logger::getInstance();
	// logger.enableFileOutput(false);  // Don't write log to file
  //  logger.setShowFlags(utils::Logger::eSHOW_TIME);
	logger.setLogLevel(utils::Logger::LogLevel::eINFO);  // Default is Warning, we show more information

	LOGI("hasConsole = %d", (int)hasConsole);
	LOGI("isDebuggerPresent = %d", (int)isDebuggerPresent);	

	// open Retry/Abort debug window in VisualStudio, even if run from console and not debugger attached
//	ASSERT(0);

	{
		char str[1024];
		GetCurrentDirectoryA(1024, str);
		LOGI("CurrentDir: '%s'", str);
	}


	// VVerse.exe --selftest: unlike --test above, this DOES need GLFW/Vulkan (it's a scripted pass
	// through the real app, see TestMode.h's own comment) -- only parse argv here, the actual run
	// happens below via App::runSelfTest() instead of the normal ::run().
	const bool wantSelfTest = tests::WantsSelfTestMode(argc, argv);
	int selfTestExitCode = 0;

	// --validate=0/1: see g_enableValidationLayers's own comment for why this has to be a
	// command-line flag (read before vkCreateInstance) rather than a live Settings-window toggle.
	for (int i = 1; i < argc; ++i)
		if (strcmp(argv[i], "--validate=0") == 0)
			g_enableValidationLayers = false;
		else if (strcmp(argv[i], "--validate=1") == 0)
			g_enableValidationLayers = true;
	LOGI("g_enableValidationLayers = %d", (int)g_enableValidationLayers);

	try
	{
		ASSERT(glfwInit() == GLFW_TRUE, "Could not initialize GLFW!");
		ASSERT(glfwVulkanSupported() == GLFW_TRUE, "GLFW: Vulkan not supported!");

		{
			// app must be destroyed (window, Vulkan surface/swapchain, ImGui-GLFW callbacks)
			// before glfwTerminate() tears down GLFW's window/callback state.
			App app({ 1600, 1024 }, /*hidden=*/wantSelfTest);
			if (wantSelfTest)
				selfTestExitCode = app.runSelfTest();
			else
				app.run();
		}

		glfwTerminate();
	}
	catch (const std::exception& e)
	{
		LOGE("%s", e.what());
		return 1;
	}
	return wantSelfTest ? selfTestExitCode : 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// __argc/__argv (declared by <stdlib.h>, already pulled in transitively) are populated
	// by the CRT startup code before WinMain runs, regardless of which entry point/subsystem
	// is used; this forwards the real command line to main(int argc, char** argv) without
	// needing to parse lpCmdLine by hand.
	return main(__argc, __argv);
}
