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
#include "EventManager.h"
#include "Platform.h"
#include "Texture.h"
#include "Renderer.h"
#include "DeferredRelease.h"
#include "App.h"
#include "imgui_stdlib.h"
#include <glm/gtc/type_ptr.hpp>
#include <filesystem>

uint32 g_frameIndex = 0;

Renderer g_renderer;

Shader g_post("shaders/post.slang", "vertexMain", "fragmentMain");

// Screen-space rect of the "Viewport" window's content region, captured once per frame (see the
// mouse-position block in run()) so the gizmo (drawn later, in drawFrame()) can be positioned with
// ImGuizmo::SetRect() using the same rect the mouse-picking code already computes.
ImVec2 g_viewportScreenPos(0, 0);
ImVec2 g_viewportSize(0, 0);

// See global.h's own declaration/comment (its ASSERT() redefinition is the actual reader) -- set
// below in main(), right after AttachParentConsole().
bool g_suppressAssertDialogs = false;

// [mesh->getHandle().index] = IB+VB
Buffer g_meshBuffer(EF_Type_Unknown);
// [entity->getHandle().index] = mesh->getHandle().index -- see RayPushData::entityMeshIndexAddress
// (shader_io.h) for why a closest-hit shader needs this indirection instead of using InstanceID()
// (an entity index, see Renderer::createTopLevelAS) to index g_meshBuffer directly.
Buffer g_entityMeshIndexBuffer(EF_Type_Unknown);

bool g_InfoHUD = false;

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

// Populates the scene with the same hand-authored content init() used to load unconditionally
// behind LOAD_BSP_ON_START/LOAD_GLTF_ON_START -- now a runtime action (the "setup #1" button)
// instead of a compile-time choice, so it can run after init() has already built an empty scene.
void setupScene1()
{
	VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);
	VulkanContext vulkanContext(g_renderer.m_context.getDevice(), g_renderer.m_context.getPhysicalDevice(), g_renderer.m_allocator, cmd, g_renderer.m_context.getGraphicsQueue().queue, g_renderer.m_transientCmdPool);

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


	// createDescriptorHeap() (called once from init()) also builds the one-time sampler heap;
	// only the resource heap needs rebuilding here, to account for the textures just added --
	// same reasoning as loadScene()'s equivalent rebuild after reloading textures.
	g_renderer.m_descriptorHeap.build(g_renderer.m_context.getDevice(), g_renderer.m_allocator, cmd);

	utils::endSingleTimeCommands(cmd, g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	// Staging buffers created above are already sitting in g_renderer.m_allocator, no longer freed
	// here directly -- the main loop's run() now drains them into g_deferredRelease every frame
	// (see that call site's comment), the same frame-safe deferred destruction every other Buffer
	// already gets, instead of each caller managing its own staging buffers' lifetime by hand.
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

//--- App ------------------------------------------------------------------------------------------------------------
// See App.h for the class definition.

App::App(VkExtent2D size)
	: m_windowSize(size)
{
	// Windows 10 Creators update adds Per Monitor V2 DPI awareness context.
	// Using this awareness context allows the client area of the window
	// to achieve 100% scaling while still allowing non-client window content to
	// be rendered in a DPI sensitive fashion.
	SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

	// Vulkan Loader
	VK_CHECK(volkInitialize());
	// Create the GLFW window
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	const char* windowTitle = "VulkanSlangStarter Window";

	m_window = glfwCreateWindow(m_windowSize.width, m_windowSize.height, windowTitle, nullptr, nullptr);

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


	init();
}

App::~App()
{
	destroy();

	glfwDestroyWindow(m_window);
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

		if (glfwGetWindowAttrib(m_window, GLFW_ICONIFIED) == GLFW_TRUE)
		{
			ImGui_ImplGlfw_Sleep(10);  // Do nothing when minimized
			continue;
		}
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

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

	//	const char* spGetBuildTagString();
	//	Log("Slang Version: %s\n", spGetBuildTagString());

	Log("\n");

	extern CObjManager<CTexHandle, Texture> g_TextureManager;
	g_TextureManager.InitOnce(4096);

	// Create GPU buffers (SSBO) containing the vertex data and the point data, and the image (uploading data to GPU)
	{
		VkCommandBuffer cmd = utils::beginSingleTimeCommands(g_renderer.m_context.getDevice(), g_renderer.m_transientCmdPool);

		g_renderer.init(g_renderer.m_context.getPhysicalDevice(), cmd);

		EventContext context{ g_renderer.m_context.getDevice(), g_renderer.m_allocator };
		// after g_renderer.init as this is where the shadercache is loaded
		g_EventManager.callAppInit(EF_Type_Mask, context);

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
}

void App::destroy()
{
	VkDevice device = g_renderer.m_context.getDevice();

	if (device)
		VK_CHECK(vkDeviceWaitIdle(device));

	m_swapchain.deinit();
	m_samplerPool.deinit();
	m_imTexture.deinit();  // Destroy the ImGui texture wrapper before shutting down ImGui itself

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

	m_renderTarget.deinit();

	// device is confirmed idle (VK_CHECK(vkDeviceWaitIdle(device)) above), and every
	// Buffer::deinit() call from this shutdown sequence -- including g_EventManager's
	// callAppDeInit() above, which reaches every still-alive Buffer (mesh vertex/index/BLAS,
	// g_meshBuffer, each ray tracing Shader's own m_sbt, g_outputBuffer, ...) --
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

void App::drawFrame(VkCommandBuffer cmd)
{
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
		ImGui::Image(m_imTexture, ImGui::GetContentRegionAvail());

		// Adding overlay text on the upper left corner
		ImGui::SetCursorPos(ImVec2(0, 0));
		ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

		for(const auto &el : g_UIErrors)
		{
			ImGui::TextColored(ImVec4(1,0,0,1), "Error: %s", el.c_str());
		}
		g_UIErrors.clear();
	}
	ImGui::End();


	if (ImGui::Begin("Settings"))
	{
		ImGui::Separator();

		if(ImGui::Combo("VSync Mode", (int*)&m_vSyncMode, "Off\0" "TearFree\0" "VSync\0"))
			m_swapchain.requestRebuild();  // Recreate the swapchain with the new vSync setting

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
			setupScene1();
		}
	}
	ImGui::End();

	ImGui::Render();  // This is creating the data to draw the UI (not on GPU yet)

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

	// Start rendering to the swapchain
	beginDynamicRenderingToSwapchain(cmd);
	{
		// The ImGui draw commands are recorded to the command buffer, which includes the display of our RenderTarget image
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
	}
	endDynamicRenderingToSwapchain(cmd);
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

void App::onViewportSizeChange(VkExtent2D size)
{
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
void App::updateSceneBuffer(VkCommandBuffer cmd) const
{
	// Updating the data for the frame
	shaderio::SceneInfo sceneInfo{};
	float time = static_cast<float>(ImGui::GetTime());
	float sineValue = std::sin(time);    // Get the sine of the current time
	float mappedValue = sineValue;

	sceneInfo.resolution = float2(m_viewportSize.width, m_viewportSize.height);
	sceneInfo.clearColor = float3(0, 1, 0);	// green

	sceneInfo.infoHUD = g_InfoHUD;

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


void App::recordGraphicsCommands(VkCommandBuffer cmd)
{
	DBG_VK_SCOPE(cmd);  // <-- Helps to debug in Nsight

	// Update g_lights before the scene buffer, which points SceneInfo::lightsAddress/lightCount at
	// it (cannot be done inside dynamic rendering)
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

	VkClearColorValue clearColor; // dark yellow
	clearColor.float32[0] = 0.5f;
	clearColor.float32[1] = 0.5f;
	clearColor.float32[2] = 0;
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


	// Begin dynamic rendering
	vkCmdBeginRendering(cmd, &renderingInfo);

	// ---- Dynamic state ----------------------------------------------------
	// With shader objects, every piece of state that used to live in the
	// graphics pipeline must be set explicitly with a vkCmdSet*EXT call before
	// each draw. We set everything once here since none of it changes between
	// the two draws below.
	setGraphicsDynamicState(cmd, viewport, scissor);

	vkCmdPushDataEXT(cmd, &pushDataInfo);

	// post #################################

	if(g_post.m_shader[0] && g_post.m_shader[1])
	{
		utils::cmdBindGraphicsShaders(cmd, g_post.m_shader[0], g_post.m_shader[1]);
		vkCmdDraw(cmd, 3, 1, 0, 0);  // 3 vertices
	}
	else g_UIErrors.push_back("g_post failed to compile");

	vkCmdEndRendering(cmd);
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
		.mipLodBias = 0.0f,
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

	printf("VulkanSlangStarter\n");

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
			App app({ 1600, 1024 });
			app.run();
		}

		glfwTerminate();
	}
	catch (const std::exception& e)
	{
		LOGE("%s", e.what());
		return 1;
	}
	return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// __argc/__argv (declared by <stdlib.h>, already pulled in transitively) are populated
	// by the CRT startup code before WinMain runs, regardless of which entry point/subsystem
	// is used; this forwards the real command line to main(int argc, char** argv) without
	// needing to parse lpCmdLine by hand.
	return main(__argc, __argv);
}
