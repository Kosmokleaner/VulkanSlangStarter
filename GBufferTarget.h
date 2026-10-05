#pragma once
#include "vulkanBase.h"

// A single G-buffer-style render target: a VkImage (STORAGE + SAMPLED, for a compute/raygen shader's
// RWTexture2D writes and a later pass's filtered/point reads) + its view. Shared by the raster
// G-buffers (g_gbufferAlbedo/NormalRoughness/ViewZ/MotionVector, App.cpp) and the TAA
// resolve pass's ping-pong history images (TaaResolveState::history, TaaResolve.h) -- see
// createGBufferTarget()/destroyGBufferTarget() (App.cpp) for how these get built/freed.
struct GBufferTarget
{
	utils::Image image{};
	VkImageView view{};
};
