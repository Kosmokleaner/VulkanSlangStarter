/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2026 Martin Mittring
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Martin Mittring
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from src/debug_util.h in https://github.com/nvpro-samples/vk_minimal_latest,
 * modified by Martin Mittring.
 */

#pragma once

// This file defines the DebugUtil class, a singleton utility for managing Vulkan debug utilities.
// It provides functionality to set debug names for Vulkan objects and manage debug labels in command buffers.
//
// Usage:
// 1. Initialize the DebugUtil with a Vulkan device using debugUtilInitialize(VkDevice device).
// 2. Use DBG_VK_NAME(obj) macro to set debug names for Vulkan objects.
// 3. Use DBG_VK_SCOPE(cmdBuf) macro to create scoped debug labels in command buffers.
//
// Example:
// debugUtilInitialize(device);
// VkBuffer buffer = createBuffer(...);
// DBG_VK_NAME(buffer);
//
// void someFunction(VkCommandBuffer cmdBuf)
// {
//   DBG_VK_SCOPE(cmdBuf);
//   // Command buffer operations
// }

#include <type_traits>  // std::is_same_v
#include <cstring>      // for strrchr
#include <string>

namespace utils {

class DebugUtil
{
public:
  static DebugUtil& getInstance()
  {
    static DebugUtil instance;
    return instance;
  }

  void init(VkDevice device) { m_device = device; }

  bool isInitialized() const { return m_device != VK_NULL_HANDLE; }

  template <typename T>
  void setObjectName(T object, const std::string& name) const;

  class ScopedCmdLabel
  {
  public:
    ScopedCmdLabel(VkCommandBuffer cmdBuf, const std::string& label)
        : m_cmdBuf(cmdBuf)
    {
      if(vkCmdBeginDebugUtilsLabelEXT != nullptr)
      {
        VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
        vkCmdBeginDebugUtilsLabelEXT(m_cmdBuf, &s);
      }
    }

    ~ScopedCmdLabel()
    {
      if(vkCmdEndDebugUtilsLabelEXT != nullptr)
        vkCmdEndDebugUtilsLabelEXT(m_cmdBuf);
    }

  private:
    VkCommandBuffer m_cmdBuf;
  };

private:
  DebugUtil() = default;
  VkDevice m_device{VK_NULL_HANDLE};

  template <typename T>
  static constexpr VkObjectType getObjectType();
};

template <typename T>
void DebugUtil::setObjectName(T object, const std::string& name) const
{
  constexpr VkObjectType objectType = getObjectType<T>();

  if(vkSetDebugUtilsObjectNameEXT != nullptr && objectType != VK_OBJECT_TYPE_UNKNOWN && m_device != VK_NULL_HANDLE)
  {
    VkDebugUtilsObjectNameInfoEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr, objectType,
                                    (uint64_t)object, name.c_str()};
    vkSetDebugUtilsObjectNameEXT(m_device, &s);
  }
}

template <typename T>
constexpr VkObjectType DebugUtil::getObjectType()
{
  if constexpr(std::is_same_v<T, VkBuffer>)
    return VK_OBJECT_TYPE_BUFFER;
  else if constexpr(std::is_same_v<T, VkBufferView>)
    return VK_OBJECT_TYPE_BUFFER_VIEW;
  else if constexpr(std::is_same_v<T, VkCommandBuffer>)
    return VK_OBJECT_TYPE_COMMAND_BUFFER;
  else if constexpr(std::is_same_v<T, VkCommandPool>)
    return VK_OBJECT_TYPE_COMMAND_POOL;
  else if constexpr(std::is_same_v<T, VkDescriptorPool>)
    return VK_OBJECT_TYPE_DESCRIPTOR_POOL;
  else if constexpr(std::is_same_v<T, VkDescriptorSet>)
    return VK_OBJECT_TYPE_DESCRIPTOR_SET;
  else if constexpr(std::is_same_v<T, VkDescriptorSetLayout>)
    return VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
  else if constexpr(std::is_same_v<T, VkDevice>)
    return VK_OBJECT_TYPE_DEVICE;
  else if constexpr(std::is_same_v<T, VkDeviceMemory>)
    return VK_OBJECT_TYPE_DEVICE_MEMORY;
  else if constexpr(std::is_same_v<T, VkFence>)
    return VK_OBJECT_TYPE_FENCE;
  else if constexpr(std::is_same_v<T, VkFramebuffer>)
    return VK_OBJECT_TYPE_FRAMEBUFFER;
  else if constexpr(std::is_same_v<T, VkImage>)
    return VK_OBJECT_TYPE_IMAGE;
  else if constexpr(std::is_same_v<T, VkImageView>)
    return VK_OBJECT_TYPE_IMAGE_VIEW;
  else if constexpr(std::is_same_v<T, VkInstance>)
    return VK_OBJECT_TYPE_INSTANCE;
  else if constexpr(std::is_same_v<T, VkPipeline>)
    return VK_OBJECT_TYPE_PIPELINE;
  else if constexpr(std::is_same_v<T, VkPipelineCache>)
    return VK_OBJECT_TYPE_PIPELINE_CACHE;
  else if constexpr(std::is_same_v<T, VkPipelineLayout>)
    return VK_OBJECT_TYPE_PIPELINE_LAYOUT;
  else if constexpr(std::is_same_v<T, VkQueryPool>)
    return VK_OBJECT_TYPE_QUERY_POOL;
  else if constexpr(std::is_same_v<T, VkRenderPass>)
    return VK_OBJECT_TYPE_RENDER_PASS;
  else if constexpr(std::is_same_v<T, VkSampler>)
    return VK_OBJECT_TYPE_SAMPLER;
  else if constexpr(std::is_same_v<T, VkSemaphore>)
    return VK_OBJECT_TYPE_SEMAPHORE;
  else if constexpr(std::is_same_v<T, VkShaderModule>)
    return VK_OBJECT_TYPE_SHADER_MODULE;
  else if constexpr(std::is_same_v<T, VkSurfaceKHR>)
    return VK_OBJECT_TYPE_SURFACE_KHR;
  else if constexpr(std::is_same_v<T, VkSwapchainKHR>)
    return VK_OBJECT_TYPE_SWAPCHAIN_KHR;
  else
    return VK_OBJECT_TYPE_UNKNOWN;
}
}  // namespace utils

#define DBG_VK_SCOPE(_cmd) utils::DebugUtil::ScopedCmdLabel scopedCmdLabel(_cmd, __FUNCTION__)

namespace utils {
// Helpers for DBG_VK_NAME: pick the final path component and the class-name
// tail past any leading namespace / 'struct ' / 'class ' prefix emitted by
// typeid(...).name().
inline const char* dbgFilenameOnly(const char* path)
{
  const char* slash     = strrchr(path, '/');
  const char* backslash = strrchr(path, '\\');
  const char* sep       = (slash > backslash) ? slash : backslash;
  return sep ? sep + 1 : path;
}

inline const char* dbgTypeNameTail(const char* typeidName)
{
  const char* lastSpace = strrchr(typeidName, ' ');
  return lastSpace ? lastSpace + 1 : typeidName;
}
}  // namespace utils

#define DBG_VK_NAME(obj)                                                                                               \
  if(utils::DebugUtil::getInstance().isInitialized())                                                                  \
  utils::DebugUtil::getInstance().setObjectName(obj, std::string(utils::dbgTypeNameTail(typeid(*this).name()))         \
                                                         + "::" + #obj + " (in " + utils::dbgFilenameOnly(__FILE__)    \
                                                         + ":" + std::to_string(__LINE__) + ")")

#define DBG_VK_NAME2(obj, name)                                                                                         \
  if(utils::DebugUtil::getInstance().isInitialized())                                                                  \
  utils::DebugUtil::getInstance().setObjectName(obj, std::string(name)         \
                                                         + "::" + #obj + " (in " + utils::dbgFilenameOnly(__FILE__)    \
                                                         + ":" + std::to_string(__LINE__) + ")")


inline void debugUtilInitialize(VkDevice device)
{
  utils::DebugUtil::getInstance().init(device);
}
