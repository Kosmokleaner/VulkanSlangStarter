#include "global.h"	// include first in any cpp
#include "vulkanBase.h"
#include "Timer.h"
#include "ShaderCache.h"

#include <slang.h>
#include <slang-com-ptr.h>
#include <vulkan/vulkan_core.h>
#include <windows.h>	// GetCurrentDirectory()
#include "Shader.h"
#include <filesystem>

namespace shaderio {  // Shader IO namespace -- shared layout between C++ and shaders
	using namespace glm;  // GLSL-style types without the glm:: prefix inside the namespace
#include "shaders/shader_io.h"
}  // namespace shaderio

using namespace slang;


Shader::Shader(const char* inFilePath, const char* inCS)
	: filePath(inFilePath)
	, IAppInit(EF_Type_Shader)
	, m_sbt(EF_Type_Unknown)
{
	ASSERT(inCS);
	functionName[0] = inCS;
}

Shader::Shader(const char* inFilePath, const char* inRayGen, const char* inRayMiss, const char* inRayClosestHit)
	: filePath(inFilePath)
	, IAppInit(EF_Type_Shader)
	, m_sbt(EF_Type_Unknown)
{
	ASSERT(inRayGen);
	ASSERT(inRayMiss);
	ASSERT(inRayClosestHit);
	functionName[0] = inRayGen;
	functionName[1] = inRayMiss;
	functionName[2] = inRayClosestHit;
}

Shader::Shader(const char* inFilePath, const char* inVS, const char* inPS)
	: filePath(inFilePath)
	, IAppInit(EF_Type_Shader)
	, m_sbt(EF_Type_Unknown)
{
	ASSERT(inVS);
	ASSERT(inPS);

	functionName[0] = inVS;
	functionName[1] = inPS;
}

// createGlobalSession() is slow (~180ms) so don't do for each shader
// todo: release
Slang::ComPtr<slang::IGlobalSession> g_globalSession;

void Shader::compile()
{
	// todo
	int stageCount = 1;
	if (!functionName[1].empty())
		stageCount = 2;
	if (!functionName[2].empty())
		stageCount = 3;

	CShaderHash Hash[3];

	bool needToCompile = false;
	for (int stageId = 0; stageId < stageCount; ++stageId)
	{
		Hash[stageId] = ComputeShaderHash(filePath.c_str()/*, D3DInclude*/);
		Hash[stageId].Add(functionName[stageId].c_str(), (uint32)functionName[stageId].size());
//		Hash.Add(Target, strlen(Target));
//		Hash.Add((const char*)&dwShaderFlags, 4);

		auto it = g_ShaderCache.KeyValuePairs.find(Hash[stageId]);
		if (it != g_ShaderCache.KeyValuePairs.end())
		{
//			HRESULT hr = D3DCreateBlob(it->second.size(), &ShaderBlob);
//			ASSERT(hr == S_OK);
//			void* Ptr = ShaderBlob->GetBufferPointer();
			code[stageId] = it->second;
			continue;
		} else needToCompile = true;
	}

	if(!needToCompile)
		return;

	SCOPED_CPU_TIMER_LOG(Shader_Compile________________________);

	Log("Compile '%s' '%s' '%s'\n", filePath.c_str(), functionName[0].c_str(), functionName[1].c_str());

	//	Log("Slang Version: %s\n", spGetBuildTagString());

		// 1. Initialize the Slang session
	if (!g_globalSession)
	{
		//		SCOPED_CPU_TIMER_LOG(createGlobalSession);
		createGlobalSession(g_globalSession.writeRef());
	}

	// 2. Set up compilation options
	slang::SessionDesc sessionDesc = {};
	slang::TargetDesc targetDesc = {};
	targetDesc.format = SLANG_SPIRV; // Select your target UI/backend
	targetDesc.profile = g_globalSession->findProfile("spirv_1_5");
	sessionDesc.targets = &targetDesc;
	sessionDesc.targetCount = 1;
	// extern/Rtxdi/Include lets shader code #include "Rtxdi/..." headers verbatim, matching the
	// exact include paths RTXDI's own vendored .hlsli files already use internally -- avoids having
	// to rewrite any vendored file's #include lines.
	const char* searchPaths[] = { "shaders", "extern/Rtxdi/Include" };
	sessionDesc.searchPaths = searchPaths;
	sessionDesc.searchPathCount = 2;

	slang::CompilerOptionEntry entry[2];
	entry[0].name = slang::CompilerOptionName::Capability;
	entry[0].value.kind = slang::CompilerOptionValueKind::String;
	entry[0].value.stringValue0 = "spvDescriptorHeapEXT";
	entry[1].name = slang::CompilerOptionName::Optimization;
	entry[1].value.kind = slang::CompilerOptionValueKind::Int;
//	entry[1].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_NONE;		// compile time ~500ms
	entry[1].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_DEFAULT;	// compile time ~60-0ms
//	entry[1].value.intValue0 = SLANG_OPTIMIZATION_LEVEL_MAXIMAL;	// compile time ~1500ms

	sessionDesc.compilerOptionEntries = entry;
	sessionDesc.compilerOptionEntryCount = sizeof(entry) / sizeof(*entry);
//	sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR; // todo: consider

	Slang::ComPtr<slang::ISession> session;
	{
		//		SCOPED_CPU_TIMER_LOG(createSession);
		g_globalSession->createSession(sessionDesc, session.writeRef());
	}

	// new API
	Slang::ComPtr<slang::IModule> module;
	Slang::ComPtr<slang::IBlob> diagnostics;

	// todo: refactor
	std::filesystem::path p(filePath);
	// e.g. "shaders/shader.comp.slang" => "shader.comp"
	std::string moduleName = p.stem().string();
	{
		//		SCOPED_CPU_TIMER_LOG(load_Module);

		module = session->loadModule(moduleName.c_str(), // module name, not filename
			diagnostics.writeRef());
	}

	if (ISlangBlob* blob = diagnostics.get())
	{
		Log("Compiler Output:\n%s\n", (char*)blob->getBufferPointer());
		OutputDebugStringA((char*)blob->getBufferPointer());
	}


	if (!module)
		return;

	//	request->setCommandLineCompilerFlags("-fvk-use-entrypoint-name");

	// todo
	Slang::ComPtr<slang::IEntryPoint> entryPoint[3];

	// todo
	for (int stageId = 0; stageId < stageCount; ++stageId)
	{
		SlangStage stageEnum = SLANG_STAGE_COMPUTE;

		if (stageCount == 2)	// VSPS
		{
			if (stageId == 0) stageEnum = SLANG_STAGE_VERTEX;
			if (stageId == 1) stageEnum = SLANG_STAGE_FRAGMENT;
		}
		else if (stageCount == 3)	// ray
		{
			if (stageId == 1) stageEnum = SLANG_STAGE_RAY_GENERATION;
			if (stageId == 2) stageEnum = SLANG_STAGE_MISS;
			if (stageId == 3) stageEnum = SLANG_STAGE_CLOSEST_HIT;
		}

		SlangResult res = module->findEntryPointByName(
			functionName[stageId].c_str(),
			entryPoint[stageId].writeRef());

		if (SLANG_FAILED(res) || !entryPoint[stageId])
		{
			Log("Cannot find entry point: %s\n",
				functionName[stageId].c_str());
		}
	}

	// todo
	slang::IComponentType* components[4];
	{
		components[0] = module;

		for (int stageId = 0; stageId < stageCount; ++stageId)
			components[stageId + 1] = entryPoint[stageId];
	}

	Slang::ComPtr<slang::IComponentType> composed;

	SlangResult result;
	{
		//		SCOPED_CPU_TIMER_LOG(createCompositeComponentType);
		result = session->createCompositeComponentType(
			components,
			stageCount + 1,
			composed.writeRef(),
			diagnostics.writeRef());
	}

	if (SLANG_FAILED(result) || !composed)
	{
		if (diagnostics)
		{
			Log("%s\n",
				(const char*)diagnostics->getBufferPointer());
		}
		else
		{
			Log("No Slang diagnostics\n");
		}
	}

	ASSERT(composed);

	Slang::ComPtr<slang::IComponentType> linked;

	SlangResult linkResult;
	{
		//		SCOPED_CPU_TIMER_LOG(link);
		linkResult = composed->link(linked.writeRef(), diagnostics.writeRef());
	}

	if (SLANG_FAILED(linkResult))
	{
		if (diagnostics)
			Log("%s\n",
				(const char*)diagnostics->getBufferPointer());
	}

	// Trigger the compiler, 0 is success
//old	const SlangResult compileResult = request->compile();

	// Always check diagnostics/logs for GLSL syntax errors
//old	if (const char* diagnostics = request->getDiagnosticOutput())
//old	if (*diagnostics)
//old	{
//old		//			Log("Compiler Output:\n%s\n", diagnostics);
//old		OutputDebugStringA(diagnostics);
//old	}

//	if (SLANG_FAILED(compileResult))
//	{
//		ASSERT(0);
//		return;		// Handle compilation failure
//	}

	// todo
	for (int stageId = 0; stageId < stageCount; ++stageId)
	{
		//		SCOPED_CPU_TIMER_LOG(getStages);

		//old		const void* spirvCode = request->getEntryPointCode(entryPointIndex[stageId], &spirvSize);
		Slang::ComPtr<ISlangBlob> spirv;

		//		SlangInt entryPointIndex = module->getDefinedEntryPoint(stageId);

		linked->getEntryPointCode(stageId, 0, spirv.writeRef(), nullptr);

		if(spirv)
		{
			// in bytes
			size_t spirvSize = spirv->getBufferSize();
			const void* spirvCode = spirv->getBufferPointer();

			ASSERT((spirvSize % 4) == 0);

			code[stageId].resize(spirvSize / 4);

			memcpy(code[stageId].data(), spirvCode, spirvSize);
		}
		else code[stageId].clear();


		// Uncomment to dump compiled SPIR-V for inspection (e.g. with spirv-dis.exe from the Vulkan SDK,
		// to compare binding/capability decorations between shaders when something bindless goes wrong):
//		{
//			std::filesystem::path dumpPath = std::filesystem::path("spv_dump") /
//				(moduleName + "_" + functionName[stageId] + ".spv");
//			std::filesystem::create_directories("spv_dump");
//			FILE* f = nullptr;
//			fopen_s(&f, dumpPath.string().c_str(), "wb");
//			if (f)
//			{
//				fwrite(spirvCode, 1, spirvSize, f);
//				fclose(f);
//			}
//		}


//		if (hr == S_OK)
		{
			std::pair<CShaderHash, std::vector<uint32> > entry;

			entry.first = CShaderHash(Hash[stageId]);
			entry.second = code[stageId];

			g_ShaderCache.KeyValuePairs.insert(entry);
		}
	}
}

void Shader::create(EventContext& context)
{
	// Create graphics shader objects (VK_EXT_shader_object). No graphics pipeline
	// is built; everything that used to live in a pipeline is now dynamic state.

	// functionName[2], not code[2]: functionName[] is set once at construction and never touched
	// again, so it reliably tells a 3-stage ray shader from a 2-stage raster one even when compile()
	// failed and left every code[] empty (see its own "if(!module) return" comment) -- code[2].empty()
	// would then misclassify a broken RAY shader as a raster one, dispatching to the wrong (harmless
	// here only because createGfx() has its own, unrelated empty-code guard) creation function.
	if (functionName[2].empty())
		createGfx(context);
	else
		createRay(context);
}

void Shader::createGfx(EventContext& context)
{
	// Guards against a failed compile() (see its if(!module) return -- that leaves code[] empty
	// rather than hard-asserting, so a hot-reload with a syntax error doesn't take the whole app
	// down). This must check the compiled bytecode, not m_shader[0]/[1]: those are this function's
	// own output, VK_NULL_HANDLE until vkCreateShadersEXT below runs, so checking them here always
	// fired -- on first init and after every reload() (deinit() resets them too) -- and skipped
	// shader object creation unconditionally, leaving every graphics shader permanently unbound.
	if (code[0].empty() || code[1].empty())
		return;

	/*--
	 * Specialization constants. Two distinct VkSpecializationInfo instances, each
	 * pointing at its own immutable VkBool32 -- the same pattern we used with
	 * pipelines, equally valid for shader objects.
	-*/
//	static constexpr VkBool32      kUseTextureTrue = VK_TRUE;
//	const VkSpecializationMapEntry specMapEntry = { .constantID = 0, .offset = 0, .size = sizeof(VkBool32) };
//	const VkSpecializationInfo     specInfoTextured = {
//			.mapEntryCount = 1,
//			.pMapEntries = &specMapEntry,
//			.dataSize = sizeof(VkBool32),
//			.pData = &kUseTextureTrue,
//	};

	// Three unlinked shader objects. nextStage is a hint to the driver about
	// which stage will follow at bind time -- it doesn't constrain what we
	// can actually bind.
	const VkShaderCreateFlagsEXT commonFlags = VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;

	VkShaderCreateInfoEXT vertCreateInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.flags = commonFlags,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.codeSize = code[0].size() * sizeof(uint32),
		.pCode = code[0].data(),
		.pName = "main", // VS
		.setLayoutCount = 0,  // Descriptor heap: no descriptor set layouts
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 0,  // Push data (vkCmdPushDataEXT) is used instead
		.pPushConstantRanges = nullptr,
		.pSpecializationInfo = nullptr,  // Vertex shader has no spec constants
	};

	VkShaderCreateInfoEXT fragCreateInfoTextured{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.flags = commonFlags,
		.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.nextStage = 0,  // Last stage in the pipeline
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.codeSize = code[1].size() * sizeof(uint32),
		.pCode = code[1].data(),
		.pName = "main", // PS
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = nullptr,
//		.pSpecializationInfo = &specInfoTextured,
	};

	// Create all three in separate calls. Each creates an unlinked shader object
	// that can be bound independently.
	VK_CHECK(vkCreateShadersEXT(context.device, 1, &vertCreateInfo, nullptr, &m_shader[0]));
	DBG_VK_NAME2(m_shader[0], "VS");
	VK_CHECK(vkCreateShadersEXT(context.device, 1, &fragCreateInfoTextured, nullptr, &m_shader[1]));
	DBG_VK_NAME2(m_shader[1], "FS");
}

// @return 0 if failed
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32>& spirv)
{
	VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
	info.codeSize = spirv.size() * sizeof(uint32);
	info.pCode = spirv.data();

	if(!info.codeSize)
		return 0;	// e.g. shader error

	VkShaderModule module;
	VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
	return module;
}

void Shader::createRay(EventContext& context)
{
	VkShaderModule raygenModule = createShaderModule(context.device, code[0]);
	VkShaderModule missModule = createShaderModule(context.device, code[1]);
	VkShaderModule chitModule = createShaderModule(context.device, code[2]);

	// m_pipeline might be 0 if there is a shader error 
	if (!(raygenModule && missModule && chitModule))
		return;

	VkPipelineShaderStageCreateInfo stages[] =
	{
		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			nullptr, 0, VK_SHADER_STAGE_RAYGEN_BIT_KHR,
			raygenModule, "main", nullptr },

		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			nullptr, 0, VK_SHADER_STAGE_MISS_BIT_KHR,
			missModule, "main", nullptr },

		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			nullptr, 0, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
			chitModule, "main", nullptr }
	};

	VkRayTracingShaderGroupCreateInfoKHR groups[3]{};

	groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[0].generalShader = 0;
	groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
	groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

	groups[1] = groups[0];
	groups[1].generalShader = 1;

	groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	groups[2].generalShader = VK_SHADER_UNUSED_KHR;
	groups[2].closestHitShader = 2;
	groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

	// bindings 3/4: traditional (non-heap) texture + sampler access for the closest-hit shader.
	// VK_EXT_descriptor_heap can't be used for this pipeline: a descriptor-heap pipeline requires
	// layout == VK_NULL_HANDLE (VUID-VkRayTracingPipelineCreateInfoKHR-flags-11311), which conflicts
	// with binding 0's traditional TLAS descriptor set. So the texture lookup goes through an
	// ordinary bound array, capped at shaderio::kMaxRayTracingTextures entries (see
	// Renderer::m_rtTextureViews / Texture::uploadPixels), instead of Texture2D.Handle(...).
	VkDescriptorSetLayoutBinding bindings[] =
	{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
			.descriptorCount = 1,
			// Closest-hit's own texture lookup (g_textures[], binding 4) plus, now, the miss
			// shader's IBL cubemap sample (g_iblSpecular, binding 9, see getSkyColor()) -- both
			// reuse this same sampler.
			.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
		},
		{
			.binding = 4,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.descriptorCount = shaderio::kMaxRayTracingTextures,
			.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
		},
		// bindings 5-8: G-buffer render targets (albedo, normal+roughness, view-space depth, motion
		// vector), written by the raygen shader via RWTexture2D -- see
		// App::EnsureGBufferCapacity (App.cpp) for how they're created and
		// Shader::updateRayDescriptorSet for how they're bound each frame.
		{
			.binding = 5,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 6,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 7,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		{
			.binding = 8,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
		},
		// binding 9: image-based lighting environment cubemap (see IBLEnvironment.h), sampled by
		// rayMissMain (skybox + every bounce/shadow ray that escapes to open sky -- see
		// getSkyColor(), ray.slang) at mip 0 always; unlike the rasterizer's roughness->mip
		// approximation, ray tracing gets its roughness response for free from the existing
		// GGX-importance-sampled bounce direction (rayGenMain's path-tracing loop), so no mip chain
		// is needed here at all.
		{
			.binding = 9,
			.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR
		},
	};

	//	{ 0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR },
	//	{ 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1, VK_SHADER_STAGE_RAYGEN_BIT_KHR },
	//	{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1, VK_SHADER_STAGE_RAYGEN_BIT_KHR },


	{
		VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
		info.bindingCount = sizeof(bindings) / sizeof(*bindings);
		info.pBindings = bindings;

		VK_CHECK(vkCreateDescriptorSetLayout(context.device, &info, nullptr, &m_rtDescSetLayout));
	}

	VkDescriptorSetLayout layouts[] = { m_rtDescSetLayout };

	// closest-hit reads pushData too (indexBufferAddress, vertexBufferAddress, sceneInfoAddress for texId)
	VkPushConstantRange pushRange{};
	pushRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	pushRange.offset = 0;
	pushRange.size = sizeof(shaderio::RayPushData);

	VkPipelineLayoutCreateInfo info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	info.setLayoutCount = 1;
	info.pSetLayouts = layouts;
	info.pushConstantRangeCount = 1;
	info.pPushConstantRanges = &pushRange;

	VK_CHECK(vkCreatePipelineLayout(context.device, &info, nullptr, &m_rtPipelineLayout));

	{
		// Pool sizes are totals across every set the pool will ever allocate -- one set per
		// frame-in-flight slot (see Renderer::kMaxFramesInFlight), so each binding's count is multiplied by it.
		VkDescriptorPoolSize sizes[] =
		{
			{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, Renderer::kMaxFramesInFlight },
			// 5 STORAGE_IMAGE bindings per set now: binding 1 (legacy outputView) + bindings 5-8
			// (G-buffer targets).
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * Renderer::kMaxFramesInFlight },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, Renderer::kMaxFramesInFlight },
			{ VK_DESCRIPTOR_TYPE_SAMPLER, Renderer::kMaxFramesInFlight },
			// +1 per set: binding 9, the IBL environment cubemap, alongside binding 4's texture array.
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, (shaderio::kMaxRayTracingTextures + 1) * Renderer::kMaxFramesInFlight },
		};

		VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		poolInfo.maxSets = Renderer::kMaxFramesInFlight;
		poolInfo.poolSizeCount = sizeof(sizes) / sizeof(*sizes);
		poolInfo.pPoolSizes = sizes;

		VK_CHECK(vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &m_rtPool));
	}

	{
		VkDescriptorSetLayout setLayouts[Renderer::kMaxFramesInFlight];
		for (uint32 i = 0; i < Renderer::kMaxFramesInFlight; ++i)
			setLayouts[i] = m_rtDescSetLayout;

		VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		ai.descriptorPool = m_rtPool;
		ai.descriptorSetCount = Renderer::kMaxFramesInFlight;
		ai.pSetLayouts = setLayouts;

		VK_CHECK(vkAllocateDescriptorSets(context.device, &ai, m_rtDescSet));
	}

	{
		VkRayTracingPipelineCreateInfoKHR info{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
		info.stageCount = 3;
		info.pStages = stages;
		info.groupCount = 3;
		info.pGroups = groups;
		//		info.maxPipelineRayRecursionDepth = 2;	// primary ray + shadow ray ?
		info.maxPipelineRayRecursionDepth = 1;
		info.layout = m_rtPipelineLayout;

		VK_CHECK(vkCreateRayTracingPipelinesKHR(context.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &info, nullptr, &m_pipeline));
	}

	// Only needed to build the pipeline above -- the pipeline doesn't keep referencing them, and
	// nothing else ever destroys these (deinit() only knows about m_shader[]/m_pipeline/etc, not
	// these locals), so left as-is they leak every time a ray tracing Shader compiles or reloads.
	vkDestroyShaderModule(context.device, raygenModule, nullptr);
	vkDestroyShaderModule(context.device, missModule, nullptr);
	vkDestroyShaderModule(context.device, chitModule, nullptr);
}

void Shader::updateRayDescriptorSet(VkDevice device, VkAccelerationStructureKHR tlas, VkImageView outputView,
	VkBuffer sceneBuffer, const VkImageView* textureViews, uint32 textureViewCount, VkSampler sampler, uint32 frameSlot,
	VkImageView gbufferAlbedoView, VkImageView gbufferNormalRoughnessView, VkImageView gbufferViewZView, VkImageView gbufferMotionVectorView,
	VkImageView iblSpecularCubeView)
{
	ASSERT(frameSlot < Renderer::kMaxFramesInFlight);
	VkDescriptorSet descSet = m_rtDescSet[frameSlot];

	if (descSet == VK_NULL_HANDLE || m_rtPool == VK_NULL_HANDLE || m_rtDescSetLayout == VK_NULL_HANDLE)
		return;
	if (tlas == VK_NULL_HANDLE || outputView == VK_NULL_HANDLE || sceneBuffer == VK_NULL_HANDLE)
		return;
	if (gbufferAlbedoView == VK_NULL_HANDLE || gbufferNormalRoughnessView == VK_NULL_HANDLE
		|| gbufferViewZView == VK_NULL_HANDLE || gbufferMotionVectorView == VK_NULL_HANDLE)
		return;
	// IBL (see IBLEnvironment.h) loads lazily on the first setupScene1()/setupScene2()/loadScene()
	// call, same reasoning as those functions' own g_ibl.load() guard -- until then, this binding
	// has nothing valid to point at, so hold off on the whole descriptor set (like every other
	// required-resource guard above) rather than leaving it unwritten.
	if (iblSpecularCubeView == VK_NULL_HANDLE)
		return;
	// Slot 0 (Renderer::init()'s own procedural fallback texture) is always registered before this
	// is ever called; slots beyond it may still be empty (fewer textures loaded than
	// shaderio::kMaxRayTracingTextures) -- filled in below with slot 0's view, since this binding has
	// no PARTIALLY_BOUND flag and every descriptor must be valid.
	if (textureViews[0] == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE)
		return;

	VkWriteDescriptorSetAccelerationStructureKHR asInfo{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		.accelerationStructureCount = 1,
		.pAccelerationStructures = &tlas,
	};

	VkDescriptorImageInfo image{};
	image.imageView = outputView;
	image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkDescriptorBufferInfo buffer{};
	buffer.buffer = sceneBuffer;
	buffer.range = VK_WHOLE_SIZE;

	VkDescriptorImageInfo samplerInfo{};
	samplerInfo.sampler = sampler;

	std::vector<VkDescriptorImageInfo> textureInfo(textureViewCount);
	for (uint32 i = 0; i < textureViewCount; ++i)
	{
		textureInfo[i].imageView = (textureViews[i] != VK_NULL_HANDLE) ? textureViews[i] : textureViews[0];
		textureInfo[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}

	VkDescriptorImageInfo gbufferAlbedoInfo{ .imageView = gbufferAlbedoView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo gbufferNormalRoughnessInfo{ .imageView = gbufferNormalRoughnessView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo gbufferViewZInfo{ .imageView = gbufferViewZView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo gbufferMotionVectorInfo{ .imageView = gbufferMotionVectorView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo iblSpecularInfo{ .imageView = iblSpecularCubeView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

	VkWriteDescriptorSet writes[10]{};

	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].pNext = &asInfo;
	writes[0].dstSet = descSet;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = descSet;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[1].pImageInfo = &image;

	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = descSet;
	writes[2].dstBinding = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[2].pBufferInfo = &buffer;

	writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet = descSet;
	writes[3].dstBinding = 3;
	writes[3].descriptorCount = 1;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	writes[3].pImageInfo = &samplerInfo;

	writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[4].dstSet = descSet;
	writes[4].dstBinding = 4;
	writes[4].descriptorCount = textureViewCount;
	writes[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	writes[4].pImageInfo = textureInfo.data();

	writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[5].dstSet = descSet;
	writes[5].dstBinding = 5;
	writes[5].descriptorCount = 1;
	writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[5].pImageInfo = &gbufferAlbedoInfo;

	writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[6].dstSet = descSet;
	writes[6].dstBinding = 6;
	writes[6].descriptorCount = 1;
	writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[6].pImageInfo = &gbufferNormalRoughnessInfo;

	writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[7].dstSet = descSet;
	writes[7].dstBinding = 7;
	writes[7].descriptorCount = 1;
	writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[7].pImageInfo = &gbufferViewZInfo;

	writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[8].dstSet = descSet;
	writes[8].dstBinding = 8;
	writes[8].descriptorCount = 1;
	writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writes[8].pImageInfo = &gbufferMotionVectorInfo;

	writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[9].dstSet = descSet;
	writes[9].dstBinding = 9;
	writes[9].descriptorCount = 1;
	writes[9].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	writes[9].pImageInfo = &iblSpecularInfo;

	vkUpdateDescriptorSets(device, 10, writes, 0, nullptr);
}

bool Shader::dispatchRayTracing(VkCommandBuffer cmd, uint32 frameSlot, VkCommandPool transientCmdPool,
	VkExtent2D size, const shaderio::RayPushData& pushData)
{
	if (!m_pipeline)
		return false;

	ASSERT(frameSlot < Renderer::kMaxFramesInFlight);

	extern Renderer g_renderer;
	const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps = g_renderer.m_rtProps;
	VkDevice device = g_renderer.m_context.getDevice();

	ASSERT(rtProps.shaderGroupHandleSize);
	const uint32 handleSize = rtProps.shaderGroupHandleSize;
	const uint32 handleSizeAligned = (uint32)utils::alignUp(handleSize, rtProps.shaderGroupHandleAlignment);
	const uint32 regionStride = (uint32)utils::alignUp(handleSizeAligned, rtProps.shaderGroupBaseAlignment);

	// The SBT only depends on this shader's own pipeline (built once), so only build it once too
	// -- rebuilding every frame would leak a new buffer per frame (m_sbt.m_data overwritten with
	// no deinit of the previous allocation).
	if (m_sbt.m_data.buffer == VK_NULL_HANDLE)
	{
		VkCommandBuffer sbtCmd = utils::beginSingleTimeCommands(device, transientCmdPool);

		constexpr uint32 groupCount = 3; // raygen, miss, hit

		std::vector<uint8> handles(groupCount * handleSize);
		VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device, m_pipeline, 0, groupCount, (uint32)handles.size(), handles.data()));

		std::vector<uint8> sbtData(groupCount * regionStride);
		for (uint32 i = 0; i < groupCount; ++i)
			memcpy(sbtData.data() + i * regionStride, handles.data() + i * handleSize, handleSize);

		// need to be outside vkCmdBeginRendering/vkCmdEndRendering
		m_sbt.initRaw(g_renderer.m_allocator, sbtCmd, sbtData.data(), sbtData.size(), 1, Buffer::eBT_ShaderBindingTable);

		utils::endSingleTimeCommands(sbtCmd, device, transientCmdPool, g_renderer.m_context.getGraphicsQueue().queue);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0, 1, &m_rtDescSet[frameSlot], 0, nullptr);

	const VkStridedDeviceAddressRegionKHR raygenRegion{ .deviceAddress = m_sbt.m_data.address + 0 * regionStride, .stride = handleSizeAligned, .size = handleSizeAligned };
	const VkStridedDeviceAddressRegionKHR missRegion{ .deviceAddress = m_sbt.m_data.address + 1 * regionStride, .stride = handleSizeAligned, .size = handleSizeAligned };
	const VkStridedDeviceAddressRegionKHR hitRegion{ .deviceAddress = m_sbt.m_data.address + 2 * regionStride, .stride = handleSizeAligned, .size = handleSizeAligned };
	const VkStridedDeviceAddressRegionKHR callableRegion{};

	vkCmdPushConstants(cmd, m_rtPipelineLayout,
		VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
		0, sizeof(pushData), &pushData);

	// cannot be inside vkCmdBeginRendering/vkCmdEndRendering
	// filePath.c_str() (not a generic string) so a VK_EXT_device_fault report's last-reached
	// checkpoint says which of g_ray/g_laser/g_interaction's dispatches actually faulted --
	// filePath outlives this call (member of the long-lived global Shader instance), matching
	// vkCmdSetCheckpointNV's requirement that the marker stay valid until retrieved.
	if (vkCmdSetCheckpointNV)
		vkCmdSetCheckpointNV(cmd, filePath.c_str());
	vkCmdTraceRaysKHR(cmd, &raygenRegion, &missRegion, &hitRegion, &callableRegion, size.width, size.height, 1);
	if (vkCmdSetCheckpointNV)
		vkCmdSetCheckpointNV(cmd, "after vkCmdTraceRaysKHR");

	return true;
}

void Shader::init(EventContext& context)
{
	compile();
	create(context);
}

void Shader::reload(EventContext& context)
{
	deinit(context);
	init(context);
}

void Shader::deinit(EventContext& context)
{
	if (m_rtPool != VK_NULL_HANDLE)
	{
		// Destroying the pool implicitly frees every descriptor set allocated from it.
		vkDestroyDescriptorPool(context.device, m_rtPool, nullptr);
		m_rtPool = {};
	}
	if (m_rtDescSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(context.device, m_rtDescSetLayout, nullptr);
		m_rtDescSetLayout = {};
	}
	if (m_rtPipelineLayout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(context.device, m_rtPipelineLayout, nullptr);
		m_rtPipelineLayout = {};
	}
	for (VkDescriptorSet& descSet : m_rtDescSet)
		descSet = {};
//	m_rtDescSet = {};

	vkDestroyShaderEXT(context.device, m_shader[0], nullptr); m_shader[0] = {};
	vkDestroyShaderEXT(context.device, m_shader[1], nullptr); m_shader[1] = {};
	vkDestroyShaderEXT(context.device, m_shader[2], nullptr); m_shader[2] = {};
	vkDestroyPipeline(context.device, m_pipeline, nullptr); m_pipeline = {};

	// m_sbt's shader group handles are baked from m_pipeline's own groups (see
	// dispatchRayTracing()'s "built lazily ... once" comment, written back when this class only
	// ever built a pipeline a single time) -- now that reload() can rebuild m_pipeline as a
	// different VkPipeline object, leaving m_sbt non-null here makes dispatchRayTracing()'s "already
	// built?" check skip rebuilding it, binding a stale SBT (handles from the OLD, now-destroyed
	// pipeline) against the NEW one. Deinit it here too so the next dispatchRayTracing() rebuilds it
	// from the new pipeline's real handles.
	m_sbt.deinit(context);

	code[0].clear();
	code[1].clear();
	code[2].clear();
}

// @param format e.g. "test\n"
void Log(const char* format, ...)
{
	char buffer[1024 * 10];
	va_list args;
	va_start(args, format);

	vsnprintf(buffer, sizeof(buffer), format, args);

	va_end(args);
	OutputDebugStringA(buffer);
	// Was OutputDebugStringA-only -- invisible without a debugger attached, which meant Slang
	// compiler diagnostics (warnings/errors from shader recompiles, e.g. after the RTXDI
	// integration) went unseen. "%s" (not buffer directly) since buffer already had vsnprintf
	// applied and may itself contain literal '%' from the compiler's own message text.
	LOGW("%s", buffer);
}

