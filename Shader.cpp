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
		ASSERT(0);
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

