
# VVerse V 0.37

A Vulkan renderer/engine experiment by Martin Mittring

"V" as a nod to Vulkan (V is Roman 5, and Vulkan is basically OpenGL 5).
"Verse" as this might be leading to some virtual world like thing.

Platform: Windows11 x64, later Linux

Run with commandline parameter: --test for running the headless (no GPU) tests, --selftest for the
scripted real-GPU/screenshot-compare tests (see TestMode.h)

### todo (next ones on top):
* high fps with pretty bright content (Pica pica)
* light and camera need geometry for ray intersection, => select, property edit
* AppendArena copy & paste, undo
* AppendArena txt version
  * copy camera transform
* replace_me mesh if mesh cannot be found e.g. plane
* load scene on startup
* [minor, cosmetic] laser/gizmo debug lines draw on top of the NRD Validation Overlay's debug grid
  (they're a separate ImGui overlay pass, unaware of it) -- harmless, just visually noisy while
  using that overlay
* Scene Outliner UI
* material grid for scene
* laser should use proper importance sampling
* fix couch materials and textures
* Eye Adaptation
* [polish] consider writing shadercache after shaders have been compiled, not just on normal app quit
* Stable Sample for fps and ms, no VSync >120fps
* error handling if glb cannot be found
* edit objects (delete, insert)
  * persistent scene, networked
* Laser game
* [optimization] rasterizer should use culling (CPU for now)
* [feature] support normal maps
* minor: fix bug with MAX_TEMPORAL_ACCUM 1
* organic sphere primitive
* temporal filter to TemporalAA, pixel frame jitter or better
* better shader error handling e.g. comment s2h include 
* [optimization] texture mips for ray tracing
* Copy from KosmoRay2: DataDefinitions, CTangentSpaceCalculation, HighResolutionClock, pathtracing
* glass material with refraction
  * subsurface scattering thin/thick
* fix WindowPersist
* test quit at various places in the engine with clean shutdown
* Callable Shaders https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/samples/13-callable-shader/
* Shader Execution Reorder https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/samples/11-shader-execution-reorder/ 
* ray cone https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/samples/19-ray-differentials/
* shader error with double click in debug window
* fix __debugbreak when window appears, disable docking? repro: drag window
* Combine with DearGPU?
* optimize mesh (each vertex is unwelded)

### done (add at top, some element have been added at bottom):
* [feature] setup #3: SEED's Pica Pica mini diorama (resources/pica-pica-mini-diorama-01.glb) on a
  small ground plane -- "setup #3" UI button, also covered by --selftest. All nine pica-pica-* asset
  folders (https://github.com/SEED-EA/pica-pica-assets, CC BY-NC 4.0 -- see
  resources/pica-pica-assets/LICENSE_AND_ATTRIBUTION.md) are vendored under resources/pica-pica-assets/
  for future use, but only the mini diorama is wired in so far. Its source .fbx was converted to .glb
  via Blender run headless (--background + bpy.ops.import_scene.fbx/export_scene.gltf); the FBX's
  parent-child hierarchy wasn't carrying through to glTF nodes correctly (a child mesh rendered
  floating in the wrong place), fixed by baking each object's world transform into its mesh data and
  clearing parenting before export, done via direct bpy.data access rather than the transform_apply()/
  parent_clear() operators (which need a 3D viewport context --background doesn't have).
* refactor: renamed class MinimalLatest to App, and file minimal_latest.cpp/.h to App.cpp/.h
* [feature] screenshot function (Settings window "Screenshot" button -> screenshots/screenshot_NNNN.png,
  lossless PNG via vendored stb_image_write.h, folder created + index resumed automatically) and
  --selftest: a scripted, deterministic real-GPU test (setup#1/#2, save, load, each twice) that
  screenshots and compares against checked-in TestData/gold_*.png (EA SEED Gigi-style exact-match,
  missing gold image bootstrapped from the run instead of failing) -- see TestMode.h/
  App::runSelfTest(). Scene file name is now an editable UI field (was stubbed out
  commented-out before). See AppendArena.h's new top comment for the save/load backward-compat rule
  this exists to help enforce.
* get rid of g_BSPTransform -- updateLights() now finds the BSP level's Entity in
  g_scene.m_sceneObjects (matching hndRenderComp) and reads its transform live, instead of a
  separate global cached once at scene setup and never updated again
* fix AAJitter toggle
* [feature] HDR Envmap from D:\oldCode\Hive\MasterCD\Textures\Cubemaps
* fix: if shader was working once, error does not show up
* [minor optimization] compute GetTemporalJitter() on CPU
* remove glm::, rename to float2...?
* 3d gizmo
* NRD: split diffuse/specular (REBLUR_DIFFUSE_SPECULAR, two bridge channels) -- fused-signal
  denoising was visibly smearing/ghosting specular highlights under motion, the concrete case the
  earlier "defer until needed" note was waiting for. Scope boundary: only the direct/NEE specular
  BRDF lobe (bounce 0/1) gets its own channel; indirect/GI-bounced light still folds into the
  diffuse channel regardless of which lobe produced it -- a full per-lobe path split through the
  bounce loop and GI reservoir was judged too large/risky for the value it'd add right now. The
  specular channel also gets no GI/EMA temporal smoothing of its own (NRD's own denoiser is the
  right tool for that, given it does virtual-motion reprojection the EMA reservoir never did). Its
  "hit distance" is a luminance-weighted average distance-to-light, standing in for a true
  specular-reflection-ray hit distance NRD's virtual-motion reprojection wants (there's no traced
  reflection ray backing this NEE term) -- a documented approximation, not exact REBLUR semantics.
* fixed three real NRD integration bugs found while chasing a "flickering/unusable" Realtime image:
  (1) a half-pixel bug in the G-buffer motion vector (`ray.slang`) that was silently preventing NRD
  from ever holding temporal history (every pixel looked disoccluded, every frame, even with a
  static camera) -- found via NRD's own `OUT_VALIDATION` debug overlay, now exposed as a permanent
  "NRD Validation Overlay" checkbox (Settings) for future denoiser debugging; (2) the NRD-denoised
  background getting corrupted with stale data (NRD never writes pixels outside its
  `denoisingRange`, `nrd_resolve.slang` was blindly overwriting them anyway); (3) periodic vertical
  banding from telling NRD every bound texture was exactly viewport-sized (`CommonSettings::
  resourceSize`) when they were actually sized to this integration's larger grow-only capacity,
  aliasing NRD's own internal bilinear resampling. Also added a "ReSTIR DI (RTXDI)" checkbox to
  isolate RTXDI's own contribution to noise from everything else when debugging denoising quality
  (falls back to plain uniform-light-picking when off).
* NRD (NVIDIA Real-time Denoiser) real-time dispatch: a "NRD Realtime Denoise" checkbox (Settings)
  switches the ray tracer from the default freeze-and-accumulate "Reference" behavior to REBLUR_DIFFUSE
  denoising a fresh single-sample frame every frame, staying clean while the camera moves. See
  NrdIntegration.h/.cpp and the extern section below.
* G-buffer (albedo, normal+roughness, view-space depth, motion vector) as real render targets,
  with VisualizeMode debug views for each
* move/rotate scene objects (3D gizmo via ImGuizmo)
* load/save cameras
* load/save lights
* camera to scene, no longer serialize in Settings.ini
* light to scene, no longer serialize in Settings.ini
* make GenHandle 32+32 bit
* polish select object
* V0.37
* --- switched done order ---
* render 3d mesh
* integrate S2H
* window/app icon
* fix window aspect ratio
* fixed S2H background
* full screen pass (S2H background or S2H), custom shader
* remove USE_COMPUTE
* removed exception handling
* remove need for runtime type info (used by DBG_VK_NAME)
* ray tracing
  old: VK_NVX_raytracing, NV only: VK_NV_ray_tracing, best: VK_KHR_ray_tracing
  https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/en/courses/18_Ray_tracing/02_Acceleration_structures.adoc
  https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/ray_tracing_basic/ray_tracing_basic.cpp
* resources should auto destroy themselves e.g. buffer, mesh, shader
* fix toggle m_rayTracing
* multiple mesh objects, add ground plane
* add ground plane
* rename globals.h to global.h
* shader hot reload
* VSPS shading like ray tracing
* per object material for ray tracing and VSPS
* fix gfx PositionWS
* background color for ray and gfx is not the same (still some tonemapper difference)
* fix task bar icon
* unify 4 space tabs
* automated tests for RHI
* path tracing
* printout to screen if there is error (e.g. gfx draw)
* faster shader compile (cache, only compile files that changed)
* support include in shader cache
* UI to edit materials
* PBR materials (polished mirror, plastic, rough metal) 
* UI to edit materials
* fix relative mouse movement with Wacom tablet
* load materials from GLTF file
* line light and ray tracing to generate laser line light
* changes to textures without such bad texture compression artifacts
* expose 3 VSyncModes in UI, see if(mailboxSupported)
* dynmaic TLAS
* fix couch pillow visual
* use Settings.ini for other than window properties e.g. light pos, color, ...
* persistent camera pos/orientation
* atmospheric fog
* V0.35 ReSTIR DI
* serialize material name
* single append file scene
* serialize texture

===============================================

### extern (part of the repo unless stated)
* VulkanSDK 1.4.357.0 (not part of repo)
* glfw3.lib 3.3.4
* ImGUI (docking branch) v1.92.8
* stb_image 2.30
* Volk 
* GLM
* VMA (Vulkan Memory Allocator)
* slang-2026.13.1-windows-x86_64
* [ShaderToHuman (S2H)](https://github.com/electronicarts/ShaderToHuman)
* tinyGLTF
* hash-library (sha256 http://create.stephan-brumme.com/disclaimer.html)
* [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) v1.92.5 WIP
* [NRD (NVIDIA Real-time Denoiser)](https://github.com/NVIDIA-RTX/NRD) commit bf87718 (2026-09-10) --
  built locally (CMake + dxc, `NRD_STATIC_LIBRARY=ON`, `NRD_NORMAL_ENCODING=4`), only the compiled
  `NRD.lib`/`ShaderMakeBlob.lib` (Debug+Release) + public headers are vendored under extern/NRD (its
  shaders are pre-embedded SPIR-V/DXIL/DXBC in the lib, so there's no separate shader source to
  build); driven directly against `NRD.h`'s core API (no NRI) from NrdIntegration.h/.cpp
* mech_m-6k.glb from https://sketchfab.com/3d-models/mech-m-6k-6eba9b1addb7411a85b0c86e41a39a2e
  license: https://creativecommons.org/licenses/by/4.0/
* pica-pica-mini-diorama-01.glb (setup #3) and the rest of resources/pica-pica-assets/ from
  https://github.com/SEED-EA/pica-pica-assets (SEED, Electronic Arts), converted from the original
  .fbx via Blender -- license: https://creativecommons.org/licenses/by-nc/4.0/ (CC BY-NC 4.0,
  non-commercial), see resources/pica-pica-assets/LICENSE_AND_ATTRIBUTION.md
* todo: other glb in the resources folder
* Quake 3 BSP loading based on http://www.gametutorials.com/Tutorials/OpenGL/Quake3Format.htm 
* BSP levels: https://ws.q3df.org/map/ospdm2
 
personal google docs [document](https://docs.google.com/document/d/1pFKq6ALe9zLD0PdL-Dg-UyAAXa7jnLTIR3wn0CPyL90/edit?tab=t.0):

### random notes
* filesystem requires c++ 17+
* span requires c++ 20+
* RenderDoc does not support descriptorheap yet so cannot use it
* NSight

### useful references
* [Digital Dragons 2021: Adam Sawicki - Efficient Use of GPU Memory in Modern Games](https://youtu.be/ML0YC77bSOc?is=BsXDu_daVjyW1nQC)

