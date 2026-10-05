
//#define PI 3.14159265
static const float PI = 3.14159265359f;

static const float INFINITE = 1e32F;

// Standard Rec.709 relative luminance -- same coefficients RtxdiBridge.slang's own
// RAB_GetLightSampleTargetPdfForSurface already inlines; factored out here so ReSTIR GI's
// reservoir resampling (gi_reservoir.slang) can reuse it instead of re-inlining the constant.
float Luminance(float3 c)
{
	return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

// Which descriptor-heap/texture-array index to sample: the material's own glTF texture if it has
// one, otherwise the debug "Image 1"/"Image 2" toggle (SceneInfo::texId) both shaders already
// exposed before per-material textures existed. Used by both shader.rast.slang (unbounded
// descriptor heap) and ray.slang (which additionally clamps the result to its fixed-size
// g_textures[] array, see kMaxRayTracingTextures).
uint resolveTextureId(uint materialTextureId, int fallbackSceneTexId)
{
	return (materialTextureId != kInvalidTextureId) ? materialTextureId : uint(fallbackSceneTexId);
}

// The baked IBL cubemaps (IBLEnvironment.h) came from an old CryEngine-era DirectX codebase
// (D:\oldCode\Hive\MasterCD\Shaders\SF_Base.fx's Phong(), which samples texCUBE with the raw
// world-space reflection/normal vector -- no adjustment applied there), which used that engine's
// classic Z-up world convention. This engine is Y-up, -Z-forward (glTF convention, see
// Transform::GetForward()). Z-up -> Y-up is a 90 degree rotation around X (right stays right, old
// "up" (Z) becomes new Y, old "forward" (Y) becomes new -Z) -- a real rotation, not a mirror, which
// matches the reported symptom. Fixing it here (instead of re-baking/re-processing the cubemap
// data) fixes every sampler at once: ray.slang's getSkyColor() (primary/bounce/shadow misses) and
// shader.rast.slang's diffuse/specular IBL lookups. This is the direction confirmed in RenderDoc:
// world down (0,-1,0) must land on the cubemap's -Z face. The other rotation direction
// (float3(dir.x, dir.z, -dir.y)) got the horizontal rotation right but was upside down -- world
// down landed on +Z instead.
float3 RotateCubemapLookupDir(float3 dir)
{
	return float3(dir.x, -dir.z, dir.y);
}

// Smooth windowed falloff for a Light (shader_io.h): 1 at the light's own position, 0 at and
// beyond radius, no divide-by-distance singularity when a surface is very close to the light.
// Used by both shader.rast.slang and ray.slang so every light (the single global one and every
// BSP level light, see g_lights in App.cpp) attenuates with distance the same way --
// without this, summing dozens of unattenuated lights (each contributing at full strength
// regardless of distance) blows every pixel out to solid white.
float EvalLightFalloff(float dist, float radius)
{
	float falloff = saturate(1.0f - dist / max(radius, 0.001f));
	return falloff * falloff;
}


float4x4 inverse(float4x4 input)
{
#define minor(a,b,c) determinant(float3x3(input.a, input.b, input.c))
	float4x4 cofactors = float4x4(
    minor(_22_23_24, _32_33_34, _42_43_44),
    -minor(_21_23_24, _31_33_34, _41_43_44),
    minor(_21_22_24, _31_32_34, _41_42_44),
    -minor(_21_22_23, _31_32_33, _41_42_43),

    -minor(_12_13_14, _32_33_34, _42_43_44),
    minor(_11_13_14, _31_33_34, _41_43_44),
    -minor(_11_12_14, _31_32_34, _41_42_44),
    minor(_11_12_13, _31_32_33, _41_42_43),

    minor(_12_13_14, _22_23_24, _42_43_44),
    -minor(_11_13_14, _21_23_24, _41_43_44),
    minor(_11_12_14, _21_22_24, _41_42_44),
    -minor(_11_12_13, _21_22_23, _41_42_43),

    -minor(_12_13_14, _22_23_24, _32_33_34),
    minor(_11_13_14, _21_23_24, _31_33_34),
    -minor(_11_12_14, _21_22_24, _31_32_34),
    minor(_11_12_13, _21_22_23, _31_32_33)
    );
#undef minor
	return transpose(cofactors) / determinant(input);
}


float4x4 ComputePerspectiveMatrix(float fovAngle, float aspect, float nearZ, float farZ)
{
	float tanHalfFov = tan(fovAngle * 0.5f);
	float fRange = farZ / (nearZ - farZ);

    // Right-handed perspective projection matrix (DirectX / NDC Z range [0, 1])
    // NOTE: this Z/W row is NOT the mathematically standard one (see the commented-out version
    // below) -- but GetPerspMatrix() (which calls this) is shared by vertexMain's forward transform,
    // and that transform only renders correctly with THIS row, not the standard one. Restored after
    // confirming the "correct" version breaks the vertex shader's forward projection entirely
    // (nothing renders). Whatever compensates for this elsewhere in the pipeline needs to be found
    // before this can actually be fixed -- don't change this row again without checking vertexMain.
	float4x4 proj = float4x4(
        1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f,
//        0.0f, 0.0f, farZ / (nearZ - farZ), -1.0f,
 //       0.0f, 0.0f, (nearZ * farZ) / (nearZ - farZ), 0.0f
        0.0f, 0.0f, 0, 0.1f,
        0.0f, 0.0f, 1.0f, 0.0f
    );
    
	return proj;
}

// reference for return mul(A, B);
float4x4 MultiplyMatrices(float4x4 A, float4x4 B)
{
	float4x4 result;
    
    // Extract columns of B to treat them as vectors for dot products
    // (Assuming standard column-major or row-major compatibility via explicit rows)
	result[0] = float4(
        dot(A[0], float4(B[0][0], B[1][0], B[2][0], B[3][0])),
        dot(A[0], float4(B[0][1], B[1][1], B[2][1], B[3][1])),
        dot(A[0], float4(B[0][2], B[1][2], B[2][2], B[3][2])),
        dot(A[0], float4(B[0][3], B[1][3], B[2][3], B[3][3]))
    );
    
	result[1] = float4(
        dot(A[1], float4(B[0][0], B[1][0], B[2][0], B[3][0])),
        dot(A[1], float4(B[0][1], B[1][1], B[2][1], B[3][1])),
        dot(A[1], float4(B[0][2], B[1][2], B[2][2], B[3][2])),
        dot(A[1], float4(B[0][3], B[1][3], B[2][3], B[3][3]))
    );
    
	result[2] = float4(
        dot(A[2], float4(B[0][0], B[1][0], B[2][0], B[3][0])),
        dot(A[2], float4(B[0][1], B[1][1], B[2][1], B[3][1])),
        dot(A[2], float4(B[0][2], B[1][2], B[2][2], B[3][2])),
        dot(A[2], float4(B[0][3], B[1][3], B[2][3], B[3][3]))
    );
    
	result[3] = float4(
        dot(A[3], float4(B[0][0], B[1][0], B[2][0], B[3][0])),
        dot(A[3], float4(B[0][1], B[1][1], B[2][1], B[3][1])),
        dot(A[3], float4(B[0][2], B[1][2], B[2][2], B[3][2])),
        dot(A[3], float4(B[0][3], B[1][3], B[2][3], B[3][3]))
    );
    
	return result;
}

float sqr(float x)
{
	return x * x;
}

float3 homAway(float4 p)
{
	return p.xyz / p.w;
}


// https://www.diva-portal.org/smash/get/diva2:1573746/FULLTEXT01.pdf
// Generates a seed for a random number generator from 2 inputs
uint initRand(uint val0, uint val1, uint backoff = 16)
{
	uint v0 = val0, v1 = val1, s0 = 0;
	
	[unroll]
	for (uint n = 0; n < backoff; n++)
	{
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}
	return v0;
}

// Returns a pseudorandom float in [0..1] from seed
float nextRand(inout uint rnd)
{
	rnd = (1664525u * rnd + 1013904223u);
	return float(rnd & 0x00FFFFFF) / float(0x01000000);
}
// @return float2(0..1, 0..1)
float2 nextRand2(inout uint rnd)
{
	return float2(nextRand(rnd), nextRand(rnd));
}
// can be optimized
// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaUnitDisk
// @return float2(-1..1, -1..1)
float2 nextRand2Disc(inout uint rnd)
{
	// 0..1
	float2 ret = nextRand2(rnd);
	float2 uOffset = ret * 2 - 1;

	if (uOffset.x == 0 && uOffset.y == 0)
		return float2(0, 0);

	float theta, r;
	if (abs(uOffset.x) > abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI * 4 * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI * 2 - PI * 4 * (uOffset.x / uOffset.y);
	}
	return r * float2(cos(theta), sin(theta));
}
// @return float3(-1..1, -1..1, 0..1), length()=1
float3 nextRand2CosineWeightedHemisphere(inout uint rnd)
{
	float2 disc = nextRand2Disc(rnd);

	return float3(disc.x, disc.y, sqrt(1.0f - disc.x * disc.x - disc.y * disc.y));
}

// @return normal must be normalized
void getOtherBaseVec(out float3 outA, out float3 outB, float3 normal)
{
//    float3 v = normalize(normal);
	float3 v = normal; // faster

	if (v.z < -0.5f || v.z > 0.5f)
	{
		outA.x = v.z;
		outA.y = v.y;
		outA.z = -v.x;
	}
	else
	{
		outA.x = v.y;
		outA.y = -v.x;
		outA.z = v.z;
	}

	outB = normalize(cross(v, outA));
	outA = normalize(cross(outB, v));
}

// @return normal not normalized
float3 getCosHemisphereSample(inout uint rnd, float3 normal)
{
	float3 u, v;
	getOtherBaseVec(u, v, normal);

	float3 tsNormal = nextRand2CosineWeightedHemisphere(rnd);

	return mul(transpose(float3x3(u, v, normal)), tsNormal);
}

// Inigo Quilez sphere ray intersection https://iquilezles.org/articles/intersectors
// sphere of size ra centered at point ce
float2 sphIntersect(float3 ro, float3 rd, float3 ce, float ra)
{
	float3 oc = ro - ce;
	float b = dot(oc, rd);
	float c = dot(oc, oc) - ra * ra;
	float h = b * b - c;
	if (h < 0)
		return float2(-1, -1); // no intersection
	h = sqrt(h);
	return float2(-b - h, -b + h);
}


// @return in linear or sRGB space ?
float3 filmicToneMapping(float3 color)
{
	color = max(0, color - 0.004f);
	color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
	return color;
}



// glTF Dielectric F0 is constant 0.04, metals use baseColor as F0
float3 GetF0(float3 baseColor, float metallic)
{
	return lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
}

// Same as GetF0(baseColor, metallic) above, but scales the dielectric (non-metal) 4% term by a
// non-physical "specular occlusion" approximation for hand-painted/flat-shaded content (BSP decals,
// Quake-era textures) where "black" represents ink/shadow rather than a real dielectric material --
// see MaterialData::specularOcclusionFromAlbedo's comment (shader_io.h) for the full rationale.
// Metals are unaffected (their F0 already comes straight from baseColor, which a dark albedo
// already suppresses on its own -- applying this on top would double-suppress them).
//
// specularOcclusionFromAlbedo <= 0 must behave exactly like the plain GetF0 above (the "disabled"
// case) -- written branchless (no `if`) since this runs per-bounce/per-light-sample in ray.slang's
// inner loops, over materials with wildly different values across neighboring rays/threads:
// `enabled` is 0 or 1 (a comparison result, not a control-flow branch) and the final lerp(1.0,
// rawOcclusion, enabled) resolves to exactly 1.0 (no occlusion at all) whenever disabled, REGARDLESS
// of what rawOcclusion evaluated to -- deliberately, so a literal (0,0,0) albedo under a disabled
// (0) threshold still reads as "unaffected", not "fully occluded" (the max(..., 1e-8f) below only
// exists to keep that intermediate division finite, its result is discarded by the lerp either way).
float3 GetF0(float3 baseColor, float metallic, float specularOcclusionFromAlbedo)
{
	float enabled = float(specularOcclusionFromAlbedo > 0.0f);
	float rawOcclusion = saturate(Luminance(baseColor) / max(specularOcclusionFromAlbedo, 1e-8f));
	float specOcclusion = lerp(1.0f, rawOcclusion, enabled);

	float3 dielectricF0 = float3(0.04f, 0.04f, 0.04f) * specOcclusion;
	return lerp(dielectricF0, baseColor, metallic);
}

// Importance-samples the GGX normal distribution function (Trowbridge-Reitz, same D_GGX below
// evaluates analytically) around `normal` -- classic NDF-only importance sampling (Karis, "Real
// Shading in Unreal Engine 4", 2013), not full VNDF (Heitz 2018): simpler, at the cost of extra
// noise at grazing angles, an acceptable tradeoff for a basic path tracer. Reflect the caller's
// view direction around the returned half-vector to get a roughness-distributed glossy bounce
// direction: tight around `normal` (near-mirror) for low roughness, spread towards a uniform
// hemisphere for high roughness.
// @return H, a unit half-vector
float3 importanceSampleGGX(inout uint rnd, float3 normal, float roughness)
{
	float2 u = nextRand2(rnd);

	float a = roughness * roughness;
	float phi = 2.0f * PI * u.x;
	float cosTheta = sqrt((1.0f - u.y) / (1.0f + (a * a - 1.0f) * u.y));
	float sinTheta = sqrt(saturate(1.0f - cosTheta * cosTheta));

	float3 tsHalf = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

	float3 u_, v_;
	getOtherBaseVec(u_, v_, normal);

	return normalize(mul(transpose(float3x3(u_, v_, normal)), tsHalf));
}

// GGX / Trowbridge-Reitz Normal Distribution Function (D)
float D_GGX(float NdotH, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float denom = (NdotH * NdotH * (a2 - 1.0f) + 1.0f);
	return a2 / (PI * denom * denom);
}

// Smith Joint G1 term (Schlick-GGX)
float G1_GGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;
	return NdotV / (NdotV * (1.0f - k) + k);
}

// Smith-GGX Visibility Function (V = G2 / (4 * NdotL * NdotV))
float V_GGX(float NdotV, float NdotL, float roughness)
{
	return 1.0f / (G1_GGX(NdotV, roughness) * G1_GGX(NdotL, roughness) + 1e-5f);
}

// Fresnel-Schlick
float3 F_Schlick(float3 f0, float VdotH)
{
	return f0 + (1.0f - f0) * pow(saturate(1.0f - VdotH), 5.0f);
}

// Minimum roughness to use when evaluating EvalGGXBRDF against a punctual (point/spot) light via
// next-event estimation. A true point light has zero angular size, so the GGX specular lobe of a
// low-roughness surface only lines up with it for a razor-thin sliver of view/surface angles --
// everywhere else the analytic term is correctly near-zero, which reads as "the highlight vanished
// / specular ignores the light" rather than what it actually is (a highlight too narrow to land on
// any given shading point). Real-time renderers commonly floor roughness for exactly this case
// (e.g. Frostbite's punctual-light roughness clamp) since a physically-sized light source would
// naturally widen the highlight the same way; this only affects the direct/NEE evaluation, not the
// true roughness used for GGX bounce-direction importance sampling (importanceSampleGGX), so
// reflections/bounces still use the material's real roughness.
static const float kMinPunctualLightRoughness = 0.045f;

float3 EvalGGXBRDF(float3 N, float3 V, float3 L, float3 baseColor, float metallic, float roughness)
{
	return EvalGGXBRDF(N, V, L, baseColor, metallic, roughness, 0.0f);
}

// Same as EvalGGXBRDF above, with GetF0's own specularOcclusionFromAlbedo parameter (see its
// comment) threaded through to the specular Fresnel term instead of always disabled.
float3 EvalGGXBRDF(float3 N, float3 V, float3 L, float3 baseColor, float metallic, float roughness, float specularOcclusionFromAlbedo)
{
	float3 H = normalize(V + L);
	float NdotL = saturate(dot(N, L));
	float NdotV = saturate(dot(N, V));
	float NdotH = saturate(dot(N, H));
	float VdotH = saturate(dot(V, H));

	if (NdotL <= 0.0f || NdotV <= 0.0f)
	{
		return 0.0f;
	}

	float3 f0 = GetF0(baseColor, metallic, specularOcclusionFromAlbedo);
	float3 F = F_Schlick(f0, VdotH);
	float D = D_GGX(NdotH, roughness);
	float V_term = V_GGX(NdotV, NdotL, roughness);

    // Specular microfacet component
	float3 specular = D * V_term * F;

    // glTF energy conservation: diffuse component scales down for metals
	float3 diffuse = (1.0f - metallic) * (baseColor / PI);

	return (diffuse + specular) * NdotL;
}

// Same BRDF as EvalGGXBRDF above, but returns the diffuse and specular lobes separately instead of
// their sum -- for callers that need to route each into its own NRD denoising channel (REBLUR_
// DIFFUSE_SPECULAR, see ray.slang's hdrSpecular) instead of one fused signal.
void EvalGGXBRDFSplit(float3 N, float3 V, float3 L, float3 baseColor, float metallic, float roughness, float specularOcclusionFromAlbedo, out float3 diffuseOut, out float3 specularOut)
{
	float3 H = normalize(V + L);
	float NdotL = saturate(dot(N, L));
	float NdotV = saturate(dot(N, V));
	float NdotH = saturate(dot(N, H));
	float VdotH = saturate(dot(V, H));

	if (NdotL <= 0.0f || NdotV <= 0.0f)
	{
		diffuseOut = 0.0f;
		specularOut = 0.0f;
		return;
	}

	float3 f0 = GetF0(baseColor, metallic, specularOcclusionFromAlbedo);
	float3 F = F_Schlick(f0, VdotH);
	float D = D_GGX(NdotH, roughness);
	float V_term = V_GGX(NdotV, NdotL, roughness);

	specularOut = (D * V_term * F) * NdotL;
	diffuseOut = ((1.0f - metallic) * (baseColor / PI)) * NdotL;
}

// P0 and P1: Tube line segment endpoints in world space
// surfacePos: Shaded pixel/vertex position in world space
// R: Reflection or light lookup direction vector
float3 GetTubeLightContribution(float3 P0, float3 P1, float3 surfacePos, float3 R)
{
	float3 L0 = P0 - surfacePos;
	float3 L1 = P1 - surfacePos;
	float3 Ld = L1 - L0;

	// Calculate projection parameter t along the tube segment
	float ldDotLd = dot(Ld, Ld);
	float t = (dot(R, L0) * dot(R, Ld) - dot(L0, Ld)) / (ldDotLd - sqr(dot(R, Ld)));

	// Clamp t to the actual finite length of the tube segment [0, 1]
	float3 L = L0 + saturate(t) * Ld;

	// Return closest vector light contribution
	return L;
}