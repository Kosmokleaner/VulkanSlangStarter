
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
