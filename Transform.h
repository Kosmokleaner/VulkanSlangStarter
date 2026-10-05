#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define PI 3.14159265f

// SRT (Scale Rotation Translation)
class Transform
{
public:
	// constructor
	Transform(float3 InPos = float3(0, 0, 0))
	{
		SetPos(InPos);
		rot = glm::quat(1, 0, 0, 0);
		scale = float3(1, 1, 1);
	}

//	Transform& operator=(const Transform& rhs) { *this = rhs; }

	void SetPos(float3 InPos)
	{
		position = InPos;
	}

	void Move(float3 delta)
	{
		position += delta;
	}

	float2 GetAbsRotation() const
	{
		float3 forward = GetForward();

		return float2(atan2f(forward.x, forward.z) + (float)PI, asinf(forward.y));
	}

	// relative
	// @param dx: positive rotates to the right, in radiants (0..2*PI is 360)
	// @param dy: positive rotates upwards, in radiants (0..2*PI is 360)
	void Rotate(float dx, float dy)
	{
		float2 w = GetAbsRotation();

		w.x += dx;
		w.y += dy;

		SetRotation(w);
	}

	// absolute
	void SetRotation(float2 w)
	{
		const float yborder = 0.00001f;

		// clamp top
		w.y = glm::max(w.y, -PI * 0.5f + yborder);
		// clamp bottom
		w.y = glm::min(w.y, PI * 0.5f - yborder);

		float3 euler(w.y, w.x, 0);
		rot = glm::quat(euler);
		//        Rot.SetRot(XMQuaternionRotationRollPitchYaw(w.y, w.x, 0));
	}
	/*
		void DumpState(const char* Name) const
		{
			char str[256];

			sprintf_s(str, sizeof(str), "\n%s.SetPos(%ff, %ff, %ff);\n", Name, position[0], position[1], position[2]);
			OutputDebugStringA(str);

			Rot.DumpState(Name);
		}
	*/
	// camera center
	float3 GetPos() const { return position; }

	void SetScale(float3 InScale) { scale = InScale; }

	float3 GetScale() const { return scale; }

	// @return forward vector, is normalized
	float3 GetForward() const
	{
		return rot * float3(0, 0, -1);
	}

	// @return up vector, is normalized
	float3 GetUp() const
	{
		return GetDirY();
	}

	// @return right vector, is normalized
	float3 GetDirX() const
	{
		return rot * float3(1, 0, 0);
	}
	// @return up vector, is normalized
	float3 GetDirY() const
	{
		return rot * float3(0, 1, 0);
	}
	// @return back vector, is normalized
	float3 GetDirZ() const
	{
		return rot * float3(0, 0, 1);
	}

	// @return worldToObject aka objectFromWorld
	glm::mat4 GetObjectMatrix() const
	{
		// todo scale
		return glm::translate(glm::mat4(1.0f), position) * glm::scale(glm::mat4(1.0f), scale) * glm::mat4(rot);
	}
	glm::mat4 GetViewMatrix() const
	{
		// computes world To view
		return glm::lookAtLH(position, (position + GetForward()), GetUp());
	}

	Transform& operator*=(const Transform& rhs) { *this = *this * rhs; return *this; }

	// y = M * x
	// e.g. if this is n object matrix, b is in object space and the output is in world space
	float3 TransformPosition(const float3& x) const
	{
		return position + rot * x;
	}

	// multiply, b is applied first (todo: comment was flagged by AI)
	friend Transform operator*(const Transform& a, const Transform& b)
	{
		Transform Ret;

		// The result represents the rotation a followed by the rotation b to be consistent with XMMatrixMulplity concatenation
		Ret.rot = a.rot * b.rot;
		Ret.position = a.position + a.rot * b.position;

		return Ret;
	}

	// rotation as quaternion (todo: normalized?)
	glm::quat rot;
	// non uniform scale or mirroring
	float3 scale;
	// position
	float3 position;
};

// todo: move
// input vector don't have to be normalized
// can be optimized
inline void GetOtherBaseVec(float3 in, float3 &a, float3 &b)
{
	float3 v = glm::normalize(in);			// normalized input allows stable asserts

	if (v.z < -0.5f || v.z > 0.5f)
	{
		a.x = v.z;
		a.y = v.y;
		a.z = -v.x;
	}
	else
	{
		a.x = v.y;
		a.y = -v.x;
		a.z = v.z;
	}

	b = glm::cross(in, a);
	b = glm::normalize(b);
	a = glm::cross(b, in);
	a = glm::normalize(a);

	//	ASSERT(fabs(a*b)<0.01f);
	//	ASSERT(fabs((*this)*a)<0.01f);
	//	ASSERT(fabs((*this)*b)<0.01f);
}

// cast like HLSL function
// todo: move
inline float asfloat2(uint32 x)
{
	return *(float*)&x;
}
inline uint32 asuint(float x)
{
	return *(uint32*)&x;
}
inline int32 asint(float x)
{
	return *(int32*)&x;
}
