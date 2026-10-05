#ifndef MATHLIB_PLANE3_H
#define MATHLIB_PLANE3_H

#include "Math.h"
#include <float.h>												// FLT_MAX

#include <glm/glm.hpp>


class CPlane3
{
public:

	//!< has to be normalized
	float3 m_vNormal;
	//!< distance to (0,0,0)
	float m_fDist;

	// default constructor (no initialization for performace reasons)
	CPlane3()
	{
	}

	// constructor get plane from 3 points
	CPlane3( const float3 &vNormal, const float fDist )
		:m_vNormal(vNormal), m_fDist(fDist)
	{
//		ASSERT(vNormal.IsNormalized());
	}

	//! constructor get plane from normal and a point on the plane
	CPlane3( const float3 &vNormal, const float3 &vPointOnPlane )
		:m_vNormal(vNormal)
	{
		m_fDist = -glm::dot(vPointOnPlane, m_vNormal);

		//		ASSERT(vNormal.IsNormalized());
	}

	//! constructor get plane from 3 points
	CPlane3( const float3 &p1, const float3 &p2, const float3 &p3 )
	{
		float3 vA=p2-p1;
		float3 vB=p3-p1;

		m_vNormal = normalize( glm::cross(vA, vB) );

//		ASSERT(m_vNormal.IsNormalized());

		m_fDist = -glm::dot(p1, m_vNormal);
	}

	float GetPointDistance( const float3 &vPoint ) const
	{
		return glm::dot(vPoint, m_vNormal) + m_fDist;
	}

	//! get point nearest to (0,0,0)
	float3 GetPlanePoint() const
	{
		return -m_vNormal*m_fDist;
	}

	void Negate()
	{
		m_vNormal=-m_vNormal;
		m_fDist=-m_fDist;
	}

	void SetInvalid()
	{
		m_vNormal= float3(0,0,0);m_fDist=FLT_MAX;
	}

	bool IsValid() const
	{
		return m_vNormal== float3(0,0,0) || m_fDist==FLT_MAX;
	}
};


inline bool IntersectRayPlane( const float3 vPos, const float3 vDir, const CPlane3 p, float &t, float3 &q )
{
	float fDiv = glm::dot(p.m_vNormal, vDir);

	if(fDiv==0.0f)
		return false;

	t = (p.m_fDist + glm::dot(p.m_vNormal, vPos)) / fDiv;
	q = vPos - t * vDir;
	return true;
}

#endif // MATHLIB_PLANE3_H