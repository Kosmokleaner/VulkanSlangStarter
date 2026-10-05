// axis aligned bounding box (3-dim)

#ifndef MATHLIB_AABB3_H
#define MATHLIB_AABB3_H

#include "Math.h"

#include <glm/glm.hpp>
#include "Plane3.h"


inline bool IsNAN(float3 in)
{
	return _isnan(in.x) || _isnan(in.y) || _isnan(in.z);					// if any element is NAN, we consider the whole object being bad (NAN)
}

inline float sqr(float x ){ return x * x; }
inline double sqr(double x) { return x * x; }




class CAABB3
{
public:

	float3 m_vMin;
	float3 m_vMax;

	//! default constructor (no initialization for performace reasons)
	CAABB3()
	{
	}

	//! constructor
	CAABB3( const float3 &vMin, const float3 &vMax )
	{
		m_vMin=vMin;m_vMax=vMax;
	}

	//! constructor from vPoints[0..nCount-1]
	//! \param vPoints!=0
	//! \param nCount >0
	CAABB3( const float3 *vPoints, const int nCount )
	{
		ASSERT(vPoints);
		ASSERT(nCount);

		m_vMin = m_vMax = *vPoints++;

		for(int i=1;i<nCount;++i)
			Union(*vPoints++);
	}

	//!
	 CAABB3 GetTransformed( const glm::mat4 &mat ) const;

	//!
	float3 getCenter() const
	{
		return (m_vMin+m_vMax)*0.5f;
	}

	//!
	float3 GetSize() const
	{
		return (m_vMax-m_vMin);
	}

	//! \return axis index of the longest axis (or one of - if there are more)
	uint32 GetLongestAxis() const
	{
		float3 vExtend = m_vMax-m_vMin;

		if(vExtend.x>vExtend.y)
		{
			if(vExtend.x>vExtend.z)
				return 0;
			
			return 2;
		}

		if(vExtend.y>vExtend.z)
			return 1;
			
		return 2;
	}

	//!
	float3 GetRadialSize() const
	{
		return GetSize()*0.5f;
	}

	//!
	float3 &getMax()
	{
		return m_vMax;
	}

	//!
	float3 &getMin()
	{
		return m_vMin;
	}

	//!
	const float3 &getMax() const
	{
		return m_vMax;
	}

	//!
	const float3 &getMin() const
	{
		return m_vMin;
	}
	//! on the AABB counts as inside
	bool IsPointInside( const float3 &vPos ) const
	{
		for(int i=0;i<3;++i)
		{
			if(vPos[i]<m_vMin[i])
				return false;
			
			if(vPos[i]>m_vMax[i])
				return false;
		}

		return true;
	}

	//! with another AAB
	void Union( const CAABB3 &aabbOther )
	{
		for(int i=0;i<3;++i)
		{
			if(aabbOther.m_vMin[i]<m_vMin[i])
				m_vMin[i]=aabbOther.m_vMin[i];
			
			if(aabbOther.m_vMax[i]>m_vMax[i])
				m_vMax[i]=aabbOther.m_vMax[i];
		}
	}

	//! with a point
	void Union( const float3 &vPos )
	{
		for(int i=0;i<3;++i)
		{
			if(vPos[i]<m_vMin[i])
				m_vMin[i]=vPos[i];
			
			if(vPos[i]>m_vMax[i])
				m_vMax[i]=vPos[i];
		}
	}

	void SetInvalid()
	{
		m_vMin = float3(FLT_MAX,FLT_MAX,FLT_MAX);
		m_vMax = -m_vMin;
	}

	bool IsInvalid() const
	{
		float3 vSize=GetSize();

		return !(vSize.x>=0.0f && vSize.y>=0.0f && vSize.z>=0.0f);
	}

	bool IsVolume() const
	{
		float3 vSize=GetSize();

		return vSize.x>0.0f && vSize.y>0.0f && vSize.z>0.0f;
	}

	//!
	bool intersectRay( const float3 &vOrigin, const float3 &vDir, float3 &vHitPos ) const;

	//!
	bool intersectRay( const float3 &vOrigin, const float3 &vDir ) const;

	//! touching does not return true
	bool intersectAABB( const CAABB3 &rAABB ) const;

	//! touching does not return true
	//! \param fRadius2 = sqr(fRadius)
	bool intersectSphere2( const float3 &vMidPos, const double fRadius2 ) const;

	//! can be used for efficent Sphere AABB intersection test
	//! \param vPos
	double EuklidDistance2( const float3 vPos ) const
	{
		ASSERT(!IsNAN(vPos));

		double fDist2=0;

		for(int i=0;i<3;++i)
		{
			if(vPos[i]<m_vMin[i])
			{
				fDist2+=sqr((double)vPos[i]-(double)m_vMin[i]);
			}
			else if(vPos[i]>m_vMax[i])
			{
				fDist2+=sqr((double)m_vMax[i]-(double)vPos[i]);
			}
		}

		return fDist2;
	}

	//! use EuklidDistance2 if possible to avoid the sqrt
	//! \param vPos
	float EuklidDistance( const float3 vPos ) const
	{
		ASSERT(!IsNAN(vPos));

		return (float)sqrt(EuklidDistance2(vPos));
	}

	//! from http://agora.cubik.org/wiki/view/Scratch/BruceCollisionStuffFromGDAlgorithms
	//! at max 32 planes
	bool intersectFrustum( const CPlane3 *pPlanes, unsigned int &outClipMask, unsigned int inClipMask ) const
	{
		float3 m = getCenter();     // center of AABB
		float3 d = getMax() - m;    // half-diagonal

		unsigned int mk = 1;
		outClipMask = 0;                // init outclip mask

		while(mk <= inClipMask) // loop while there are active planes..
		{
			if(inClipMask&mk) // if clip plane is active...
			{
				float NP = (float)(d.x*fabs(pPlanes->m_vNormal.x)+d.y*fabs(pPlanes->m_vNormal.y)+d.z*fabs(pPlanes->m_vNormal.z));
				float MP = glm::dot(m, pPlanes->m_vNormal) + pPlanes->m_fDist;

				if((MP+NP) < 0.0f)
					return false; // behind clip plane
				if((MP-NP) < 0.0f)
					outClipMask |= mk;
			}
			mk+=mk; // mk = (1<<iter)
			pPlanes++; // next plane
		}
		return true; // AABB intersects frustum
	}

	// intersect with convex object defines by planes (no limit on the plane count)
	bool intersectConvex( const CPlane3 *pPlanes, unsigned int dwPlaneCount ) const
	{
		float3 m = getCenter();     // center of AABB
		float3 d = getMax() - m;    // half-diagonal

		for(unsigned int dwI=0;dwI<dwPlaneCount;++dwI,++pPlanes)
		{
			float NP = (float)(d.x*fabs(pPlanes->m_vNormal.x)+d.y*fabs(pPlanes->m_vNormal.y)+d.z*fabs(pPlanes->m_vNormal.z));
			float MP = glm::dot(m, pPlanes->m_vNormal) + pPlanes->m_fDist;

			if((MP+NP) < 0.0f)
				return false;										// behind clip plane
	    }
		return true;                            // AABB intersects frustum
	}
	
	// make the aabb bigger (positive input value) or smaller (negative input value)
	void Inflate( const float fAmount )
	{
		m_vMin-=float3(fAmount,fAmount,fAmount);
		m_vMax+=float3(fAmount,fAmount,fAmount);
	}

	double GetVolume() const
	{
		float3 vSize=GetSize();

		return (double)vSize.x * (double)vSize.y * (double)vSize.z;
	}

	float3 GetPoint( const uint32 dwPointNo ) const
	{
		switch(dwPointNo)
		{
			case 0:	return float3(m_vMin.x,m_vMin.y,m_vMin.z);
			case 1:	return float3(m_vMax.x,m_vMin.y,m_vMin.z);
			case 2:	return float3(m_vMax.x,m_vMax.y,m_vMin.z);
			case 3:	return float3(m_vMin.x,m_vMax.y,m_vMin.z);

			case 4:	return float3(m_vMin.x,m_vMin.y,m_vMax.z);
			case 5:	return float3(m_vMax.x,m_vMin.y,m_vMax.z);
			case 6:	return float3(m_vMax.x,m_vMax.y,m_vMax.z);
			case 7:	return float3(m_vMin.x,m_vMax.y,m_vMax.z);

			default: ASSERT(0); return float3();
		}
	}

	static void GetEdgeNo( const uint32 dwEdgeNo, uint32 &dwA, uint32 &dwB )
	{
		switch(dwEdgeNo)
		{
			case  0:	dwA=0;dwB=1;break;
			case  1:	dwA=1;dwB=2;break;
			case  2:	dwA=2;dwB=3;break;
			case  3:	dwA=3;dwB=0;break;

			case  4:	dwA=4;dwB=5;break;
			case  5:	dwA=5;dwB=6;break;
			case  6:	dwA=6;dwB=7;break;
			case  7:	dwA=7;dwB=4;break;

			case  8:	dwA=0;dwB=4;break;
			case  9:	dwA=1;dwB=5;break;
			case 10:	dwA=2;dwB=6;break;
			case 11:	dwA=3;dwB=7;break;

			default: ASSERT(0);
		}
	}

	//! not too much optimized
	//! \return plane count (0) if point is inside the object)
	uint32 GetEdgesToPoint( const float3 vPos, float3 outEdgeA[12], float3 outEdgeB[12] ) const
	{
		if(IsPointInside(vPos))
			return 0;

		uint32 dwRet=0;
		float3 vMid = getCenter();

		for(uint32 dwEdge=0;dwEdge<12;++dwEdge)
		{
			uint32 dwA,dwB;

			GetEdgeNo(dwEdge,dwA,dwB);

			float3 vA=GetPoint(dwA),vB=GetPoint(dwB);

			CPlane3 plane(vA,vB,vPos);

			if(plane.GetPointDistance(vMid)>0)
			{
				plane.Negate();
				ASSERT(plane.GetPointDistance(vMid)<=0);
			}

			bool bAcceptPlane=true;

			// check new plane against all points
			for(uint32 dwP=0;dwP<8;++dwP)
			{
				if(dwP==dwA || dwP==dwB)
					continue;

				float3 vCorner = GetPoint(dwP);

				if(plane.GetPointDistance(vCorner)>0)
				{
					bAcceptPlane=false;
					break;
				}
			}

			if(bAcceptPlane)
			{
				ASSERT(dwRet<12);

				outEdgeA[dwRet]=vA;
				outEdgeB[dwRet]=vB;

				++dwRet;
			}
		}

		return dwRet;
	}

	void SetMAX()
	{
		m_vMax = float3(FLT_MAX,FLT_MAX,FLT_MAX);
		m_vMin = -m_vMax;
	}

	bool IsMAX() const
	{
		float3 vMax = float3(FLT_MAX,FLT_MAX,FLT_MAX);

		return m_vMax==vMax && m_vMin==-vMax;
	}
};


inline float3 TransformDir(const glm::mat4& m, const float3 &inDir)
{
	return float3(
		inDir.x * m[0][0] + inDir.y * m[0][1] + inDir.z * m[0][2],
		inDir.x * m[1][0] + inDir.y * m[1][1] + inDir.z * m[1][2],
		inDir.x * m[2][0] + inDir.y * m[2][1] + inDir.z * m[2][2]);
}



// out.xyz = Mat * in.xyz1
inline float3 TransformPos(const glm::mat4 &m, const float3& inPos)
{
	return float3(
		inPos.x * m[0][0] + inPos.y * m[0][1] + inPos.z * m[0][2] + m[0][3],
		inPos.x * m[1][0] + inPos.y * m[1][1]+ inPos.z * m[1][2] + m[1][3],
		inPos.x * m[2][0] + inPos.y * m[2][1] + inPos.z * m[2][2] + m[2][3]);
}

inline CPlane3 TransformPlane(const glm::mat4 &m, const CPlane3& inPlane)
{
	float3 vNormal = TransformDir(m, inPlane.m_vNormal);

	vNormal = glm::normalize(vNormal);

	// can be optimized
	CPlane3 ret(vNormal, TransformPos(m, inPlane.GetPlanePoint()));

	return ret;
}


// todo: optimize
inline CAABB3 CAABB3::GetTransformed( const glm::mat4 &mat ) const
{
	CAABB3 ret;

	// slow but good as a test reference
	float3 v[8];
	
	v[0] = TransformPos(mat, float3(m_vMin.x,m_vMin.y,m_vMin.z));
	v[1] = TransformPos(mat, float3(m_vMin.x,m_vMin.y,m_vMax.z));
	v[2] = TransformPos(mat, float3(m_vMin.x,m_vMax.y,m_vMin.z));
	v[3] = TransformPos(mat, float3(m_vMin.x,m_vMax.y,m_vMax.z));
	v[4] = TransformPos(mat, float3(m_vMax.x,m_vMin.y,m_vMin.z));
	v[5] = TransformPos(mat, float3(m_vMax.x,m_vMin.y,m_vMax.z));
	v[6] = TransformPos(mat, float3(m_vMax.x,m_vMax.y,m_vMin.z));
	v[7] = TransformPos(mat, float3(m_vMax.x,m_vMax.y,m_vMax.z));

	int i;

	ret.m_vMin = ret.m_vMax = v[0];

	for(i=1;i<8;++i)
	{
		ret.m_vMin = glm::min(ret.m_vMin, v[i]);
		ret.m_vMax = glm::min(ret.m_vMax, v[i]);
	}

	return ret;
}


inline bool CAABB3::intersectAABB( const CAABB3 &rAABB ) const
{
	for(uint32 dwI = 0; dwI < 3; ++dwI)
	{
		if(rAABB.m_vMin[dwI]>=m_vMax[dwI])
			return false;	// no overlap

		if(m_vMin[dwI]>=rAABB.m_vMax[dwI])
			return false;	// no overlap
	}

	return true;	// overlap
}

inline bool CAABB3::intersectSphere2( const float3 &rMidPos, const double fRadius2 ) const
{
	return EuklidDistance2(rMidPos)<fRadius2;
}


// based on Fast Ray-Box Intersection by Andrew Woo from "Graphics Gems", Academic Press, 1990
// http://www.acm.org/tog/GraphicsGems/gems/RayBox.c
// better code might be: http://www.codercorner.com/RayAABB.cpp
inline bool CAABB3::intersectRay( const float3 &vOrigin, const float3 &vDir, float3 &vHitPos ) const
{
	bool bInside = true;
	char quadrant[3];
	int i;
	int whichPlane;
	float maxT[3];
	float candidatePlane[3];

	// Find candidate planes; this loop can be avoided if	rays cast all from the eye(assume perpsective view)
	for(i=0; i<3; i++)
		if(vOrigin[i] < m_vMin[i]) 
		{
			quadrant[i] = 1;
			candidatePlane[i] = m_vMin[i];
			bInside = false;
		}
		else if(vOrigin[i] > m_vMax[i]) 
		{
			quadrant[i] = 0;
			candidatePlane[i] = m_vMax[i];
			bInside = false;
		}
		else	
			quadrant[i] = 2;

	// Ray vOrigin inside bounding box
	if(bInside)	
	{
		vHitPos = vOrigin;
		return true;
	}

	// Calculate T distances to candidate planes
	for(i = 0; i < 3; i++)
		if(quadrant[i] != 2 && vDir[i] !=0.)
			maxT[i] = (candidatePlane[i]-vOrigin[i]) / vDir[i];
		else
			maxT[i] = -1.;

	// Get largest of the maxT's for final choice of intersection
	whichPlane = 0;

	for(i = 1; i < 3; i++)
		if(maxT[whichPlane] < maxT[i])
			whichPlane = i;

	// Check final candidate actually inside box
	if(maxT[whichPlane] < 0)
		return false;

	for(i = 0; i < 3; i++)
		if(whichPlane != i) 
		{
			vHitPos[i] = vOrigin[i] + maxT[whichPlane] * vDir[i];

			if(vHitPos[i] < m_vMin[i] || vHitPos[i] > m_vMax[i])
				return false;
		} 
		else 
			vHitPos[i] = candidatePlane[i];

	return true;				// ray hits box
}	


// based on Fast Ray-Box Intersection by Andrew Woo from "Graphics Gems", Academic Press, 1990
// http://www.acm.org/tog/GraphicsGems/gems/RayBox.c
// better code might be: http://www.codercorner.com/RayAABB.cpp
inline bool CAABB3::intersectRay( const float3 &vOrigin, const float3 &vDir ) const
{
	bool bInside = true;
	char quadrant[3];
	int i;
	int whichPlane;
	float maxT[3];
	float candidatePlane[3];

	// Find candidate planes; this loop can be avoided if	rays cast all from the eye(assume perpsective view)
	for(i=0; i<3; i++)
		if(vOrigin[i] < m_vMin[i]) 
		{
			quadrant[i] = 1;
			candidatePlane[i] = m_vMin[i];
			bInside = false;
		}
		else if(vOrigin[i] > m_vMax[i]) 
		{
			quadrant[i] = 0;
			candidatePlane[i] = m_vMax[i];
			bInside = false;
		}
		else	
			quadrant[i] = 2;

	// Ray vOrigin inside bounding box
	if(bInside)	
		return true;

	// Calculate T distances to candidate planes
	for(i = 0; i < 3; i++)
		if(quadrant[i] != 2 && vDir[i] !=0.)
			maxT[i] = (candidatePlane[i]-vOrigin[i]) / vDir[i];
		else
			maxT[i] = -1.;

	// Get largest of the maxT's for final choice of intersection
	whichPlane = 0;

	for(i = 1; i < 3; i++)
		if(maxT[whichPlane] < maxT[i])
			whichPlane = i;

	// Check final candidate actually inside box
	if(maxT[whichPlane] < 0)
		return false;

	for(i = 0; i < 3; i++)
		if(whichPlane != i) 
		{
			float vHitPosAxis = vOrigin[i] + maxT[whichPlane] * vDir[i];

			if(vHitPosAxis < m_vMin[i] || vHitPosAxis > m_vMax[i])
				return false;
		} 

	return true;				// ray hits box
}	

//!
inline bool IsNAN(CAABB3 in)
{
	return IsNAN(in.m_vMin) || IsNAN(in.m_vMax);
}





#endif	// MATHLIB_AABB3_H
