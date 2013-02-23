#ifndef ist_Math_FrustumCulling_h
#define ist_Math_FrustumCulling_h

#include "ist/Base.h"

namespace ist {

typedef __m128 simdvec4_t;

struct FrustumPlanes
{
private:
    simdvec4_t m_data[6];

public:

    // view projection matrix から frustum を構成する 6 つの plane を算出します
    // MatrixType は XMMATRIX か glm::mat4x4 を想定
    template<class MatrixType>
    void constructFromViewProjectionMatrix(const MatrixType &mat, bool normalize=true);

    simdvec4_t& operator[](int i) { return m_data[i]; }
    const simdvec4_t& operator[](int i) const { return m_data[i]; }
};

struct AxisAlignedBoundingBox
{
private:
    simdvec4_t m_data[2];

public:
    // 全ての simdvec4_t の引数の w 要素は 1.0f である必要があります

    AxisAlignedBoundingBox() { m_data[0]=m_data[1]=_mm_set1_ps(0.0f); }
    AxisAlignedBoundingBox(const simdvec4_t v) { m_data[0]=m_data[1]=v; }
    AxisAlignedBoundingBox(const simdvec4_t v_min, const simdvec4_t v_max) { m_data[0]=v_min; m_data[1]=v_max; }

    void addPoint(const simdvec4_t v);

    simdvec4_t& operator[](int i) { return m_data[i]; }
    const simdvec4_t& operator[](int i) const { return m_data[i]; }
};
typedef AxisAlignedBoundingBox AABB;

// true なら視界内、false なら視界外
bool TestFrustumAABB(const FrustumPlanes &frustum, const AABB &aabb);


} // namespace ist

istSerializeRaw(ist::FrustumPlanes);
istSerializeRaw(ist::AxisAlignedBoundingBox);

#include "FrustumCulling.inl"

#endif // ist_Math_FrustumCulling_h
