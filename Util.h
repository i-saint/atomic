#ifndef __atomic_Util__
#define __atomic_Util__


namespace atomic {

    struct CollisionSphere;
    struct CollisionBox;

    vec4 GenRandomVector2();
    vec4 GenRandomVector3();
    vec4 GenRandomUnitVector2();
    vec4 GenRandomUnitVector3();
    void UpdateCollisionSphere(CollisionSphere &o, const vec4& pos, float32 r);
    void UpdateCollisionBox(CollisionBox &o, const mat4& t, const vec4 &size);

    vec4 GetNearestPlayerPosition(const vec4 &pos);
    void ShootSimpleBullet(EntityHandle owner, const vec4 &pos, const vec4 &vel);

    void CreateDateString(char *buf, uint32 len);

} // namespace atomic

namespace glm {
namespace detail {

// glm ÇÃíÜÇ…é¿ëÃÇ™Ç»Ç¢ÇÊÇ§Ç»ÇÃÇ≈Åc
__forceinline fvec4SIMD operator* (fmat4x4SIMD const & M, fvec4SIMD const & V)
{
    // Splat x,y,z and w
    fvec4SIMD vTempX = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(0,0,0,0));
    fvec4SIMD vTempY = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(1,1,1,1));
    fvec4SIMD vTempZ = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(2,2,2,2));
    fvec4SIMD vTempW = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(3,3,3,3));
    // Mul by the matrix
    vTempX = vTempX * M.Data[0];
    vTempY = vTempY * M.Data[1];
    vTempZ = vTempZ * M.Data[2];
    vTempW = vTempW * M.Data[3];
    // Add them all together
    vTempX = vTempX + vTempY;
    vTempZ = vTempZ + vTempW;
    vTempX = vTempX + vTempZ;
    return vTempX;
}

} // namespace detail
} // namespace glm

#endif // __atomic_Util__
