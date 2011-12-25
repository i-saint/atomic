#ifndef __atomic_Util__
#define __atomic_Util__

struct sphRigidSphere;
struct sphRigidBox;

namespace atomic {

    vec4 GenRotateAxis();
    void CreateRigidSphere(sphRigidSphere &o, EntityHandle h, const vec4& pos, float32 r);
    void CreateRigidBox(sphRigidBox &o, EntityHandle h, const mat4& t, const vec4 &size);

    // glm ÇÃ simd_mat4x4 * simd_vec4 Ç™é¿ëïÇ≥ÇÍÇƒÇ»Ç¢Ç¡Ç€Ç¢ÇÃÇ≈Åc
    __forceinline simdvec4 simd_mul(const simdmat4 &M, const simdvec4 &V)
    {
        // Splat x,y,z and w
        simdvec4 vTempX = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(0,0,0,0));
        simdvec4 vTempY = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(1,1,1,1));
        simdvec4 vTempZ = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(2,2,2,2));
        simdvec4 vTempW = _mm_shuffle_ps(V.Data, V.Data, _MM_SHUFFLE(3,3,3,3));
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

} // namespace atomic
#endif // __atomic_Util__
