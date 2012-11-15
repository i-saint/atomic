#ifndef __ist_Math_simdvec_h__
#define __ist_Math_simdvec_h__

#if defined(ist_env_Windows)
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#elif defined(ist_env_Android)
#endif

namespace ist {

#if defined(ist_env_x86)
    typedef __m128  simdvec4;
    typedef __m128i simdvec4i;
    typedef __m256  simdvec8;
    typedef __m256i simdvec8i;
#elif defined(ist_env_ARM32)
    typedef float32x4   simdvec4;
    typedef int32x4     simdvec4i;
#endif

#if defined(ist_env_x86)
    istForceInline simdvec4 simdvec4_set() { return _mm_set1_ps(0.0f); }
    istForceInline simdvec4 simdvec4_set(float v) { return _mm_set1_ps(v); }
    istForceInline simdvec4 simdvec4_set(float _x, float _y, float _z, float _w) { return _mm_set_ps(_x, _y, _z, _w); }
    istForceInline simdvec4 operator+(simdvec4 a, simdvec4 b)               { return _mm_add_ps(a, b); }
    istForceInline simdvec4 operator-(simdvec4 a, simdvec4 b)               { return _mm_sub_ps(a, b); }
    istForceInline simdvec4 operator*(simdvec4 a, simdvec4 b)               { return _mm_mul_ps(a, b); }
    istForceInline simdvec4 operator/(simdvec4 a, simdvec4 b)               { return _mm_div_ps(a, b); }
    istForceInline simdvec4 operator&(simdvec4 a, simdvec4 b)               { return _mm_and_ps(a, b); }
    istForceInline simdvec4 operator|(simdvec4 a, simdvec4 b)               { return _mm_or_ps(a, b); }
    istForceInline simdvec4 operator^(simdvec4 a, simdvec4 b)               { return _mm_xor_ps(a, b); }
    istForceInline simdvec4 operator> (simdvec4 a, simdvec4 b)              { return _mm_cmpgt_ps(a, b); }
    istForceInline simdvec4 operator>=(simdvec4 a, simdvec4 b)              { return _mm_cmpge_ps(a, b); }
    istForceInline simdvec4 operator< (simdvec4 a, simdvec4 b)              { return _mm_cmplt_ps(a, b); }
    istForceInline simdvec4 operator<=(simdvec4 a, simdvec4 b)              { return _mm_cmple_ps(a, b); }
    istForceInline simdvec4 operator==(simdvec4 a, simdvec4 b)              { return _mm_cmpeq_ps(a, b); }
    istForceInline simdvec4 operator!=(simdvec4 a, simdvec4 b)              { return _mm_cmpneq_ps(a, b); }
    istForceInline simdvec4 simd_muladd(simdvec4 a, simdvec4 b, simdvec4 c) { return (a*b)+c; }
    istForceInline simdvec4 simd_mulsub(simdvec4 a, simdvec4 b, simdvec4 c) { return (a*b)-c; }
    istForceInline simdvec4 simd_andnot(simdvec4 a, simdvec4 b)             { return _mm_andnot_ps(a, b); }
    istForceInline simdvec4 simd_rcp(simdvec4 a)                            { return _mm_rcp_ps(a); }
    istForceInline simdvec4 simd_rsqrt(simdvec4 a)                          { return _mm_rsqrt_ps(a); }
    istForceInline simdvec4 simd_sqrt(simdvec4 a)                           { return _mm_sqrt_ps(a); }
    istForceInline simdvec4 simd_min(simdvec4 a, simdvec4 b)                { return _mm_min_ps(a, b); }
    istForceInline simdvec4 simd_max(simdvec4 a, simdvec4 b)                { return _mm_max_ps(a, b); }
    istForceInline simdvec4 simd_abs(simdvec4 a)                            { return (const simdvec4&)_mm_abs_epi32((const simdvec4i&)a); }

    istForceInline simdvec4 simd_select(simdvec4 a, simdvec4 b, simdvec4 c)
    {
        simdvec4 t1 = _mm_andnot_ps(c, a);
        simdvec4 t2 = _mm_and_ps(b, c);
        return _mm_or_ps(t1, t2);
    }

    istForceInline simdvec4i operator+(simdvec4i a, simdvec4i b)                { return _mm_add_epi32(a, b); }
    istForceInline simdvec4i operator-(simdvec4i a, simdvec4i b)                { return _mm_sub_epi32(a, b); }
    istForceInline simdvec4i operator*(simdvec4i a, simdvec4i b)                { return _mm_mul_epi32(a, b); }
    istForceInline simdvec4i simd_muladd(simdvec4i a, simdvec4i b, simdvec4i c) { return (a*b)+c; }
    istForceInline simdvec4i simd_mulsub(simdvec4i a, simdvec4i b, simdvec4i c) { return (a*b)-c; }
    istForceInline simdvec4i simd_min(simdvec4i a, simdvec4i b)                 { return _mm_min_epi32(a, b); }
    istForceInline simdvec4i simd_max(simdvec4i a, simdvec4i b)                 { return _mm_max_epi32(a, b); }
    istForceInline simdvec4i simd_abs(simdvec4i a)                              { return _mm_abs_epi32(a); }

#elif defined(ist_env_ARM32)
    istForceInline simdvec4 operator+(simdvec4 a, simdvec4 b)               { return vaddq_f32(a, b); }
    istForceInline simdvec4 operator-(simdvec4 a, simdvec4 b)               { return vsubq_f32(a, b); }
    istForceInline simdvec4 operator*(simdvec4 a, simdvec4 b)               { return vmulq_f32(a, b); }
    istForceInline simdvec4 operator/(simdvec4 a, simdvec4 b)               { return vmulq_f32(a, vrecpsq_f32(b)); }
    istForceInline simdvec4 simd_muladd(simdvec4 a, simdvec4 b, simdvec4 c) { return vmlaq_f32(a,b,c); }
    istForceInline simdvec4 simd_mulsub(simdvec4 a, simdvec4 b, simdvec4 c) { return vmlsq_f32(a,b,c); }
    istForceInline simdvec4 simd_rcp(simdvec4 a)                            { return vrecpsq_f32(a); }
    istForceInline simdvec4 simd_rsqrt(simdvec4 a)                          { return vrsqrtsq_f32(a); }
    istForceInline simdvec4 simd_sqrt(simdvec4 a)                           { return vmulq_f32(a, vrsqrtsq_f32(a)); }
    istForceInline simdvec4 simd_min(simdvec4 a)                            { return vmin_f32(a); }
    istForceInline simdvec4 simd_max(simdvec4 a)                            { return vmax_f32(a); }
    istForceInline simdvec4 simd_abs(simdvec4 a)                            { return (simdvec4)vabsq_s32((simdvec4i)a); }

    istForceInline simdvec4i operator+(simdvec4i a, simdvec4i b)                { return vaddq_s32(a, b); }
    istForceInline simdvec4i operator-(simdvec4i a, simdvec4i b)                { return vsubq_s32(a, b); }
    istForceInline simdvec4i operator*(simdvec4i a, simdvec4i b)                { return vmulq_s32(a, b); }
    istForceInline simdvec4i simd_muladd(simdvec4i a, simdvec4i b, simdvec4i c) { return vmlaq_s32(a,b,c); }
    istForceInline simdvec4i simd_mulsub(simdvec4i a, simdvec4i b, simdvec4i c) { return vmlsq_s32(a,b,c); }
    istForceInline simdvec4i simd_min(simdvec4i a)                              { return vmin_s32(a); }
    istForceInline simdvec4i simd_max(simdvec4i a)                              { return vmax_s32(a); }
    istForceInline simdvec4i simd_abs(simdvec4i a)                              { return vabsq_s32(a); }

#endif

} // namespace ist

#include "simdvec_permute.h"

#endif // __ist_Math_simdvec_h__
