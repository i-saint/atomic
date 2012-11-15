#ifndef __ist_Math_simdvec_permute_h__
#define __ist_Math_simdvec_permute_h__

namespace ist {

#if defined(ist_env_Windows)

    #define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)

    #define simd_swizzle(v, _0, _1, _2, _3) _simd_swizzle<_0, _1, _2, _3>(v)

    template<int x, int y, int z, int w> istForceInline simdvec4 _simd_swizzle(simdvec4 a)
    {
        return return _mm_shuffle_ps(a, a, SSE_SHUFFLE(x,y,z,w));
    }

    template<int x, int y, int z, int w> istForceInline simdvec4 simd_permute(simdvec4 a, simdvec4 b);
    template<> istForceInline simdvec4 simd_permute<0,1,4,5>(simdvec4 a, simdvec4 b) { return _mm_movelh_ps(a, b); }
    template<> istForceInline simdvec4 simd_permute<0,4,1,5>(simdvec4 a, simdvec4 b) { return _mm_unpacklo_ps(a, b); }
    template<> istForceInline simdvec4 simd_permute<2,3,6,7>(simdvec4 a, simdvec4 b) { return _mm_movehl_ps(b, a); }
    template<> istForceInline simdvec4 simd_permute<2,6,3,7>(simdvec4 a, simdvec4 b) { return _mm_unpackhi_ps(a, b); }
    template<> istForceInline simdvec4 simd_permute<4,0,5,1>(simdvec4 a, simdvec4 b) { return _mm_unpacklo_ps(b, a); }
    template<> istForceInline simdvec4 simd_permute<4,5,0,1>(simdvec4 a, simdvec4 b) { return _mm_movelh_ps(b, a); }
    template<> istForceInline simdvec4 simd_permute<6,7,2,3>(simdvec4 a, simdvec4 b) { return _mm_movehl_ps(a, b); }

#undef SSE_SHUFFLE

#elif defined()
#endif

} // namespace ist

#endif // __ist_Math_simdvec_permute_h__
