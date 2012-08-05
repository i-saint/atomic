#ifndef __ist_Math_simdvec_permute_h__
#define __ist_Math_simdvec_permute_h__

namespace ist {

#if defined(istWindows)

#define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)

    template<int x, int y, int z, int w> istForceInline simdvec4 simd_permute(simdvec4 a);
    template<> istForceInline simdvec4 simd_permute<0,0,0,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,0,0)); }
    template<> istForceInline simdvec4 simd_permute<0,0,0,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,0,1)); }
    template<> istForceInline simdvec4 simd_permute<0,0,0,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,0,2)); }
    template<> istForceInline simdvec4 simd_permute<0,0,0,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,0,3)); }
    template<> istForceInline simdvec4 simd_permute<0,0,1,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,1,0)); }
    template<> istForceInline simdvec4 simd_permute<0,0,1,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,1,1)); }
    template<> istForceInline simdvec4 simd_permute<0,0,1,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,1,2)); }
    template<> istForceInline simdvec4 simd_permute<0,0,1,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,1,3)); }
    template<> istForceInline simdvec4 simd_permute<0,0,2,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,2,0)); }
    template<> istForceInline simdvec4 simd_permute<0,0,2,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,2,1)); }
    template<> istForceInline simdvec4 simd_permute<0,0,2,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,2,2)); }
    template<> istForceInline simdvec4 simd_permute<0,0,2,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,2,3)); }
    template<> istForceInline simdvec4 simd_permute<0,0,3,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,3,0)); }
    template<> istForceInline simdvec4 simd_permute<0,0,3,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,3,1)); }
    template<> istForceInline simdvec4 simd_permute<0,0,3,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,3,2)); }
    template<> istForceInline simdvec4 simd_permute<0,0,3,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,0,3,3)); }
    template<> istForceInline simdvec4 simd_permute<0,1,0,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,0,0)); }
    template<> istForceInline simdvec4 simd_permute<0,1,0,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,0,1)); }
    template<> istForceInline simdvec4 simd_permute<0,1,0,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,0,2)); }
    template<> istForceInline simdvec4 simd_permute<0,1,0,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,0,3)); }
    template<> istForceInline simdvec4 simd_permute<0,1,1,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,1,0)); }
    template<> istForceInline simdvec4 simd_permute<0,1,1,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,1,1)); }
    template<> istForceInline simdvec4 simd_permute<0,1,1,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,1,2)); }
    template<> istForceInline simdvec4 simd_permute<0,1,1,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,1,3)); }
    template<> istForceInline simdvec4 simd_permute<0,1,2,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,2,0)); }
    template<> istForceInline simdvec4 simd_permute<0,1,2,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,2,1)); }
    template<> istForceInline simdvec4 simd_permute<0,1,2,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,2,2)); }
    template<> istForceInline simdvec4 simd_permute<0,1,2,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,2,3)); }
    template<> istForceInline simdvec4 simd_permute<0,1,3,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,3,0)); }
    template<> istForceInline simdvec4 simd_permute<0,1,3,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,3,1)); }
    template<> istForceInline simdvec4 simd_permute<0,1,3,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,3,2)); }
    template<> istForceInline simdvec4 simd_permute<0,1,3,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,1,3,3)); }
    template<> istForceInline simdvec4 simd_permute<0,2,0,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,0,0)); }
    template<> istForceInline simdvec4 simd_permute<0,2,0,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,0,1)); }
    template<> istForceInline simdvec4 simd_permute<0,2,0,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,0,2)); }
    template<> istForceInline simdvec4 simd_permute<0,2,0,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,0,3)); }
    template<> istForceInline simdvec4 simd_permute<0,2,1,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,1,0)); }
    template<> istForceInline simdvec4 simd_permute<0,2,1,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,1,1)); }
    template<> istForceInline simdvec4 simd_permute<0,2,1,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,1,2)); }
    template<> istForceInline simdvec4 simd_permute<0,2,1,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,1,3)); }
    template<> istForceInline simdvec4 simd_permute<0,2,2,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,2,0)); }
    template<> istForceInline simdvec4 simd_permute<0,2,2,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,2,1)); }
    template<> istForceInline simdvec4 simd_permute<0,2,2,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,2,2)); }
    template<> istForceInline simdvec4 simd_permute<0,2,2,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,2,3)); }
    template<> istForceInline simdvec4 simd_permute<0,2,3,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,3,0)); }
    template<> istForceInline simdvec4 simd_permute<0,2,3,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,3,1)); }
    template<> istForceInline simdvec4 simd_permute<0,2,3,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,3,2)); }
    template<> istForceInline simdvec4 simd_permute<0,2,3,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,2,3,3)); }
    template<> istForceInline simdvec4 simd_permute<0,3,0,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,0,0)); }
    template<> istForceInline simdvec4 simd_permute<0,3,0,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,0,1)); }
    template<> istForceInline simdvec4 simd_permute<0,3,0,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,0,2)); }
    template<> istForceInline simdvec4 simd_permute<0,3,0,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,0,3)); }
    template<> istForceInline simdvec4 simd_permute<0,3,1,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,1,0)); }
    template<> istForceInline simdvec4 simd_permute<0,3,1,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,1,1)); }
    template<> istForceInline simdvec4 simd_permute<0,3,1,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,1,2)); }
    template<> istForceInline simdvec4 simd_permute<0,3,1,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,1,3)); }
    template<> istForceInline simdvec4 simd_permute<0,3,2,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,2,0)); }
    template<> istForceInline simdvec4 simd_permute<0,3,2,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,2,1)); }
    template<> istForceInline simdvec4 simd_permute<0,3,2,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,2,2)); }
    template<> istForceInline simdvec4 simd_permute<0,3,2,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,2,3)); }
    template<> istForceInline simdvec4 simd_permute<0,3,3,0>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,3,0)); }
    template<> istForceInline simdvec4 simd_permute<0,3,3,1>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,3,1)); }
    template<> istForceInline simdvec4 simd_permute<0,3,3,2>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,3,2)); }
    template<> istForceInline simdvec4 simd_permute<0,3,3,3>(simdvec4 a) { return _mm_shuffle_ps(a, a, SSE_SHUFFLE(0,3,3,3)); }

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
