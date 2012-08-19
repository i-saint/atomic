#include <windows.h>
#include "ist/Base.h"

#define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)

//#define __ist_enable_soavec8__



namespace ist {

template<class simdvec> struct simdvec_traits;

template<>
struct simdvec_traits<simdvec4>
{
    enum{ simd_elements = 4 };
};
#ifdef __ist_enable_soavec8__
template<>
struct soavec_traits<simdvec8>
{
    enum{ simd_elements = 8 };
};
#endif // __ist_enable_soavec8__


template<class soavec>
struct soavec_traits
{
    enum{ soa_elements  = soavec::soa_elements };
    enum{ simd_elements = soavec::simd_elements };
};



template<class simdvec>
struct tsoavec2x
{
    typedef simdvec simdvec_t;
    enum{
        soa_elements  = 2,
        simd_elements = simdvec_traits<simdvec_t>::simd_elements,
    };

    simdvec sv[2];

    istForceInline tsoavec2x() {}
    istForceInline tsoavec2x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; }
    istForceInline tsoavec2x(const simdvec &_x, const simdvec &_y) { sv[0]=_x; sv[1]=_y; }

    istForceInline simdvec x() const { return sv[0]; }
    istForceInline simdvec y() const { return sv[1]; }
    istForceInline simdvec v(size_t i) const { return sv[i]; }

    istForceInline void x(const simdvec &_v) { sv[0]=_v; }
    istForceInline void y(const simdvec &_v) { sv[1]=_v; }
    istForceInline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

template<class simdvec>
struct tsoavec3x
{
    typedef simdvec simdvec_t;
    enum{
        soa_elements  = 3,
        simd_elements = simdvec_traits<simdvec_t>::simd_elements,
    };

    simdvec sv[3];

    istForceInline tsoavec3x() {}
    istForceInline tsoavec3x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; sv[2]=_sv[2]; }
    istForceInline tsoavec3x(const simdvec &_x, const simdvec &_y, const simdvec &_z) { sv[0]=_x; sv[1]=_y; sv[2]=_z; }

    istForceInline simdvec x() const { return sv[0]; }
    istForceInline simdvec y() const { return sv[1]; }
    istForceInline simdvec z() const { return sv[2]; }
    istForceInline simdvec v(size_t i) const { return sv[i]; }

    istForceInline void x(const simdvec &_v) { sv[0]=_v; }
    istForceInline void y(const simdvec &_v) { sv[1]=_v; }
    istForceInline void z(const simdvec &_v) { sv[2]=_v; }
    istForceInline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

template<class simdvec>
struct tsoavec4x
{
    typedef simdvec simdvec_t;
    typedef tsoavec3x<simdvec_t> soavec3x;
    enum{
        soa_elements  = 4,
        simd_elements = simdvec_traits<simdvec_t>::simd_elements,
    };

    simdvec sv[4];

    istForceInline tsoavec4x() {}
    istForceInline tsoavec4x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; sv[2]=_sv[2]; sv[3]=_sv[3]; }
    istForceInline tsoavec4x(const simdvec &_x, const simdvec &_y, const simdvec &_z, const simdvec &_w) { sv[0]=_x; sv[1]=_y; sv[2]=_z; sv[3]=_w; }
    istForceInline tsoavec4x(const soavec3x &_xyz, const simdvec &_w) { sv[0]=_xyz.sv[0]; sv[1]=_xyz.sv[1]; sv[2]=_xyz.sv[2]; sv[3]=_w; }

    istForceInline simdvec x() const { return sv[0]; }
    istForceInline simdvec y() const { return sv[1]; }
    istForceInline simdvec z() const { return sv[2]; }
    istForceInline simdvec w() const { return sv[3]; }
    istForceInline soavec3x xyz() const { return soavec3x(sv[0], sv[1], sv[2]); }
    istForceInline simdvec v(size_t i) const { return sv[i]; }

    istForceInline void x(const simdvec &_v) { sv[0]=_v; }
    istForceInline void y(const simdvec &_v) { sv[1]=_v; }
    istForceInline void z(const simdvec &_v) { sv[2]=_v; }
    istForceInline void w(const simdvec &_v) { sv[3]=_v; }
    istForceInline void xyz(const soavec3x &_v) { sv[0]=_v.sv[0]; sv[1]=_v.sv[1]; sv[2]=_v.sv[2]; }
    istForceInline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

typedef __declspec(align(16)) tsoavec2x<simdvec4> soavec24;
typedef __declspec(align(16)) tsoavec3x<simdvec4> soavec34;
typedef __declspec(align(16)) tsoavec4x<simdvec4> soavec44;
#ifdef __ist_enable_soavec8__
typedef __declspec(align(32)) tsoavec2x<simdvec8> soavec28;
typedef __declspec(align(32)) tsoavec3x<simdvec8> soavec38;
typedef __declspec(align(32)) tsoavec4x<simdvec8> soavec48;
#endif // __ist_enable_soavec8__


namespace detail {

    template<size_t S> struct soa_types;

    template<>
    struct soa_types<4>
    {
        typedef simdvec4 simdvec;
        typedef soavec24 soavec2x;
        typedef soavec34 soavec3x;
        typedef soavec44 soavec4x;
    };
#ifdef __ist_enable_soavec8__
    template<>
    struct soa_types<8>
    {
        typedef simdvec8 simdvec;
        typedef soavec28 soavec2x;
        typedef soavec38 soavec3x;
        typedef soavec48 soavec4x;
    };
#endif // __ist_enable_soavec8__

} // namespace detail

#define DECLARE_SOATYPES(T)                                         \
    enum { simd_elements = T::simd_elements, };                     \
    typedef typename soa_types<simd_elements>::simdvec simdvec;     \
    typedef typename soa_types<simd_elements>::soavec2x soavec2x;   \
    typedef typename soa_types<simd_elements>::soavec3x soavec3x;   \
    typedef typename soa_types<simd_elements>::soavec4x soavec4x;





///////////////////////////////////////////////////////////////
//  AoS <-> SoA
///////////////////////////////////////////////////////////////

#if defined(istx84)
// 2 要素 SoA ベクトルへ
istForceInline soavec24 soa_transpose24(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}


// 3 要素 SoA ベクトルへ
istForceInline soavec34 soa_transpose34(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, zero);
    __m128 r4 = zero;
    return soavec34(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

istForceInline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = zero;
    return soavec34(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

istForceInline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, zero);
    return soavec34(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

istForceInline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, v3.Data);
    return soavec34(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}


// 4 要素 SoA ベクトルへ
istForceInline soavec44 soa_transpose44(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, zero);
    __m128 r4 = zero;
    return soavec44(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = zero;
    return soavec44(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, zero);
    return soavec44(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

istForceInline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, v3.Data);
    return soavec44(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}
#elif defined(__ist_env_ARM__)
#endif

#ifdef __ist_enable_soavec8__
// todo
#endif // __ist_enable_soavec8__


///////////////////////////////////////////////////////////////
//  SoA Add
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_add
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const U &b)
    {
        return soavec2x(
            a.sv[0] + b.sv[0],
            a.sv[1] + b.sv[1] );
    }

    istForceInline static soavec3x op3(const T &a, const U &b)
    {
        return soavec3x(
            a.sv[0] + b.sv[0],
            a.sv[1] + b.sv[1],
            a.sv[2] + b.sv[2]);
    }

    istForceInline static soavec4x op4(const T &a, const U &b)
    {
        return soavec4x(
            a.sv[0] + b.sv[0],
            a.sv[1] + b.sv[1],
            a.sv[2] + b.sv[2],
            a.sv[3] + b.sv[3]);
    }
};

template<class T>
struct soa_add<T, typename T::simdvec_t>
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const simdvec &b)
    {
        return soavec2x(
            a.sv[0] + b,
            a.sv[1] + b);
    }

    istForceInline static soavec3x op3(const T &a, const simdvec &b)
    {
        return soavec3x(
            a.sv[0] + b,
            a.sv[1] + b,
            a.sv[2] + b);
    }

    istForceInline static soavec4x op4(const T &a, const simdvec &b)
    {
        return soavec4x(
            a.sv[0] + b,
            a.sv[1] + b,
            a.sv[2] + b,
            a.sv[3] + b);
    }
};

} // namespace detail

template<class T, class U> istForceInline soavec24 soa_add24(const T &a, const U &b) { return detail::soa_add<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec34 soa_add34(const T &a, const U &b) { return detail::soa_add<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec44 soa_add44(const T &a, const U &b) { return detail::soa_add<T, U>::op4(a, b); }
istForceInline soavec24 operator+(const soavec24 &a, const soavec24 &b) { return soa_add24(a, b); }
istForceInline soavec34 operator+(const soavec34 &a, const soavec34 &b) { return soa_add34(a, b); }
istForceInline soavec44 operator+(const soavec44 &a, const soavec44 &b) { return soa_add44(a, b); }
istForceInline soavec24 operator+(const soavec24 &a, const simdvec4 &b) { return soa_add24(a, b); }
istForceInline soavec34 operator+(const soavec34 &a, const simdvec4 &b) { return soa_add34(a, b); }
istForceInline soavec44 operator+(const soavec44 &a, const simdvec4 &b) { return soa_add44(a, b); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline soavec28 soa_add28(const T &a, const U &b) { return detail::soa_add<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec38 soa_add38(const T &a, const U &b) { return detail::soa_add<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec48 soa_add48(const T &a, const U &b) { return detail::soa_add<T, U>::op4(a, b); }
istForceInline soavec28 operator+(const soavec28 &a, const soavec28 &b) { return soa_add28(a, b); }
istForceInline soavec38 operator+(const soavec38 &a, const soavec38 &b) { return soa_add38(a, b); }
istForceInline soavec48 operator+(const soavec48 &a, const soavec48 &b) { return soa_add48(a, b); }
istForceInline soavec28 operator+(const soavec28 &a, const simdvec8 &b) { return soa_add28(a, b); }
istForceInline soavec38 operator+(const soavec38 &a, const simdvec8 &b) { return soa_add38(a, b); }
istForceInline soavec48 operator+(const soavec48 &a, const simdvec8 &b) { return soa_add48(a, b); }
#endif // __ist_enable_soavec8__


///////////////////////////////////////////////////////////////
//  SoA Subtract
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_sub
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const U &b)
    {
        return soavec2x(
            a.sv[0] - b.sv[0],
            a.sv[1] - b.sv[1] );
    }

    istForceInline static soavec3x op3(const T &a, const U &b)
    {
        return soavec3x(
            a.sv[0] - b.sv[0],
            a.sv[1] - b.sv[1],
            a.sv[2] - b.sv[2]);
    }

    istForceInline static soavec4x op4(const T &a, const U &b)
    {
        return soavec4x(
            a.sv[0] - b.sv[0],
            a.sv[1] - b.sv[1],
            a.sv[2] - b.sv[2],
            a.sv[3] - b.sv[3]);
    }
};

template<class T>
struct soa_sub<T, typename T::simdvec_t>
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const simdvec &b)
    {
        return soavec2x(
            a.sv[0] - b,
            a.sv[1] - b);
    }

    istForceInline static soavec3x op3(const T &a, const simdvec &b)
    {
        return soavec3x(
            a.sv[0] - b,
            a.sv[1] - b,
            a.sv[2] - b);
    }

    istForceInline static soavec4x op4(const T &a, const simdvec &b)
    {
        return soavec4x(
            a.sv[0] - b,
            a.sv[1] - b,
            a.sv[2] - b,
            a.sv[3] - b);
    }
};

} // namespace detail

template<class T, class U> istForceInline soavec24 soa_sub24(const T &a, const U &b) { return detail::soa_sub<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec34 soa_sub34(const T &a, const U &b) { return detail::soa_sub<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec44 soa_sub44(const T &a, const U &b) { return detail::soa_sub<T, U>::op4(a, b); }
istForceInline soavec24 operator-(const soavec24 &a, const soavec24 &b) { return soa_sub24(a, b); }
istForceInline soavec34 operator-(const soavec34 &a, const soavec34 &b) { return soa_sub34(a, b); }
istForceInline soavec44 operator-(const soavec44 &a, const soavec44 &b) { return soa_sub44(a, b); }
istForceInline soavec24 operator-(const soavec24 &a, const simdvec4 &b) { return soa_sub24(a, b); }
istForceInline soavec34 operator-(const soavec34 &a, const simdvec4 &b) { return soa_sub34(a, b); }
istForceInline soavec44 operator-(const soavec44 &a, const simdvec4 &b) { return soa_sub44(a, b); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline soavec28 soa_sub28(const T &a, const U &b) { return detail::soa_sub<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec38 soa_sub38(const T &a, const U &b) { return detail::soa_sub<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec48 soa_sub48(const T &a, const U &b) { return detail::soa_sub<T, U>::op4(a, b); }
istForceInline soavec28 operator-(const soavec28 &a, const soavec28 &b) { return soa_sub28(a, b); }
istForceInline soavec38 operator-(const soavec38 &a, const soavec38 &b) { return soa_sub38(a, b); }
istForceInline soavec48 operator-(const soavec48 &a, const soavec48 &b) { return soa_sub48(a, b); }
istForceInline soavec28 operator-(const soavec28 &a, const simdvec8 &b) { return soa_sub28(a, b); }
istForceInline soavec38 operator-(const soavec38 &a, const simdvec8 &b) { return soa_sub38(a, b); }
istForceInline soavec48 operator-(const soavec48 &a, const simdvec8 &b) { return soa_sub48(a, b); }
#endif // __ist_enable_soavec8__



///////////////////////////////////////////////////////////////
//  SoA Multiply
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_mul
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const U &b)
    {
        return soavec2x(
            a.sv[0] * b.sv[0],
            a.sv[1] * b.sv[1] );
    }

    istForceInline static soavec3x op3(const T &a, const U &b)
    {
        return soavec3x(
            a.sv[0] * b.sv[0],
            a.sv[1] * b.sv[1],
            a.sv[2] * b.sv[2]);
    }

    istForceInline static soavec4x op4(const T &a, const U &b)
    {
        return soavec4x(
            a.sv[0] * b.sv[0],
            a.sv[1] * b.sv[1],
            a.sv[2] * b.sv[2],
            a.sv[3] * b.sv[3]);
    }
};

template<class T>
struct soa_mul<T, typename T::simdvec_t>
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const simdvec &b)
    {
        return soavec2x(
            a.sv[0] * b,
            a.sv[1] * b);
    }

    istForceInline static soavec3x op3(const T &a, const simdvec &b)
    {
        return soavec3x(
            a.sv[0] * b,
            a.sv[1] * b,
            a.sv[2] * b);
    }

    istForceInline static soavec4x op4(const T &a, const simdvec &b)
    {
        return soavec4x(
            a.sv[0] * b,
            a.sv[1] * b,
            a.sv[2] * b,
            a.sv[3] * b);
    }
};

} // namespace detail

template<class T, class U> istForceInline soavec24 soa_mul24(const T &a, const U &b) { return detail::soa_mul<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec34 soa_mul34(const T &a, const U &b) { return detail::soa_mul<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec44 soa_mul44(const T &a, const U &b) { return detail::soa_mul<T, U>::op4(a, b); }
istForceInline soavec24 operator*(const soavec24 &a, const soavec24 &b) { return soa_mul24(a, b); }
istForceInline soavec34 operator*(const soavec34 &a, const soavec34 &b) { return soa_mul34(a, b); }
istForceInline soavec44 operator*(const soavec44 &a, const soavec44 &b) { return soa_mul44(a, b); }
istForceInline soavec24 operator*(const soavec24 &a, const simdvec4 &b) { return soa_mul24(a, b); }
istForceInline soavec34 operator*(const soavec34 &a, const simdvec4 &b) { return soa_mul34(a, b); }
istForceInline soavec44 operator*(const soavec44 &a, const simdvec4 &b) { return soa_mul44(a, b); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline soavec28 soa_mul28(const T &a, const U &b) { return detail::soa_mul<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec38 soa_mul38(const T &a, const U &b) { return detail::soa_mul<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec48 soa_mul48(const T &a, const U &b) { return detail::soa_mul<T, U>::op4(a, b); }
istForceInline soavec28 operator*(const soavec28 &a, const soavec28 &b) { return soa_mul28(a, b); }
istForceInline soavec38 operator*(const soavec38 &a, const soavec38 &b) { return soa_mul38(a, b); }
istForceInline soavec48 operator*(const soavec48 &a, const soavec48 &b) { return soa_mul48(a, b); }
istForceInline soavec28 operator*(const soavec28 &a, const simdvec8 &b) { return soa_mul28(a, b); }
istForceInline soavec38 operator*(const soavec38 &a, const simdvec8 &b) { return soa_mul38(a, b); }
istForceInline soavec48 operator*(const soavec48 &a, const simdvec8 &b) { return soa_mul48(a, b); }
#endif // __ist_enable_soavec8__



///////////////////////////////////////////////////////////////
//  SoA Divide
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_div
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const U &b)
    {
        return soavec2x(
            a.sv[0] / b.sv[0],
            a.sv[1] / b.sv[1] );
    }

    istForceInline static soavec3x op3(const T &a, const U &b)
    {
        return soavec3x(
            a.sv[0] / b.sv[0],
            a.sv[1] / b.sv[1],
            a.sv[2] / b.sv[2]);
    }

    istForceInline static soavec4x op4(const T &a, const U &b)
    {
        return soavec4x(
            a.sv[0] / b.sv[0],
            a.sv[1] / b.sv[1],
            a.sv[2] / b.sv[2],
            a.sv[3] / b.sv[3]);
    }
};

template<class T>
struct soa_div<T, typename T::simdvec_t>
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const simdvec &b)
    {
        return soavec2x(
            a.sv[0] / b,
            a.sv[1] / b);
    }

    istForceInline static soavec3x op3(const T &a, const simdvec &b)
    {
        return soavec3x(
            a.sv[0] / b,
            a.sv[1] / b,
            a.sv[2] / b);
    }

    istForceInline static soavec4x op4(const T &a, const simdvec &b)
    {
        return soavec4x(
            a.sv[0] / b,
            a.sv[1] / b,
            a.sv[2] / b,
            a.sv[3] / b);
    }
};

} // namespace detail

template<class T, class U> istForceInline soavec24 soa_div24(const T &a, const U &b) { return detail::soa_div<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec34 soa_div34(const T &a, const U &b) { return detail::soa_div<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec44 soa_div44(const T &a, const U &b) { return detail::soa_div<T, U>::op4(a, b); }
istForceInline soavec24 operator/(const soavec24 &a, const soavec24 &b) { return soa_div24(a, b); }
istForceInline soavec34 operator/(const soavec34 &a, const soavec34 &b) { return soa_div34(a, b); }
istForceInline soavec44 operator/(const soavec44 &a, const soavec44 &b) { return soa_div44(a, b); }
istForceInline soavec24 operator/(const soavec24 &a, const simdvec4 &b) { return soa_div24(a, b); }
istForceInline soavec34 operator/(const soavec34 &a, const simdvec4 &b) { return soa_div34(a, b); }
istForceInline soavec44 operator/(const soavec44 &a, const simdvec4 &b) { return soa_div44(a, b); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline soavec28 soa_div28(const T &a, const U &b) { return detail::soa_div<T, U>::op2(a, b); }
template<class T, class U> istForceInline soavec38 soa_div38(const T &a, const U &b) { return detail::soa_div<T, U>::op3(a, b); }
template<class T, class U> istForceInline soavec48 soa_div48(const T &a, const U &b) { return detail::soa_div<T, U>::op4(a, b); }
istForceInline soavec28 operator/(const soavec28 &a, const soavec28 &b) { return soa_div28(a, b); }
istForceInline soavec38 operator/(const soavec38 &a, const soavec38 &b) { return soa_div38(a, b); }
istForceInline soavec48 operator/(const soavec48 &a, const soavec48 &b) { return soa_div48(a, b); }
istForceInline soavec28 operator/(const soavec28 &a, const simdvec8 &b) { return soa_div28(a, b); }
istForceInline soavec38 operator/(const soavec38 &a, const simdvec8 &b) { return soa_div38(a, b); }
istForceInline soavec48 operator/(const soavec48 &a, const simdvec8 &b) { return soa_div48(a, b); }
#endif // __ist_enable_soavec8__



///////////////////////////////////////////////////////////////
//  SoA Select
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_mix
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const T &b, const U &s)
    {
        return soavec2x(
            glm::mix(a.x, b.x, s.x),
            glm::mix(a.y, b.y, s.y));
    }

    istForceInline static soavec3x op3(const T &a, const T &b, const U &s)
    {
        return soavec3x(
            glm::mix(a.x, b.x, s.x),
            glm::mix(a.y, b.y, s.y),
            glm::mix(a.z, b.z, s.z));
    }

    istForceInline static soavec4x op4(const T &a, const T &b, const U &s)
    {
        return soavec4x(
            glm::mix(a.x, b.x, s.x),
            glm::mix(a.y, b.y, s.y),
            glm::mix(a.z, b.z, s.z),
            glm::mix(a.w, b.w, s.w));
    }
};

template<class T>
struct soa_mix<T, typename T::simdvec_t>
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const T &b, const simdvec &s)
    {
        return soavec2x(
            glm::mix(a.x, b.x, s),
            glm::mix(a.y, b.y, s));
    }

    istForceInline static soavec3x op3(const T &a, const T &b, const simdvec &s)
    {
        return soavec3x(
            glm::mix(a.x, b.x, s),
            glm::mix(a.y, b.y, s),
            glm::mix(a.z, b.z, s));
    }

    istForceInline static soavec4x op4(const T &a, const T &b, const simdvec &s)
    {
        return soavec4x(
            glm::mix(a.x, b.x, s),
            glm::mix(a.y, b.y, s),
            glm::mix(a.z, b.z, s),
            glm::mix(a.w, b.w, s));
    }
};

} // namespace detail

template<class T, class U> istForceInline soavec24 soa_mix24(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op2(a, b, s); }
template<class T, class U> istForceInline soavec34 soa_mix34(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op3(a, b, s); }
template<class T, class U> istForceInline soavec44 soa_mix44(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op4(a, b, s); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline soavec28 soa_mix28(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op2(a, b, s); }
template<class T, class U> istForceInline soavec38 soa_mix38(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op3(a, b, s); }
template<class T, class U> istForceInline soavec48 soa_mix48(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U>::op4(a, b, s); }
#endif // __ist_enable_soavec8__



///////////////////////////////////////////////////////////////
//  SoA Dot
///////////////////////////////////////////////////////////////

namespace detail {

template<class T, class U>
struct soa_dot
{
    DECLARE_SOATYPES(T)

    istForceInline static soavec2x op2(const T &a, const U &b)
    {
        soavec2x sq = detail::soa_mul<T, U, S>::op2(a, b);
        return sq.sv[0] + sq.sv[1];
    }

    istForceInline static soavec3x op3(const T &a, const U &b)
    {
        soavec3x sq = detail::soa_mul<T, U, S>::op3(a, b);
        return sq.sv[0] + sq.sv[1] + sq.sv[2];
    }

    istForceInline static soavec4x op4(const T &a, const U &b)
    {
        soavec4x sq = detail::soa_mul<T, U, S>::op4(a, b);
        return sq.sv[0] + sq.sv[1] + sq.sv[2] + sq.sv[3];
    }
};

} // namespace detail

template<class T, class U> istForceInline simdvec4 soa_dot24(const T &a, const U &b) { return detail::soa_dot<T, U>::op2(a, b); }
template<class T, class U> istForceInline simdvec4 soa_dot44(const T &a, const U &b) { return detail::soa_dot<T, U>::op4(a, b); }
template<class T, class U> istForceInline simdvec4 soa_dot34(const T &a, const U &b) { return detail::soa_dot<T, U>::op3(a, b); }
#ifdef __ist_enable_soavec8__
template<class T, class U> istForceInline simdvec8 soa_dot28(const T &a, const U &b) { return detail::soa_dot<T, U>::op2(a, b); }
template<class T, class U> istForceInline simdvec8 soa_dot48(const T &a, const U &b) { return detail::soa_dot<T, U>::op4(a, b); }
template<class T, class U> istForceInline simdvec8 soa_dot38(const T &a, const U &b) { return detail::soa_dot<T, U>::op3(a, b); }
#endif // __ist_enable_soavec8__


///////////////////////////////////////////////////////////////
//  SoA Length Square
///////////////////////////////////////////////////////////////

template<class T> istForceInline simdvec4 soa_lensq24(const T &a) { return detail::soa_dot<T, T>::op2(a, a); }
template<class T> istForceInline simdvec4 soa_lensq34(const T &a) { return detail::soa_dot<T, T>::op3(a, a); }
template<class T> istForceInline simdvec4 soa_lensq44(const T &a) { return detail::soa_dot<T, T>::op4(a, a); }
#ifdef __ist_enable_soavec8__
template<class T> istForceInline simdvec8 soa_lensq28(const T &a) { return detail::soa_dot<T, T>::op2(a, a); }
template<class T> istForceInline simdvec8 soa_lensq38(const T &a) { return detail::soa_dot<T, T>::op3(a, a); }
template<class T> istForceInline simdvec8 soa_lensq48(const T &a) { return detail::soa_dot<T, T>::op4(a, a); }
#endif // __ist_enable_soavec8__


///////////////////////////////////////////////////////////////
//  SoA Length
///////////////////////////////////////////////////////////////

template<class T> istForceInline simdvec4 soa_length24(const T &a) { return glm::sqrt(soa_lensq24<T>(a)); }
template<class T> istForceInline simdvec4 soa_length34(const T &a) { return glm::sqrt(soa_lensq34<T>(a)); }
template<class T> istForceInline simdvec4 soa_length44(const T &a) { return glm::sqrt(soa_lensq44<T>(a)); }
#ifdef __ist_enable_soavec8__
template<class T> istForceInline simdvec8 soa_length28(const T &a) { return glm::sqrt(soa_lensq28<T>(a)); }
template<class T> istForceInline simdvec8 soa_length38(const T &a) { return glm::sqrt(soa_lensq38<T>(a)); }
template<class T> istForceInline simdvec8 soa_length48(const T &a) { return glm::sqrt(soa_lensq48<T>(a)); }
#endif // __ist_enable_soavec8__


///////////////////////////////////////////////////////////////
//  SoA Normalize
///////////////////////////////////////////////////////////////

template<class T> istForceInline soavec24 soa_normalize24(const T &a) { return soa_div24<T>(a, soa_length24<T>(a)); }
template<class T> istForceInline soavec34 soa_normalize34(const T &a) { return soa_div34<T>(a, soa_length34<T>(a)); }
template<class T> istForceInline soavec44 soa_normalize44(const T &a) { return soa_div44<T>(a, soa_length44<T>(a)); }
#ifdef __ist_enable_soavec8__
template<class T> istForceInline soavec28 soa_normalize28(const T &a) { return soa_div28<T>(a, soa_length28<T>(a)); }
template<class T> istForceInline soavec38 soa_normalize38(const T &a) { return soa_div38<T>(a, soa_length38<T>(a)); }
template<class T> istForceInline soavec48 soa_normalize48(const T &a) { return soa_div48<T>(a, soa_length48<T>(a)); }
#endif // __ist_enable_soavec8__


#undef DECLARE_SOATYPES

} // namespace ist
