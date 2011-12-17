#include <windows.h>
#include "../Base.h"

#define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)

#define IST_FINLINE __forceinline


namespace ist {

template<class simdvec> struct simdvec_traits;

template<>
struct simdvec_traits<simdvec4>
{
    enum{ simd_elements = 4 };
};
//
//template<>
//struct soavec_traits<simdvec8>
//{
//    enum{ simd_elements = 4 };
//};


template<class simdvec> struct soavec2x;
template<class simdvec> struct soavec3x;
template<class simdvec> struct soavec4x;

template<class simdvec> struct soavec_traits;

template<class simdvec>
struct soavec_traits< soavec2x<simdvec> >
{
    enum{ soa_elements  = 2 };
    enum{ simd_elements = simdvec_traits<simdvec>::simd_elements };
};

template<class simdvec>
struct soavec_traits< soavec3x<simdvec> >
{
    enum{ soa_elements  = 3 };
    enum{ simd_elements = simdvec_traits<simdvec>::simd_elements };
};

template<class simdvec>
struct soavec_traits< soavec4x<simdvec> >
{
    enum{ soa_elements  = 4 };
    enum{ simd_elements = simdvec_traits<simdvec>::simd_elements };
};



template<class simdvec>
struct tsoavec2x
{
    typedef simdvec simdvec_t;

    simdvec sv[2];

    __forceinline tsoavec2x() {}
    __forceinline tsoavec2x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; }
    __forceinline tsoavec2x(const simdvec &_x, const simdvec &_y) { sv[0]=_x; sv[1]=_y; }

    __forceinline simdvec x() const { return sv[0]; }
    __forceinline simdvec y() const { return sv[1]; }
    __forceinline simdvec v(size_t i) const { return sv[i]; }

    __forceinline void x(const simdvec &_v) { sv[0]=_v; }
    __forceinline void y(const simdvec &_v) { sv[1]=_v; }
    __forceinline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

template<class simdvec>
struct tsoavec3x
{
    typedef simdvec simdvec_t;

    simdvec sv[3];

    __forceinline tsoavec3x() {}
    __forceinline tsoavec3x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; sv[2]=_sv[2]; }
    __forceinline tsoavec3x(const simdvec &_x, const simdvec &_y, const simdvec &_z) { sv[0]=_x; sv[1]=_y; sv[2]=_z; }

    __forceinline simdvec x() const { return sv[0]; }
    __forceinline simdvec y() const { return sv[1]; }
    __forceinline simdvec z() const { return sv[2]; }
    __forceinline simdvec v(size_t i) const { return sv[i]; }

    __forceinline void x(const simdvec &_v) { sv[0]=_v; }
    __forceinline void y(const simdvec &_v) { sv[1]=_v; }
    __forceinline void z(const simdvec &_v) { sv[2]=_v; }
    __forceinline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

template<class simdvec>
struct tsoavec4x
{
    typedef simdvec simdvec_t;
    typedef tsoavec3x<simdvec> soavec3x;

    simdvec sv[4];

    __forceinline tsoavec4x() {}
    __forceinline tsoavec4x(const simdvec *_v) { sv[0]=_sv[0]; sv[1]=_sv[1]; sv[2]=_sv[2]; sv[3]=_sv[3]; }
    __forceinline tsoavec4x(const simdvec &_x, const simdvec &_y, const simdvec &_z, const simdvec &_w) { sv[0]=_x; sv[1]=_y; sv[2]=_z; sv[3]=_w; }
    __forceinline tsoavec4x(const soavec3x &_xyz, const simdvec &_w) { sv[0]=_xyz.sv[0]; sv[1]=_xyz.sv[1]; sv[2]=_xyz.sv[2]; sv[3]=_w; }

    __forceinline simdvec x() const { return sv[0]; }
    __forceinline simdvec y() const { return sv[1]; }
    __forceinline simdvec z() const { return sv[2]; }
    __forceinline simdvec w() const { return sv[3]; }
    __forceinline soavec3x xyz() const { return soavec3x(sv[0], sv[1], sv[2]); }
    __forceinline simdvec v(size_t i) const { return sv[i]; }

    __forceinline void x(const simdvec &_v) { sv[0]=_v; }
    __forceinline void y(const simdvec &_v) { sv[1]=_v; }
    __forceinline void z(const simdvec &_v) { sv[2]=_v; }
    __forceinline void w(const simdvec &_v) { sv[3]=_v; }
    __forceinline void xyz(const soavec3x &_v) { sv[0]=_v.sv[0]; sv[1]=_v.sv[1]; sv[2]=_v.sv[2]; }
    __forceinline void v(size_t i, const simdvec &_v) { sv[i]=_v; }
};

typedef __declspec(align(16)) tsoavec2x<simdvec4> soavec24;
typedef __declspec(align(16)) tsoavec3x<simdvec4> soavec34;
typedef __declspec(align(16)) tsoavec4x<simdvec4> soavec44;
//typedef __declspec(align(32)) tsoavec2x<simdvec8> soavec28;
//typedef __declspec(align(32)) tsoavec3x<simdvec8> soavec38;
//typedef __declspec(align(32)) tsoavec4x<simdvec8> soavec48;


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

    //// todo
    //template<>
    //struct soa_types<8>
    //{
    //    typedef simdvec8 simdvec;
    //    typedef soavec28 soavec2x;
    //    typedef soavec38 soavec3x;
    //    typedef soavec48 soavec4x;
    //};

} // namespace detail





///////////////////////////////////////////////////////////////
//  AoS <-> SoA
///////////////////////////////////////////////////////////////

// 2 要素 SoA ベクトルへ
__forceinline soavec24 soa_transpose24(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec24 soa_transpose24(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    return soavec24(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}


// 3 要素 SoA ベクトルへ
__forceinline soavec34 soa_transpose34(const simdvec4 &v0)
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

__forceinline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1)
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

__forceinline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
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

__forceinline soavec34 soa_transpose34(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
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
__forceinline soavec44 soa_transpose44(const simdvec4 &v0)
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

__forceinline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1)
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

__forceinline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
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

__forceinline soavec44 soa_transpose44(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
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


///////////////////////////////////////////////////////////////
//  SoA Add
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_add
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const U &b)
        {
            return soavec2x(
                a.sv[0] + b.sv[0],
                a.sv[1] + b.sv[1] );
        }

        __forceinline static soavec3x op3(const T &a, const U &b)
        {
            return soavec3x(
                a.sv[0] + b.sv[0],
                a.sv[1] + b.sv[1],
                a.sv[2] + b.sv[2]);
        }

        __forceinline static soavec4x op4(const T &a, const U &b)
        {
            return soavec4x(
                a.sv[0] + b.sv[0],
                a.sv[1] + b.sv[1],
                a.sv[2] + b.sv[2],
                a.sv[3] + b.sv[3]);
        }
    };

    template<class T, size_t S>
    struct soa_add<T, typename soa_types<S>::simdvec, S>
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const simdvec &b)
        {
            return soavec2x(
                a.sv[0] + b,
                a.sv[1] + b);
        }

        __forceinline static soavec3x op3(const T &a, const simdvec &b)
        {
            return soavec3x(
                a.sv[0] + b,
                a.sv[1] + b,
                a.sv[2] + b);
        }

        __forceinline static soavec4x op4(const T &a, const simdvec &b)
        {
            return soavec4x(
                a.sv[0] + b,
                a.sv[1] + b,
                a.sv[2] + b,
                a.sv[3] + b);
        }
    };

} // namespace detail

template<class T, class U> __forceinline soavec24 soa_add24(const T &a, const U &b) { return detail::soa_add<T, U, 4>::op2(a, b); }
template<class T, class U> __forceinline soavec34 soa_add34(const T &a, const U &b) { return detail::soa_add<T, U, 4>::op3(a, b); }
template<class T, class U> __forceinline soavec44 soa_add44(const T &a, const U &b) { return detail::soa_add<T, U, 4>::op4(a, b); }
__forceinline soavec24 operator+(const soavec24 &a, const soavec24 &b) { return soa_add24(a, b); }
__forceinline soavec34 operator+(const soavec34 &a, const soavec34 &b) { return soa_add34(a, b); }
__forceinline soavec44 operator+(const soavec44 &a, const soavec44 &b) { return soa_add44(a, b); }
__forceinline soavec24 operator+(const soavec24 &a, const simdvec4 &b) { return soa_add24(a, b); }
__forceinline soavec34 operator+(const soavec34 &a, const simdvec4 &b) { return soa_add34(a, b); }
__forceinline soavec44 operator+(const soavec44 &a, const simdvec4 &b) { return soa_add44(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Subtract
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_sub
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const U &b)
        {
            return soavec2x(
                a.sv[0] - b.sv[0],
                a.sv[1] - b.sv[1] );
        }

        __forceinline static soavec3x op3(const T &a, const U &b)
        {
            return soavec3x(
                a.sv[0] - b.sv[0],
                a.sv[1] - b.sv[1],
                a.sv[2] - b.sv[2]);
        }

        __forceinline static soavec4x op4(const T &a, const U &b)
        {
            return soavec4x(
                a.sv[0] - b.sv[0],
                a.sv[1] - b.sv[1],
                a.sv[2] - b.sv[2],
                a.sv[3] - b.sv[3]);
        }
    };

    template<class T, size_t S>
    struct soa_sub<T, typename soa_types<S>::simdvec, S>
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const simdvec &b)
        {
            return soavec2x(
                a.sv[0] - b,
                a.sv[1] - b);
        }

        __forceinline static soavec3x op3(const T &a, const simdvec &b)
        {
            return soavec3x(
                a.sv[0] - b,
                a.sv[1] - b,
                a.sv[2] - b);
        }

        __forceinline static soavec4x op4(const T &a, const simdvec &b)
        {
            return soavec4x(
                a.sv[0] - b,
                a.sv[1] - b,
                a.sv[2] - b,
                a.sv[3] - b);
        }
    };

} // namespace detail

template<class T, class U> __forceinline soavec24 soa_sub24(const T &a, const U &b) { return detail::soa_sub<T, U, 4>::op2(a, b); }
template<class T, class U> __forceinline soavec34 soa_sub34(const T &a, const U &b) { return detail::soa_sub<T, U, 4>::op3(a, b); }
template<class T, class U> __forceinline soavec44 soa_sub44(const T &a, const U &b) { return detail::soa_sub<T, U, 4>::op4(a, b); }
__forceinline soavec24 operator-(const soavec24 &a, const soavec24 &b) { return soa_sub24(a, b); }
__forceinline soavec34 operator-(const soavec34 &a, const soavec34 &b) { return soa_sub34(a, b); }
__forceinline soavec44 operator-(const soavec44 &a, const soavec44 &b) { return soa_sub44(a, b); }
__forceinline soavec24 operator-(const soavec24 &a, const simdvec4 &b) { return soa_sub24(a, b); }
__forceinline soavec34 operator-(const soavec34 &a, const simdvec4 &b) { return soa_sub34(a, b); }
__forceinline soavec44 operator-(const soavec44 &a, const simdvec4 &b) { return soa_sub44(a, b); }



///////////////////////////////////////////////////////////////
//  SoA Multiply
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_mul
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const U &b)
        {
            return soavec2x(
                a.sv[0] * b.sv[0],
                a.sv[1] * b.sv[1] );
        }

        __forceinline static soavec3x op3(const T &a, const U &b)
        {
            return soavec3x(
                a.sv[0] * b.sv[0],
                a.sv[1] * b.sv[1],
                a.sv[2] * b.sv[2]);
        }

        __forceinline static soavec4x op4(const T &a, const U &b)
        {
            return soavec4x(
                a.sv[0] * b.sv[0],
                a.sv[1] * b.sv[1],
                a.sv[2] * b.sv[2],
                a.sv[3] * b.sv[3]);
        }
    };

    template<class T, size_t S>
    struct soa_mul<T, typename soa_types<S>::simdvec, S>
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const simdvec &b)
        {
            return soavec2x(
                a.sv[0] * b,
                a.sv[1] * b);
        }

        __forceinline static soavec3x op3(const T &a, const simdvec &b)
        {
            return soavec3x(
                a.sv[0] * b,
                a.sv[1] * b,
                a.sv[2] * b);
        }

        __forceinline static soavec4x op4(const T &a, const simdvec &b)
        {
            return soavec4x(
                a.sv[0] * b,
                a.sv[1] * b,
                a.sv[2] * b,
                a.sv[3] * b);
        }
    };

} // namespace detail

template<class T, class U> __forceinline soavec24 soa_mul24(const T &a, const U &b) { return detail::soa_mul<T, U, 4>::op2(a, b); }
template<class T, class U> __forceinline soavec34 soa_mul34(const T &a, const U &b) { return detail::soa_mul<T, U, 4>::op3(a, b); }
template<class T, class U> __forceinline soavec44 soa_mul44(const T &a, const U &b) { return detail::soa_mul<T, U, 4>::op4(a, b); }
__forceinline soavec24 operator*(const soavec24 &a, const soavec24 &b) { return soa_mul24(a, b); }
__forceinline soavec34 operator*(const soavec34 &a, const soavec34 &b) { return soa_mul34(a, b); }
__forceinline soavec44 operator*(const soavec44 &a, const soavec44 &b) { return soa_mul44(a, b); }
__forceinline soavec24 operator*(const soavec24 &a, const simdvec4 &b) { return soa_mul24(a, b); }
__forceinline soavec34 operator*(const soavec34 &a, const simdvec4 &b) { return soa_mul34(a, b); }
__forceinline soavec44 operator*(const soavec44 &a, const simdvec4 &b) { return soa_mul44(a, b); }



///////////////////////////////////////////////////////////////
//  SoA Divide
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_div
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const U &b)
        {
            return soavec2x(
                a.sv[0] / b.sv[0],
                a.sv[1] / b.sv[1] );
        }

        __forceinline static soavec3x op3(const T &a, const U &b)
        {
            return soavec3x(
                a.sv[0] / b.sv[0],
                a.sv[1] / b.sv[1],
                a.sv[2] / b.sv[2]);
        }

        __forceinline static soavec4x op4(const T &a, const U &b)
        {
            return soavec4x(
                a.sv[0] / b.sv[0],
                a.sv[1] / b.sv[1],
                a.sv[2] / b.sv[2],
                a.sv[3] / b.sv[3]);
        }
    };

    template<class T, size_t S>
    struct soa_div<T, typename soa_types<S>::simdvec, S>
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const simdvec &b)
        {
            return soavec2x(
                a.sv[0] / b,
                a.sv[1] / b);
        }

        __forceinline static soavec3x op3(const T &a, const simdvec &b)
        {
            return soavec3x(
                a.sv[0] / b,
                a.sv[1] / b,
                a.sv[2] / b);
        }

        __forceinline static soavec4x op4(const T &a, const simdvec &b)
        {
            return soavec4x(
                a.sv[0] / b,
                a.sv[1] / b,
                a.sv[2] / b,
                a.sv[3] / b);
        }
    };

} // namespace detail

template<class T, class U> __forceinline soavec24 soa_div24(const T &a, const U &b) { return detail::soa_div<T, U, 4>::op2(a, b); }
template<class T, class U> __forceinline soavec34 soa_div34(const T &a, const U &b) { return detail::soa_div<T, U, 4>::op3(a, b); }
template<class T, class U> __forceinline soavec44 soa_div44(const T &a, const U &b) { return detail::soa_div<T, U, 4>::op4(a, b); }
__forceinline soavec24 operator/(const soavec24 &a, const soavec24 &b) { return soa_div24(a, b); }
__forceinline soavec34 operator/(const soavec34 &a, const soavec34 &b) { return soa_div34(a, b); }
__forceinline soavec44 operator/(const soavec44 &a, const soavec44 &b) { return soa_div44(a, b); }
__forceinline soavec24 operator/(const soavec24 &a, const simdvec4 &b) { return soa_div24(a, b); }
__forceinline soavec34 operator/(const soavec34 &a, const simdvec4 &b) { return soa_div34(a, b); }
__forceinline soavec44 operator/(const soavec44 &a, const simdvec4 &b) { return soa_div44(a, b); }



///////////////////////////////////////////////////////////////
//  SoA Select
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_mix
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const T &b, const U &s)
        {
            return soavec2x(
                glm::mix(a.x, b.x, s.x),
                glm::mix(a.y, b.y, s.y));
        }

        __forceinline static soavec3x op3(const T &a, const T &b, const U &s)
        {
            return soavec3x(
                glm::mix(a.x, b.x, s.x),
                glm::mix(a.y, b.y, s.y),
                glm::mix(a.z, b.z, s.z));
        }

        __forceinline static soavec4x op4(const T &a, const T &b, const U &s)
        {
            return soavec4x(
                glm::mix(a.x, b.x, s.x),
                glm::mix(a.y, b.y, s.y),
                glm::mix(a.z, b.z, s.z),
                glm::mix(a.w, b.w, s.w));
        }
    };

    template<class T, size_t S>
    struct soa_mix<T, typename soa_types<S>::simdvec, S>
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const T &b, const simdvec &s)
        {
            return soavec2x(
                glm::mix(a.x, b.x, s),
                glm::mix(a.y, b.y, s));
        }

        __forceinline static soavec3x op3(const T &a, const T &b, const simdvec &s)
        {
            return soavec3x(
                glm::mix(a.x, b.x, s),
                glm::mix(a.y, b.y, s),
                glm::mix(a.z, b.z, s));
        }

        __forceinline static soavec4x op4(const T &a, const T &b, const simdvec &s)
        {
            return soavec4x(
                glm::mix(a.x, b.x, s),
                glm::mix(a.y, b.y, s),
                glm::mix(a.z, b.z, s),
                glm::mix(a.w, b.w, s));
        }
    };

} // namespace detail

template<class T, class U> __forceinline soavec24 soa_mix24(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U, 4>::op2(a, b, s); }
template<class T, class U> __forceinline soavec34 soa_mix34(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U, 4>::op3(a, b, s); }
template<class T, class U> __forceinline soavec44 soa_mix44(const T &a, const T &b, const U &s) { return detail::soa_mix<T, U, 4>::op4(a, b, s); }



///////////////////////////////////////////////////////////////
//  SoA Dot
///////////////////////////////////////////////////////////////

namespace detail {

    template<class T, class U, size_t S>
    struct soa_dot
    {
        typedef typename soa_types<S>::simdvec simdvec;
        typedef typename soa_types<S>::soavec2x soavec2x;
        typedef typename soa_types<S>::soavec3x soavec3x;
        typedef typename soa_types<S>::soavec4x soavec4x;

        __forceinline static soavec2x op2(const T &a, const U &b)
        {
            soavec2x sq = detail::soa_mul<T, U, S>::op2(a, b);
            return sq.sv[0] + sq.sv[1];
        }

        __forceinline static soavec3x op3(const T &a, const U &b)
        {
            soavec3x sq = detail::soa_mul<T, U, S>::op3(a, b);
            return sq.sv[0] + sq.sv[1] + sq.sv[2];
        }

        __forceinline static soavec4x op4(const T &a, const U &b)
        {
            soavec4x sq = detail::soa_mul<T, U, S>::op4(a, b);
            return sq.sv[0] + sq.sv[1] + sq.sv[2] + sq.sv[3];
        }
    };

} // namespace detail

template<class T, class U> __forceinline simdvec4 soa_dot24(const T &a, const U &b) { return detail::soa_dot<T, U, 4>::op2(a, b); }
template<class T, class U> __forceinline simdvec4 soa_dot44(const T &a, const U &b) { return detail::soa_dot<T, U, 4>::op4(a, b); }
template<class T, class U> __forceinline simdvec4 soa_dot34(const T &a, const U &b) { return detail::soa_dot<T, U, 4>::op3(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Length Square
///////////////////////////////////////////////////////////////

template<class T> __forceinline simdvec4 soa_lensq24(const T &a) { return detail::soa_dot<T, T, 4>::op2(a, a); }
template<class T> __forceinline simdvec4 soa_lensq34(const T &a) { return detail::soa_dot<T, T, 4>::op3(a, a); }
template<class T> __forceinline simdvec4 soa_lensq44(const T &a) { return detail::soa_dot<T, T, 4>::op4(a, a); }


///////////////////////////////////////////////////////////////
//  SoA Length
///////////////////////////////////////////////////////////////

template<class T> __forceinline simdvec4 soa_length24(const T &a) { return glm::sqrt(soa_lensq24<T>(a)); }
template<class T> __forceinline simdvec4 soa_length34(const T &a) { return glm::sqrt(soa_lensq34<T>(a)); }
template<class T> __forceinline simdvec4 soa_length44(const T &a) { return glm::sqrt(soa_lensq44<T>(a)); }


///////////////////////////////////////////////////////////////
//  SoA Normalize
///////////////////////////////////////////////////////////////

template<class T> __forceinline soavec24 soa_normalize24(const T &a) { return soa_div24<T>(a, soa_length24<T>(a)); }
template<class T> __forceinline soavec34 soa_normalize34(const T &a) { return soa_div34<T>(a, soa_length34<T>(a)); }
template<class T> __forceinline soavec44 soa_normalize44(const T &a) { return soa_div44<T>(a, soa_length44<T>(a)); }


} // namespace ist
