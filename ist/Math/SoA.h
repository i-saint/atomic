#include <windows.h>
#include "../Base.h"

#define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)


namespace ist {

struct __declspec(align(16)) soavec2
{
    simdvec4 v[2];

    __forceinline soavec2() {}
    __forceinline soavec2(const simdvec4 *_v) { v[0]=_v[0]; v[1]=_v[1]; }
    __forceinline soavec2(const simdvec4 &_x, const simdvec4 &_y) { v[0]=_x; v[1]=_y; }

    __forceinline simdvec4 x() const { return v[0]; }
    __forceinline simdvec4 y() const { return v[1]; }

    __forceinline void x(const simdvec4 &_v) { v[0]=_v; }
    __forceinline void y(const simdvec4 &_v) { v[1]=_v; }
};

struct __declspec(align(16)) soavec3
{
    simdvec4 v[3];

    __forceinline soavec3() {}
    __forceinline soavec3(const simdvec4 *_v) { v[0]=_v[0]; v[1]=_v[1]; v[2]=_v[2]; }
    __forceinline soavec3(const simdvec4 &_x, const simdvec4 &_y, const simdvec4 &_z) { v[0]=_x; v[1]=_y; v[2]=_z; }

    __forceinline simdvec4 x() const { return v[0]; }
    __forceinline simdvec4 y() const { return v[1]; }
    __forceinline simdvec4 z() const { return v[2]; }

    __forceinline void x(const simdvec4 &_v) { v[0]=_v; }
    __forceinline void y(const simdvec4 &_v) { v[1]=_v; }
    __forceinline void z(const simdvec4 &_v) { v[2]=_v; }
};

struct __declspec(align(16)) soavec4
{
    simdvec4 v[4];

    __forceinline soavec4() {}
    __forceinline soavec4(const simdvec4 *_v) { v[0]=_v[0]; v[1]=_v[1]; v[2]=_v[2]; v[3]=_v[3]; }
    __forceinline soavec4(const simdvec4 &_x, const simdvec4 &_y, const simdvec4 &_z, const simdvec4 &_w) { v[0]=_x; v[1]=_y; v[2]=_z; v[3]=_w; }
    __forceinline soavec4(const soavec3 &_xyz, const simdvec4 &_w) { v[0]=_xyz.v[0]; v[1]=_xyz.v[1]; v[2]=_xyz.v[2]; v[3]=_w; }

    __forceinline simdvec4 x() const { return v[0]; }
    __forceinline simdvec4 y() const { return v[1]; }
    __forceinline simdvec4 z() const { return v[2]; }
    __forceinline simdvec4 w() const { return v[3]; }
    __forceinline soavec3 xyz() const { return soavec3(v[0], v[1], v[2]); }

    __forceinline void x(const simdvec4 &_v) { v[0]=_v; }
    __forceinline void y(const simdvec4 &_v) { v[1]=_v; }
    __forceinline void z(const simdvec4 &_v) { v[2]=_v; }
    __forceinline void w(const simdvec4 &_v) { v[3]=_v; }
    __forceinline void xyz(const soavec3 &_v) { v[0]=_v.v[0]; v[1]=_v.v[1]; v[2]=_v.v[2]; }
};




///////////////////////////////////////////////////////////////
//  AoS <-> SoA
///////////////////////////////////////////////////////////////

// 2 要素 SoA ベクトルへ
__forceinline soavec2 soa_transpose2(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    return soavec2(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec2 soa_transpose2(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    return soavec2(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec2 soa_transpose2(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    return soavec2(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec2 soa_transpose2(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    return soavec2(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)) );
}


// 3 要素 SoA ベクトルへ
__forceinline soavec3 soa_transpose3(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, zero);
    __m128 r4 = zero;
    return soavec3(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

__forceinline soavec3 soa_transpose3(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = zero;
    return soavec3(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

__forceinline soavec3 soa_transpose3(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, zero);
    return soavec3(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}

__forceinline soavec3 soa_transpose3(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, v3.Data);
    return soavec3(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)) );
}


// 4 要素 SoA ベクトルへ
__forceinline soavec4 soa_transpose4(const simdvec4 &v0)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, zero);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, zero);
    __m128 r4 = zero;
    return soavec4(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec4 soa_transpose4(const simdvec4 &v0, const simdvec4 &v1)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = zero;
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = zero;
    return soavec4(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec4 soa_transpose4(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2)
{
    __m128 zero = _mm_set_ps1(0.0f);
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, zero);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, zero);
    return soavec4(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}

__forceinline soavec4 soa_transpose4(const simdvec4 &v0, const simdvec4 &v1, const simdvec4 &v2, const simdvec4 &v3)
{
    __m128 r1 = _mm_unpacklo_ps(v0.Data, v1.Data);
    __m128 r2 = _mm_unpacklo_ps(v2.Data, v3.Data);
    __m128 r3 = _mm_unpackhi_ps(v0.Data, v1.Data);
    __m128 r4 = _mm_unpackhi_ps(v2.Data, v3.Data);
    return soavec4(
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1)),
        _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3)) );
}


///////////////////////////////////////////////////////////////
//  SoA Add
///////////////////////////////////////////////////////////////

// コピー渡しの方が望ましいはずですが、なんか Debug 版だとアドレスが 16 の倍数じゃなくなってクラッシュすることがあるので参照渡しにしています。
// 最適化有効時はちゃんとインライン化されてるので速度面ではたぶん大丈夫。
template<class T, class U>
__forceinline soavec2 soa_add2(const T &a, const U &b)
{
    return soavec2(
        a.v[0] + b.v[0],
        a.v[1] + b.v[1] );
}

template<class T, class U>
__forceinline soavec3 soa_add3(const T &a, const U &b)
{
    return soavec3(
        a.v[0] + b.v[0],
        a.v[1] + b.v[1],
        a.v[2] + b.v[2]);
}

template<class T, class U>
__forceinline soavec4 soa_add4(const T &a, const U &b)
{
    return soavec4(
        a.v[0] + b.v[0],
        a.v[1] + b.v[1],
        a.v[2] + b.v[2],
        a.v[3] + b.v[3]);
}

__forceinline soavec2 operator+(const soavec2 &a, const soavec2 &b) { return soa_add2(a, b); }
__forceinline soavec3 operator+(const soavec3 &a, const soavec3 &b) { return soa_add3(a, b); }
__forceinline soavec4 operator+(const soavec4 &a, const soavec4 &b) { return soa_add4(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Subtract
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline soavec2 soa_sub2(const T& a, const U& b)
{
    return soavec2(
        a.v[0] - b.v[0],
        a.v[1] - b.v[1] );
}

template<class T, class U>
__forceinline soavec3 soa_sub3(const T& a, const U& b)
{
    return soavec3(
        a.v[0] - b.v[0],
        a.v[1] - b.v[1],
        a.v[2] - b.v[2]);
}

template<class T, class U>
__forceinline soavec4 soa_sub4(const T& a, const U& b)
{
    return soavec4(
        a.v[0] - b.v[0],
        a.v[1] - b.v[1],
        a.v[2] - b.v[2],
        a.v[3] - b.v[3]);
}

__forceinline soavec2 operator-(const soavec2 &a, const soavec2 &b) { return soa_sub2(a, b); }
__forceinline soavec3 operator-(const soavec3 &a, const soavec3 &b) { return soa_sub3(a, b); }
__forceinline soavec4 operator-(const soavec4 &a, const soavec4 &b) { return soa_sub4(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Multiply
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline soavec2 soa_mul2(const T &a, const U &b)
{
    return soavec2(
        a.v[0] * b.v[0],
        a.v[1] * b.v[1] );
}

template<class T, class U>
__forceinline soavec3 soa_mul3(const T &a, const U &b)
{
    return soavec3(
        a.v[0] * b.v[0],
        a.v[1] * b.v[1],
        a.v[2] * b.v[2]);
}

template<class T, class U>
__forceinline soavec4 soa_mul4(const T &a, const U &b)
{
    return soavec4(
        a.v[0] * b.v[0],
        a.v[1] * b.v[1],
        a.v[2] * b.v[2],
        a.v[3] * b.v[3]);
}

__forceinline soavec2 operator*(const soavec2 &a, const soavec2 &b) { return soa_mul2(a, b); }
__forceinline soavec3 operator*(const soavec3 &a, const soavec3 &b) { return soa_mul3(a, b); }
__forceinline soavec4 operator*(const soavec4 &a, const soavec4 &b) { return soa_mul4(a, b); }


// template 関数の部分特殊化はできず、構造体用意して部分特殊化はさもうとすると
// "__declspec(align('16')) の仮引数は配置されません"
// が出るので、simdvec4 を取るバージョンは別名にします…。
template<class T>
__forceinline soavec2 soa_mul2s(const T &a, const simdvec4 &b)
{
    return soavec2(
        a.v[0] * b,
        a.v[1] * b);
}

template<class T>
__forceinline soavec3 soa_mul3s(const T &a, const simdvec4 &b)
{
    return soavec3(
        a.v[0] * b,
        a.v[1] * b,
        a.v[2] * b);
}

template<class T>
__forceinline soavec4 soa_mul4s(const T &a, const simdvec4 &b)
{
    return soavec4(
        a.v[0] * b,
        a.v[1] * b,
        a.v[2] * b,
        a.v[3] * b);
}

__forceinline soavec2 operator*(const soavec2 &a, const simdvec4 &b) { return soa_mul2s(a, b); }
__forceinline soavec3 operator*(const soavec3 &a, const simdvec4 &b) { return soa_mul3s(a, b); }
__forceinline soavec4 operator*(const soavec4 &a, const simdvec4 &b) { return soa_mul4s(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Divide
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline soavec2 soa_div2(const T &a, const U &b)
{
    return soavec2(
        a.v[0] / b.v[0],
        a.v[1] / b.v[1] );
}

template<class T, class U>
__forceinline soavec3 soa_div3(const T &a, const U &b)
{
    return soavec3(
        a.v[0] / b.v[0],
        a.v[1] / b.v[1],
        a.v[2] / b.v[2]);
}

template<class T, class U>
__forceinline soavec4 soa_div4(const T &a, const U &b)
{
    return soavec4(
        a.v[0] / b.v[0],
        a.v[1] / b.v[1],
        a.v[2] / b.v[2],
        a.v[3] / b.v[3]);
}

__forceinline soavec2 operator/(const soavec2 &a, const soavec2 &b) { return soa_div2(a, b); }
__forceinline soavec3 operator/(const soavec3 &a, const soavec3 &b) { return soa_div3(a, b); }
__forceinline soavec4 operator/(const soavec4 &a, const soavec4 &b) { return soa_div4(a, b); }


template<class T>
__forceinline soavec2 soa_div2s(const T &a, const simdvec4 &b)
{
    return soavec2(
        a.v[0] / b,
        a.v[1] / b);
}

template<class T>
__forceinline soavec3 soa_div3s(const T &a, const simdvec4 &b)
{
    return soavec3(
        a.v[0] / b,
        a.v[1] / b,
        a.v[2] / b);
}

template<class T>
__forceinline soavec4 soa_div4s(const T &a, const simdvec4 &b)
{
    return soavec4(
        a.v[0] / b,
        a.v[1] / b,
        a.v[2] / b,
        a.v[3] / b);
}

__forceinline soavec2 operator/(const soavec2 &a, const simdvec4 &b) { return soa_div2s(a, b); }
__forceinline soavec3 operator/(const soavec3 &a, const simdvec4 &b) { return soa_div3s(a, b); }
__forceinline soavec4 operator/(const soavec4 &a, const simdvec4 &b) { return soa_div4s(a, b); }


///////////////////////////////////////////////////////////////
//  SoA Select
///////////////////////////////////////////////////////////////

template<class T>
__forceinline soavec2 soa_mix2(const T &a, const T &b, const T &s)
{
    return soavec2(
        glm::mix(a.x, b.x, s.x),
        glm::mix(a.y, b.y, s.y));
}

template<class T>
__forceinline soavec3 soa_mix3(const T &a, const T &b, const T &s)
{
    return soavec3(
        glm::mix(a.x, b.x, s.x),
        glm::mix(a.y, b.y, s.y),
        glm::mix(a.z, b.z, s.z));
}

template<class T>
__forceinline soavec4 soa_mix4(const T &a, const T &b, const T &s)
{
    return soavec4(
        glm::mix(a.x, b.x, s.x),
        glm::mix(a.y, b.y, s.y),
        glm::mix(a.z, b.z, s.z),
        glm::mix(a.w, b.w, s.w));
}


template<class T>
__forceinline soavec2 soa_mix2s(const T &a, const T &b, const simdvec4 &s)
{
    return soavec2(
        glm::mix(a.x, b.x, s),
        glm::mix(a.y, b.y, s));
}

template<class T>
__forceinline soavec3 soa_mix3s(const T &a, const T &b, const simdvec4 &s)
{
    return soavec3(
        glm::mix(a.x, b.x, s),
        glm::mix(a.y, b.y, s),
        glm::mix(a.z, b.z, s));
}

template<class T>
__forceinline soavec4 soa_mix4s(const T &a, const T &b, const simdvec4 &s)
{
    return soavec4(
        glm::mix(a.x, b.x, s),
        glm::mix(a.y, b.y, s),
        glm::mix(a.z, b.z, s),
        glm::mix(a.w, b.w, s));
}


///////////////////////////////////////////////////////////////
//  SoA Length Square
///////////////////////////////////////////////////////////////

template<class T>
__forceinline simdvec4 soa_lensq2(const T &a)
{
    soavec2 sq = soa_mul2<T, T>(a, a);
    return sq.v[0] + sq.v[1];
}

template<class T>
__forceinline simdvec4 soa_lensq3(const T &a)
{
    soavec3 sq = soa_mul3<T, T>(a, a);
    return sq.v[0] + sq.v[1] + sq.v[2];
}

template<class T>
__forceinline simdvec4 soa_lensq4(const T &a)
{
    soavec4 sq = soa_mul4<T, T>(a, a);
    return sq.v[0] + sq.v[1] + sq.v[2] + sq.v[3];
}


///////////////////////////////////////////////////////////////
//  SoA Length
///////////////////////////////////////////////////////////////

template<class T>
__forceinline simdvec4 soa_length2(const T &a)
{
    return glm::sqrt(soa_lensq2<T>(a));
}

template<class T>
__forceinline simdvec4 soa_length3(const T &a)
{
    return glm::sqrt(soa_lensq3<T>(a));
}

template<class T>
__forceinline simdvec4 soa_length4(const T &a)
{
    return glm::sqrt(soa_lensq4<T>(a));
}


///////////////////////////////////////////////////////////////
//  SoA Normalize
///////////////////////////////////////////////////////////////

template<class T>
__forceinline soavec2 soa_normalize2(const T &a)
{
    return soa_div2s(a, soa_length2(a));
}

template<class T>
__forceinline soavec3 soa_normalize3(const T &a)
{
    return soa_div3s(a, soa_length3(a));
}

template<class T>
__forceinline soavec4 soa_normalize4(const T &a)
{
    return soa_div4s(a, soa_length4(a));
}


} // namespace ist
