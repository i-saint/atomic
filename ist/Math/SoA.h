#include <windows.h>
#include <xnamath.h>

#define SSE_SHUFFLE(w,x,y,z) _MM_SHUFFLE(z,y,x,w)



union __declspec(align(16)) SOAVECTOR2
{
    struct {
        XMVECTOR x;
        XMVECTOR y;
    };
    XMVECTOR v[2];
};

union __declspec(align(16)) SOAVECTOR3
{
    struct {
        XMVECTOR x;
        XMVECTOR y;
        XMVECTOR z;
    };
    XMVECTOR v[3];
};

union __declspec(align(16)) SOAVECTOR4
{
    struct {
        XMVECTOR x;
        XMVECTOR y;
        XMVECTOR z;
        XMVECTOR w;
    };
    XMVECTOR v[4];
};



///////////////////////////////////////////////////////////////
// SoA Set
///////////////////////////////////////////////////////////////

__forceinline SOAVECTOR2 SOAVectorSet2(const XMVECTOR *v)
{
    SOAVECTOR2 r;
    r.x = v[0];
    r.y = v[1];
    return r;
}

__forceinline SOAVECTOR2 SOAVectorSet2(const XMVECTOR v0, const XMVECTOR v1)
{
    SOAVECTOR2 r;
    r.x = v0;
    r.y = v1;
    return r;
}

__forceinline SOAVECTOR3 SOAVectorSet3(const XMVECTOR *v)
{
    SOAVECTOR3 r;
    r.x = v[0];
    r.y = v[1];
    r.z = v[2];
    return r;
}

__forceinline SOAVECTOR3 SOAVectorSet3(const XMVECTOR v0, const XMVECTOR v1, const XMVECTOR v2)
{
    SOAVECTOR3 r;
    r.x = v0;
    r.y = v1;
    r.z = v2;
    return r;
}

__forceinline SOAVECTOR4 SOAVectorSet4(const XMVECTOR *v)
{
    SOAVECTOR4 r;
    r.x = v[0];
    r.y = v[1];
    r.z = v[2];
    r.w = v[3];
    return r;
}

// アライメント指定オブジェクトが 4 引数以上の場合、VC の都合で参照で渡さないといけない…
__forceinline SOAVECTOR4 SOAVectorSet4(const XMVECTOR& v0, const XMVECTOR& v1, const XMVECTOR& v2, const XMVECTOR& v3)
{
    SOAVECTOR4 r;
    r.x = v0;
    r.y = v1;
    r.z = v2;
    r.w = v3;
    return r;
}


///////////////////////////////////////////////////////////////
//  AoS <-> SoA
///////////////////////////////////////////////////////////////

// 2 要素 SoA ベクトルへ
__forceinline SOAVECTOR2 SOAVectorTranspose2(const XMVECTOR v0)
{
    SOAVECTOR2 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, zero);
    XMVECTOR r2 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR2 SOAVectorTranspose2(const XMVECTOR v0, const XMVECTOR v1)
{
    SOAVECTOR2 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR2 SOAVectorTranspose2(const XMVECTOR v0, const XMVECTOR v1, const XMVECTOR v2)
{
    SOAVECTOR2 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, zero);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR2 SOAVectorTranspose2(const XMVECTOR& v0, const XMVECTOR& v1, const XMVECTOR& v2, const XMVECTOR& v3)
{
    SOAVECTOR2 r;
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, v3);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    return r;
}


__forceinline SOAVECTOR3 SOAVectorTranspose3(const XMVECTOR v0)
{
    SOAVECTOR3 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, zero);
    XMVECTOR r2 = zero;
    XMVECTOR r3 = _mm_unpackhi_ps(v0, zero);
    XMVECTOR r4 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    return r;
}


// 3 要素 SoA ベクトルへ
__forceinline SOAVECTOR3 SOAVectorTranspose3(const XMVECTOR v0, const XMVECTOR v1)
{
    SOAVECTOR3 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = zero;
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    return r;
}

__forceinline SOAVECTOR3 SOAVectorTranspose3(const XMVECTOR v0, const XMVECTOR v1, const XMVECTOR v2)
{
    SOAVECTOR3 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, zero);
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = _mm_unpackhi_ps(v2, zero);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    return r;
}

__forceinline SOAVECTOR3 SOAVectorTranspose3(const XMVECTOR& v0, const XMVECTOR& v1, const XMVECTOR& v2, const XMVECTOR& v3)
{
    SOAVECTOR3 r;
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, v3);
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = _mm_unpackhi_ps(v2, v3);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    return r;
}


// 4 要素 SoA ベクトルへ
__forceinline SOAVECTOR4 SOAVectorTranspose4(const XMVECTOR v0)
{
    SOAVECTOR4 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, zero);
    XMVECTOR r2 = zero;
    XMVECTOR r3 = _mm_unpackhi_ps(v0, zero);
    XMVECTOR r4 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    r.w = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR4 SOAVectorTranspose4(const XMVECTOR v0, const XMVECTOR v1)
{
    SOAVECTOR4 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = zero;
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = zero;
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    r.w = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR4 SOAVectorTranspose4(const XMVECTOR v0, const XMVECTOR v1, const XMVECTOR v2)
{
    SOAVECTOR4 r;
    XMVECTOR zero = _mm_set_ps1(0.0f);
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, zero);
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = _mm_unpackhi_ps(v2, zero);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    r.w = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3));
    return r;
}

__forceinline SOAVECTOR4 SOAVectorTranspose4(const XMVECTOR& v0, const XMVECTOR& v1, const XMVECTOR& v2, const XMVECTOR& v3)
{
    SOAVECTOR4 r;
    XMVECTOR r1 = _mm_unpacklo_ps(v0, v1);
    XMVECTOR r2 = _mm_unpacklo_ps(v2, v3);
    XMVECTOR r3 = _mm_unpackhi_ps(v0, v1);
    XMVECTOR r4 = _mm_unpackhi_ps(v2, v3);
    r.x = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(0,1,0,1));
    r.y = _mm_shuffle_ps(r1, r2, SSE_SHUFFLE(2,3,2,3));
    r.z = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(0,1,0,1));
    r.w = _mm_shuffle_ps(r3, r4, SSE_SHUFFLE(2,3,2,3));
    return r;
}


///////////////////////////////////////////////////////////////
//  SoA Add
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline SOAVECTOR2 SOAVectorAdd2(const T a, const U b)
{
    SOAVECTOR2 r;
    r.x = XMVectorAdd(a.x, b.x);
    r.y = XMVectorAdd(a.y, b.y);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR3 SOAVectorAdd3(const T a, const U b)
{
    SOAVECTOR3 r;
    r.x = XMVectorAdd(a.x, b.x);
    r.y = XMVectorAdd(a.y, b.y);
    r.z = XMVectorAdd(a.z, b.z);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR4 SOAVectorAdd4(const T a, const U b)
{
    SOAVECTOR4 r;
    r.x = XMVectorAdd(a.x, b.x);
    r.y = XMVectorAdd(a.y, b.y);
    r.z = XMVectorAdd(a.z, b.z);
    r.w = XMVectorAdd(a.w, b.w);
    return r;
}


///////////////////////////////////////////////////////////////
//  SoA Subtract
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline SOAVECTOR2 SOAVectorSubtract2(const T a, const U b)
{
    SOAVECTOR2 r;
    r.x = XMVectorSubtract(a.x, b.x);
    r.y = XMVectorSubtract(a.y, b.y);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR3 SOAVectorSubtract3(const T a, const U b)
{
    SOAVECTOR3 r;
    r.x = XMVectorSubtract(a.x, b.x);
    r.y = XMVectorSubtract(a.y, b.y);
    r.z = XMVectorSubtract(a.z, b.z);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR4 SOAVectorSubtract4(const T a, const U b)
{
    SOAVECTOR4 r;
    r.x = XMVectorSubtract(a.x, b.x);
    r.y = XMVectorSubtract(a.y, b.y);
    r.z = XMVectorSubtract(a.z, b.z);
    r.w = XMVectorSubtract(a.w, b.w);
    return r;
}


///////////////////////////////////////////////////////////////
//  SoA Multiply
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline SOAVECTOR2 SOAVectorMultiply2(const T a, const U b)
{
    SOAVECTOR2 r;
    r.x = XMVectorMultiply(a.x, b.x);
    r.y = XMVectorMultiply(a.y, b.y);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR3 SOAVectorMultiply3(const T a, const U b)
{
    SOAVECTOR3 r;
    r.x = XMVectorMultiply(a.x, b.x);
    r.y = XMVectorMultiply(a.y, b.y);
    r.z = XMVectorMultiply(a.z, b.z);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR4 SOAVectorMultiply4(const T a, const U b)
{
    SOAVECTOR4 r;
    r.x = XMVectorMultiply(a.x, b.x);
    r.y = XMVectorMultiply(a.y, b.y);
    r.z = XMVectorMultiply(a.z, b.z);
    r.w = XMVectorMultiply(a.w, b.w);
    return r;
}

// template 関数の部分特殊化はできず、構造体用意して部分特殊化はさもうとすると
// "__declspec(align('16')) の仮引数は配置されません"
// が出るので、XMVECTOR を取るバージョンは別名にします…。
template<class T>
__forceinline SOAVECTOR2 SOAVectorMultiply2S(const T a, const XMVECTOR b)
{
    SOAVECTOR2 r;
    r.x = XMVectorMultiply(a.x, b);
    r.y = XMVectorMultiply(a.y, b);
    return r;
}

template<class T>
__forceinline SOAVECTOR3 SOAVectorMultiply3S(const T a, const XMVECTOR b)
{
    SOAVECTOR3 r;
    r.x = XMVectorMultiply(a.x, b);
    r.y = XMVectorMultiply(a.y, b);
    r.z = XMVectorMultiply(a.z, b);
    return r;
}

template<class T>
__forceinline SOAVECTOR4 SOAVectorMultiply4S(const T a, const XMVECTOR b)
{
    SOAVECTOR4 r;
    r.x = XMVectorMultiply(a.x, b);
    r.y = XMVectorMultiply(a.y, b);
    r.z = XMVectorMultiply(a.z, b);
    r.w = XMVectorMultiply(a.w, b);
    return r;
}

///////////////////////////////////////////////////////////////
//  SoA Divide
///////////////////////////////////////////////////////////////

template<class T, class U>
__forceinline SOAVECTOR2 SOAVectorDivide2(const T a, const U b)
{
    SOAVECTOR2 r;
    r.x = XMVectorDivide(a.x, b.x);
    r.y = XMVectorDivide(a.y, b.y);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR3 SOAVectorDivide3(const T a, const U b)
{
    SOAVECTOR3 r;
    r.x = XMVectorDivide(a.x, b.x);
    r.y = XMVectorDivide(a.y, b.y);
    r.z = XMVectorDivide(a.z, b.z);
    return r;
}

template<class T, class U>
__forceinline SOAVECTOR4 SOAVectorDivide4(const T a, const U b)
{
    SOAVECTOR4 r;
    r.x = XMVectorDivide(a.x, b.x);
    r.y = XMVectorDivide(a.y, b.y);
    r.z = XMVectorDivide(a.z, b.z);
    r.w = XMVectorDivide(a.w, b.w);
    return r;
}

template<class T>
__forceinline SOAVECTOR2 SOAVectorDivide2S(const T a, const XMVECTOR b)
{
    SOAVECTOR2 r;
    r.x = XMVectorDivide(a.x, b);
    r.y = XMVectorDivide(a.y, b);
    return r;
}

template<class T>
__forceinline SOAVECTOR3 SOAVectorDivide3S(const T a, const XMVECTOR b)
{
    SOAVECTOR3 r;
    r.x = XMVectorDivide(a.x, b);
    r.y = XMVectorDivide(a.y, b);
    r.z = XMVectorDivide(a.z, b);
    return r;
}

template<class T>
__forceinline SOAVECTOR4 SOAVectorDivide4S(const T a, const XMVECTOR b)
{
    SOAVECTOR4 r;
    r.x = XMVectorDivide(a.x, b);
    r.y = XMVectorDivide(a.y, b);
    r.z = XMVectorDivide(a.z, b);
    r.w = XMVectorDivide(a.w, b);
    return r;
}


///////////////////////////////////////////////////////////////
//  SoA Length Square
///////////////////////////////////////////////////////////////

template<class T>
__forceinline XMVECTOR SOAVectorLengthSquare2(const T a)
{
    SOAVECTOR2 square = SOAVectorMultiply2<T, T>(a, a);
    return XMVectorAdd(square.x, square.y);
}

template<class T>
__forceinline XMVECTOR SOAVectorLengthSquare3(const T a)
{
    SOAVECTOR3 square = SOAVectorMultiply3<T, T>(a, a);
    return XMVectorAdd(XMVectorAdd(square.x, square.y), square.z);
}

template<class T>
__forceinline XMVECTOR SOAVectorLengthSquare4(const T a)
{
    SOAVECTOR4 square = SOAVectorMultiply4<T, T>(a, a);
    return XMVectorAdd(XMVectorAdd(XMVectorAdd(square.x, square.y), square.z), square.w);
}


///////////////////////////////////////////////////////////////
//  SoA Length
///////////////////////////////////////////////////////////////

template<class T>
__forceinline XMVECTOR SOAVectorLength2(const T a)
{
    return XMVectorSqrt(SOAVectorLengthSquare2<T>(a));
}

template<class T>
__forceinline XMVECTOR SOAVectorLength3(const T a)
{
    return XMVectorSqrt(SOAVectorLengthSquare3<T>(a));
}

template<class T>
__forceinline XMVECTOR SOAVectorLength4(const T a)
{
    return XMVectorSqrt(SOAVectorLengthSquare4<T>(a));
}


///////////////////////////////////////////////////////////////
//  SoA Normalize
///////////////////////////////////////////////////////////////

template<class T>
__forceinline SOAVECTOR2 SOAVectorNormalize2(const T a)
{
    XMVECTOR len = SOAVectorLength2(a);
    SOAVECTOR2 r = SOAVectorDivide2S(a, len);
    return r;
}

template<class T>
__forceinline SOAVECTOR3 SOAVectorNormalize3(const T a)
{
    XMVECTOR len = SOAVectorLength3(a);
    SOAVECTOR3 r = SOAVectorDivide3S(a, len);
    return r;
}

template<class T>
__forceinline SOAVECTOR4 SOAVectorNormalize4(const T a)
{
    XMVECTOR len = SOAVectorLength4(a);
    SOAVECTOR4 r = SOAVectorDivide4S(a, len);
    return r;
}

