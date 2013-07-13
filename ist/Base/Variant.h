#ifndef ist_Base_Variant_h
#define ist_Base_Variant_h
#include "ist/Base/Serialize.h"
#include "ist/stdex/crtex.h"

namespace ist {

template<size_t S> inline void _ZeroClear(char (&buf)[S]) { istMemset(buf, 0, S); }
template<> inline void _ZeroClear<1>(char (&buf)[1]) { buf[0]=0; }
template<> inline void _ZeroClear<2>(char (&buf)[2]) { reinterpret_cast<int16&>(*buf)=0; }
template<> inline void _ZeroClear<4>(char (&buf)[4]) { reinterpret_cast<int32&>(*buf)=0; }
template<> inline void _ZeroClear<8>(char (&buf)[8]) { reinterpret_cast<int64&>(*buf)=0; }
template<> inline void _ZeroClear<16>(char (&buf)[16]) { reinterpret_cast<__m128i&>(*buf)=_mm_set1_epi32(0); }


// 何でも収容するよ型
// 収容されたオブジェクトはデストラクタは呼ばれないので注意
template<size_t Size>
class TVariant
{
public:
    TVariant() { _ZeroClear(m_buf); }

    template<class T>
    TVariant(const T& v)
    {
        operator=<T>(v);
    }

    template<class T, size_t S>
    TVariant(const T (&v)[S])
    {
        operator=<T, S>(v);
    }

    template<class T>
    TVariant& operator=(const T& v)
    {
        istStaticAssert(sizeof(T)<=Size);
        cast<T>() = v;
        return *this;
    }

    template<class T, size_t S>
    TVariant& operator=(const T (&v)[S])
    {
        istStaticAssert(sizeof(v)<=Size);
        stl::copy(v, v+S, reinterpret_cast<T*>(m_buf));
        return *this;
    }

    template<class T>
    T& cast()
    {
        istStaticAssert(sizeof(T)<=Size);
        return *reinterpret_cast<T*>(m_buf);
    }
    template<class T>
    const T& cast() const
    {
        istStaticAssert(sizeof(T)<=Size);
        return *reinterpret_cast<const T*>(m_buf);
    }

    template<class T>
    T& unsafe_cast()
    {
        return *reinterpret_cast<T*>(m_buf);
    }
    template<class T>
    const T& unsafe_cast() const
    {
        return *reinterpret_cast<const T*>(m_buf);
    }

private:
    char m_buf[Size];
};

typedef istAlign(4) TVariant<4>     variant4;
typedef istAlign(8) TVariant<8>     variant8;
typedef istAlign(16) TVariant<16>   variant16;
typedef istAlign(16) TVariant<32>   variant32;
typedef istAlign(16) TVariant<64>   variant64;
typedef istAlign(16) TVariant<128>  variant128;

template<size_t B, size_t A> inline TVariant<B>& variant_cast(TVariant<A> &a) { return (TVariant<B>&)a; }
template<size_t B, size_t A> inline const TVariant<B>& variant_cast(const TVariant<A> &a) { return (const TVariant<B>&)a; }

} // namespace ist


istSerializeRaw(ist::variant4);
istSerializeRaw(ist::variant8);
istSerializeRaw(ist::variant16);
istSerializeRaw(ist::variant32);
istSerializeRaw(ist::variant64);
istSerializeRaw(ist::variant128);

#endif // ist_Base_Variant_h
