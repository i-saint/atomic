#ifndef __ist_Base_Variant__
#define __ist_Base_Variant__
namespace ist {


// 何でも収容するよ型
// 収容されたオブジェクトはデストラクタは呼ばれないので注意
template<size_t Size>
class TVariant
{
private:
    char m_buf[Size];

public:
    TVariant() {}

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
        BOOST_STATIC_ASSERT(sizeof(T)<=Size);
        cast<T>() = v;
        return *this;
    }

    template<class T, size_t S>
    TVariant& operator=(const T (&v)[S])
    {
        BOOST_STATIC_ASSERT(sizeof(v)<=Size);
        stl::copy(v, v+S, reinterpret_cast<T*>(m_buf));
        return *this;
    }

    template<class T>
    T& cast()
    {
        BOOST_STATIC_ASSERT(sizeof(T)<=Size);
        return *reinterpret_cast<T*>(m_buf);
    }

    template<class T>
    const T& cast() const
    {
        BOOST_STATIC_ASSERT(sizeof(T)<=Size);
        return *reinterpret_cast<const T*>(m_buf);
    }
};

typedef TVariant<8>  Variant8;
typedef TVariant<16> Variant16;
typedef TVariant<32> Variant32;
typedef TVariant<64> Variant64;

template<size_t B, size_t A> inline TVariant<B>& variant_cast(TVariant<A> &a) { return (TVariant<B>&)a; }
template<size_t B, size_t A> inline const TVariant<B>& variant_cast(const TVariant<A> &a) { return (const TVariant<B>&)a; }

} // namespace ist
#endif // __ist_Base_Variant__
