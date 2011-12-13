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
        template<class T>
        TVariant(const T& v)
        {
            BOOST_STATIC_ASSERT(sizeof(T)<=Size);
            new(m_buf) T(v);
        }

        template<class T>
        T& operator=(const T& v)
        {
            BOOST_STATIC_ASSERT(sizeof(T)<=Size);
            new(m_buf) T(v);
        }

        template<class T>
        T& cast()
        {
            BOOST_STATIC_ASSERT(sizeof(T)<=Size);
            return *static_cast<T*>(m_buf);
        }

        template<class T>
        const T& cast() const
        {
            BOOST_STATIC_ASSERT(sizeof(T)<=Size);
            return *static_cast<const T*>(m_buf);
        }
    };

    typedef TVariant<8>  Variant8;
    typedef TVariant<16> Variant16;
    typedef TVariant<32> Variant32;
    typedef TVariant<64> Variant64;

} // namespace ist
#endif // __ist_Base_Variant__
