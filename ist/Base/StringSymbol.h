#ifndef ist_Base_StringSymbol_h
#define ist_Base_StringSymbol_h

namespace ist {

stl::wstring L(const stl::string &mbs);
stl::wstring L(const char *mbs, size_t len);
stl::string S(const stl::wstring &wcs);
stl::string S(const wchar_t *wcs, size_t len);


class istInterModule StringSymbolPool
{
public:
    typedef int32 id_t;
    typedef stl::map<stl::string, id_t> StringTable;

    StringSymbolPool();
    id_t genID(const stl::string &str);

private:
    istMemberPtrDecl(Members) m;
};


// 文字列から一意の整数を生成し、それを保持。比較を高速に行う class
template<class T>
class istInterModule TStringSymbol
{
public:
    typedef StringSymbolPool::id_t id_t;
    TStringSymbol() : m_id(0) {}
    TStringSymbol(const stl::string &str) : m_id(getPool()->genID(str)) {}
    id_t getID() const { return m_id; }
    bool operator==(const TStringSymbol &o) { return m_id==o.m_id; }

private:
    id_t m_id;

private:
    static StringSymbolPool* getPool();
};

#define istStringSymbolImpl(T)\
    template<> StringSymbolPool* TStringSymbol<T>::getPool() {\
        static StringSymbolPool s_pool;\
        return &s_pool;\
    }


} // namespace ist

#endif // ist_Base_StringSymbol_h
