#ifndef ist_Base_StringSymbol_h
#define ist_Base_StringSymbol_h

namespace ist {

class istInterModule StringSymbolPool
{
public:
    typedef int32 id_t;
    typedef stl::map<stl::string, id_t> StringTable;

    StringSymbolPool() : m_idgen(0) {}
    id_t genID(const stl::string &str)
    {
        StringTable::iterator i = m_table.find(str);
        if(i==m_table.end()) {
            id_t id = ++m_idgen;
            m_table[str] = id;
            return id;
        }
        return i->second;
    }

private:
    StringTable m_table;
    id_t m_idgen;
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
    static StringSymbolPool* getPool()
    {
        static StringSymbolPool s_pool;
        return &s_pool;
    }
};

} // namespace ist

#endif // ist_Base_StringSymbol_h
