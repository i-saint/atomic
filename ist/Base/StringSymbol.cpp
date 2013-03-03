#include "istPCH.h"
#include "StringSymbol.h"

namespace ist {

struct StringSymbolPool::Members
{
    StringTable table;
    id_t idgen;

    Members() : idgen(0) {}
};


StringSymbolPool::StringSymbolPool() {}
StringSymbolPool::id_t StringSymbolPool::genID(const stl::string &str)
{
    StringTable::iterator i = m->table.find(str);
    if(i==m->table.end()) {
        id_t id = ++m->idgen;
        m->table[str] = id;
        return id;
    }
    return i->second;
}

} // namespace ist

