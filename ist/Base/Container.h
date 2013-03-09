#ifndef ist_Base_Container_h
#define ist_Base_Container_h
#include "ist/Base/Allocator.h"

namespace ist {


template<class T>
class BlockedList
{
public:
    BlockedList(size_t block_size=1024, size_t align=istDefaultAlignment, IAllocator *alloc=GetDefaultAllocator())
    {
        m_allocator = istNew(ChainedFixedAllocator)(sizeof(T), block_size, align, alloc);
    }

    ~BlockedList()
    {
        istSafeDelete(m_allocator);
    }

    T* MakeElement()
    {
        return istNewA(T, m_allocator)();
    }

    void EraseElement(T *obj)
    {
        istDeleteA(obj, m_allocator);
    }

private:
    ChainedFixedAllocator *m_allocator;
};


} // namespace ist
#endif // ist_Base_Container_h
