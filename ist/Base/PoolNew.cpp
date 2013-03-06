#include "istPCH.h"
#include "New.h"
#include "PoolNew.h"

namespace ist {

static stl::vector<PoolBase*> s_all_pools;
static ist::Mutex s_mutex;

void PoolManager::clear()
{
    ist::Mutex::ScopedLock lock(s_mutex);
    for(size_t i=0; i<s_all_pools.size(); ++i) {
        s_all_pools[i]->clear();
    }
}

void PoolManager::addPool( PoolBase *p )
{
    ist::Mutex::ScopedLock lock(s_mutex);
    s_all_pools.push_back(p);
}

size_t PoolManager::getNumPool()
{
    return s_all_pools.size();
}

PoolBase* PoolManager::getPool( size_t i )
{
    return s_all_pools[i];
}

void PoolManager::printPoolStates()
{
    ist::Mutex::ScopedLock lock(s_mutex);
    char buf[512];
    for(size_t i=0; i<s_all_pools.size(); ++i) {
        const PoolBase &pool = *s_all_pools[i];
        istSPrintf(buf,
            "pool %s\n"
            "  block size: %d\n"
            "  align: %d\n"
            "  num blocks: %d\n"
            , pool.getClassName(), pool.getBlockSize(), pool.getAlign(), pool.getNumBlocks());
        istPrint(buf);
    }
}


const char* PoolBase::getClassName() const{ return m_classname; }
size_t PoolBase::getBlockSize() const     { return m_blocksize; }
size_t PoolBase::getAlign() const         { return m_align; }

PoolBase::PoolBase( const char *classname, size_t blocksize, size_t align )
{
    m_classname = classname;
    m_blocksize = blocksize;
    m_align = align;
    PoolManager::addPool(this);
}

PoolBase::~PoolBase()
{
}

template<class ThreadingPolicy>
TPoolAllocator<ThreadingPolicy>::TPoolAllocator( const char *classname, size_t blocksize, size_t align )
    : super(classname, blocksize, align)
{
}

template<class ThreadingPolicy>
TPoolAllocator<ThreadingPolicy>::~TPoolAllocator()
{
    clear();
}

template<class ThreadingPolicy>
void ist::TPoolAllocator<ThreadingPolicy>::reserve( size_t size )
{
    MutexT::ScopedLock lock(m_mutex);
    while(m_pool.size()<size) {
        m_pool.push_back(istAlignedMalloc(getBlockSize(), getAlign()));
    }
}

template<class ThreadingPolicy>
void TPoolAllocator<ThreadingPolicy>::clear()
{
    MutexT::ScopedLock lock(m_mutex);
    for(size_t i=0; i<m_pool.size(); ++i) {
        istAlignedFree(m_pool[i]);
    }
    m_pool.clear();
}

template<class ThreadingPolicy>
void* TPoolAllocator<ThreadingPolicy>::allocate()
{
    MutexT::ScopedLock lock(m_mutex);
    if(!m_pool.empty()) {
        void *ret = m_pool.back();
        m_pool.pop_back();
        return ret;
    }
    else {
        return istAlignedMalloc(getBlockSize(), getAlign());
    }
}

template<class ThreadingPolicy>
void TPoolAllocator<ThreadingPolicy>::recycle( void *p )
{
    MutexT::ScopedLock lock(m_mutex);
    m_pool.push_back(p);
}

template TPoolAllocator<PoolSingleThreaded>;
template TPoolAllocator<PoolMultiThreaded>;


} // namespace ist
