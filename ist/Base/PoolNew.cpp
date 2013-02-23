#include "istPCH.h"
#include "New.h"
#include "PoolNew.h"

namespace ist {

static stl::vector<PoolBase*> s_all_pools;
static ist::Mutex s_mutex;

void PoolNewManager::freeAll()
{
    ist::Mutex::ScopedLock lock(s_mutex);
    for(size_t i=0; i<s_all_pools.size(); ++i) {
        s_all_pools[i]->freeAll();
    }
}

void PoolNewManager::addPool( PoolBase *p )
{
    ist::Mutex::ScopedLock lock(s_mutex);
    s_all_pools.push_back(p);
}

size_t PoolNewManager::getNumPool()
{
    return s_all_pools.size();
}

PoolBase* PoolNewManager::getPool( size_t i )
{
    return s_all_pools[i];
}

void PoolNewManager::printPoolStates()
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



PoolBase::PoolBase( const char *classname, size_t blocksize, size_t align )
    : m_classname(classname)
    , m_blocksize(blocksize)
    , m_align(align)
{
    PoolNewManager::addPool(this);
}

PoolBase::~PoolBase()
{
}


template<class ThreadingPolicy>
ist::TPool<ThreadingPolicy>::TPool( const char *classname, size_t blocksize, size_t align )
    : super(classname, blocksize, align)
{
}

template<class ThreadingPolicy>
ist::TPool<ThreadingPolicy>::~TPool()
{
    freeAll();
}

template<class ThreadingPolicy>
void ist::TPool<ThreadingPolicy>::freeAll()
{
    MutexT::ScopedLock lock(m_mutex);
    for(size_t i=0; i<m_pool.size(); ++i) {
        istAlignedFree(m_pool[i]);
    }
    m_pool.clear();
}

template<class ThreadingPolicy>
void* ist::TPool<ThreadingPolicy>::allocate()
{
    MutexT::ScopedLock lock(m_mutex);
    if(!m_pool.empty()) {
        void *ret = m_pool.back();
        m_pool.pop_back();
        return ret;
    }
    else {
        return istAlignedMalloc(m_blocksize, m_align);
    }
}

template<class ThreadingPolicy>
void ist::TPool<ThreadingPolicy>::recycle( void *p )
{
    MutexT::ScopedLock lock(m_mutex);
    m_pool.push_back(p);
}

template TPool<PoolSingleThreaded>;
template TPool<PoolMultiThreaded>;

} // namespace ist
