#include "istPCH.h"
#include "New.h"
#include "PoolNew.h"

namespace ist {

static stl::vector<IPool*> s_all_pools;
static ist::Mutex s_mutex;

void PoolNewManager::freeAll()
{
    ist::Mutex::ScopedLock lock(s_mutex);
    for(size_t i=0; i<s_all_pools.size(); ++i) {
        s_all_pools[i]->freeAll();
    }
}

void PoolNewManager::addPool( IPool *p )
{
    ist::Mutex::ScopedLock lock(s_mutex);
    s_all_pools.push_back(p);
}

IPool::IPool()
{
    PoolNewManager::addPool(this);
}

IPool::~IPool()
{
}


template<class ThreadingPolicy>
ist::TPool<ThreadingPolicy>::TPool( const char *classname, size_t blocksize, size_t align )
    : m_classname(classname)
    , m_blocksize(blocksize)
    , m_align(align)
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
