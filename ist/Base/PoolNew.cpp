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

struct PoolBase::Members
{
    const char *classname;
    size_t blocksize;
    size_t align;
    ist::raw_vector<void*> pool;
};
istMemberPtrImpl_Noncopyable(PoolBase,Members);

const char* PoolBase::getClassName() const{ return m->classname; }
size_t PoolBase::getBlockSize() const     { return m->blocksize; }
size_t PoolBase::getAlign() const         { return m->align; }
size_t PoolBase::getNumBlocks() const     { return m->pool.size(); }
ist::raw_vector<void*>& PoolBase::getPool() { return m->pool; }

PoolBase::PoolBase( const char *classname, size_t blocksize, size_t align )
{
    m->classname = classname;
    m->blocksize = blocksize;
    m->align = align;
    PoolNewManager::addPool(this);
}

PoolBase::~PoolBase()
{
}

template<class ThreadingPolicy>
struct TPool<ThreadingPolicy>::Members
{
    MutexT mutex;
};
istMemberPtrImpl_Noncopyable(TPool<PoolSingleThreaded>, Members);
istMemberPtrImpl_Noncopyable(TPool<PoolMultiThreaded>, Members);

template<class ThreadingPolicy>
TPool<ThreadingPolicy>::TPool( const char *classname, size_t blocksize, size_t align )
    : super(classname, blocksize, align)
{
}

template<class ThreadingPolicy>
TPool<ThreadingPolicy>::~TPool()
{
    freeAll();
}

template<class ThreadingPolicy>
void TPool<ThreadingPolicy>::freeAll()
{
    MutexT::ScopedLock lock(m->mutex);
    ist::raw_vector<void*> &pool = getPool();
    for(size_t i=0; i<pool.size(); ++i) {
        istAlignedFree(pool[i]);
    }
    pool.clear();
}

template<class ThreadingPolicy>
void* TPool<ThreadingPolicy>::allocate()
{
    MutexT::ScopedLock lock(m->mutex);
    ist::raw_vector<void*> &pool = getPool();
    if(!pool.empty()) {
        void *ret = pool.back();
        pool.pop_back();
        return ret;
    }
    else {
        return istAlignedMalloc(getBlockSize(), getAlign());
    }
}

template<class ThreadingPolicy>
void TPool<ThreadingPolicy>::recycle( void *p )
{
    MutexT::ScopedLock lock(m->mutex);
    getPool().push_back(p);
}

template TPool<PoolSingleThreaded>;
template TPool<PoolMultiThreaded>;

} // namespace ist
