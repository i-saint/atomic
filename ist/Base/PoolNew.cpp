#include "istPCH.h"
#include "New.h"
#include "PoolNew.h"

namespace ist {

static stl::vector<PoolBase*> s_all_pools;
static ist::Mutex s_mutex;

void PoolManager::update()
{
    ist::Mutex::ScopedLock lock(s_mutex);
    for(size_t i=0; i<s_all_pools.size(); ++i) {
        s_all_pools[i]->update();
    }
}

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


} // namespace ist
