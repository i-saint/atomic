#include "istPCH.h"
#include "New.h"
#include "PoolNew.h"

namespace ist {
namespace {

struct PMMembers
{
    stl::vector<PoolBase*> all_pools;
    ist::Mutex mutex;
};
PMMembers *g_pmmem;

PMMembers* PmGet()
{
    if(!g_pmmem) {
        g_pmmem = istNew(PMMembers)();
    }
    return g_pmmem;
}

void PmRelease()
{
    istSafeDelete(g_pmmem);
}

} // namespace


void PoolManager::update()
{
    PMMembers &m = *PmGet();
    ist::Mutex::ScopedLock lock(m.mutex);
    for(size_t i=0; i<m.all_pools.size(); ++i) {
        m.all_pools[i]->update();
    }
}

void PoolManager::clear()
{
    {
        PMMembers &m = *PmGet();
        ist::Mutex::ScopedLock lock(m.mutex);
        for(size_t i=0; i<m.all_pools.size(); ++i) {
            istSafeDelete(m.all_pools[i]);
        }
    }
    PmRelease();
}

void PoolManager::addPool( PoolBase *p )
{
    PMMembers &m = *PmGet();
    ist::Mutex::ScopedLock lock(m.mutex);
    m.all_pools.push_back(p);
}

size_t PoolManager::getNumPool()
{
    PMMembers &m = *PmGet();
    return m.all_pools.size();
}

PoolBase* PoolManager::getPool( size_t i )
{
    PMMembers &m = *PmGet();
    return m.all_pools[i];
}

void PoolManager::printPoolStates()
{
    PMMembers &m = *PmGet();
    ist::Mutex::ScopedLock lock(m.mutex);
    char buf[512];
    for(size_t i=0; i<m.all_pools.size(); ++i) {
        const PoolBase &pool = *m.all_pools[i];
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
