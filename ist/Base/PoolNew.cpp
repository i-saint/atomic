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
char istAlign(8) g_pmmem_mem[sizeof(PMMembers)];
PMMembers *g_pmmem;

PMMembers* PmGet()
{
    if(!g_pmmem) {
        g_pmmem = istPlacementNew(PMMembers, g_pmmem_mem)();
    }
    return g_pmmem;
}

void PmRelease()
{
    g_pmmem->~PMMembers();
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

void PoolManager::release()
{
    {
        PMMembers &m = *PmGet();
        ist::Mutex::ScopedLock lock(m.mutex);
        for(size_t i=0; i<m.all_pools.size(); ++i) {
            m.all_pools[i]->release();
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

Mutex& PoolManager::getMutex()
{
    PMMembers &m = *PmGet();
    return m.mutex;
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
            "  num blocks: %d\n"
            , pool.getClassName(), pool.getNumBlocks());
        istPrint(buf);
    }
}


const char* PoolBase::getClassName() const{ return m_classname; }

PoolBase::PoolBase( const char *classname )
{
    m_classname = classname;
    PoolManager::addPool(this);
}

PoolBase::~PoolBase()
{
    m_classname = NULL;
}

void PoolBase::release()
{
    if(m_classname!=NULL) {
        this->~PoolBase();
    }
}


} // namespace ist
