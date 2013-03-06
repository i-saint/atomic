#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/stdex/ist_raw_vector.h"
#include "ist/Concurrency/Mutex.h"

#define istDefinePoolNew(Class, Deleter, Threading)\
public:\
    typedef ist::TPoolAllocator<Deleter, Threading> PoolT;\
    static void* operator new(size_t /*size*/)  { return getPool().allocate(); }\
    static void  operator delete(void *p)       { getPool().recycle(p); }\
    static void* operator new[](size_t /*size*/){ istAssert(false); return NULL; }\
    static void  operator delete[](void *p)     { istAssert(false); }\
private:\
    static PoolT& getPool()\
    {\
        static PoolT s_pool(#Class, sizeof(Class), istAlignof(Class));\
        return s_pool;\
    }

#define istDeclPoolNew(Class, Deleter, Threading)\
public:\
    typedef ist::TPoolAllocator<Deleter, Threading> PoolT;\
    static void* operator new(size_t /*size*/);\
    static void  operator delete(void *p);\
    static void* operator new[](size_t /*size*/);\
    static void  operator delete[](void *p);\
private:\
    static PoolT& getPool();

#define istImplPoolNew(Class, Deleter, Threading)\
void* Class::operator new(size_t /*size*/)  { return getPool().allocate(); }\
void  Class::operator delete(void *p)       { getPool().recycle(p); }\
void* Class::operator new[](size_t /*size*/){ istAssert(false); return NULL; }\
void  Class::operator delete[](void *p)     { istAssert(false); }\
Class::PoolT& Class::getPool()\
{\
    static PoolT s_pool(#Class, sizeof(Class), istAlignof(Class));\
    return s_pool;\
}

#define istDefinePoolNewSTD(Class, Del) istDefinePoolNew(Class, Del, ist::PoolSingleThreaded)
#define istDefinePoolNewMTD(Class, Del) istDefinePoolNew(Class, Del, ist::PoolMultiThreaded )
#define istDeclPoolNewSTD(Class, Del)   istDeclPoolNew(  Class, Del, ist::PoolSingleThreaded)
#define istDeclPoolNewMTD(Class, Del)   istDeclPoolNew(  Class, Del, ist::PoolMultiThreaded )
#define istImplPoolNewSTD(Class, Del)   istImplPoolNew(  Class, Del, ist::PoolSingleThreaded)
#define istImplPoolNewMTD(Class, Del)   istImplPoolNew(  Class, Del, ist::PoolMultiThreaded )
#define istDefinePoolNewST(Class)       istDefinePoolNew(Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istDefinePoolNewMT(Class)       istDefinePoolNew(Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )
#define istDeclPoolNewST(Class)         istDeclPoolNew(  Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istDeclPoolNewMT(Class)         istDeclPoolNew(  Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )
#define istImplPoolNewST(Class)         istImplPoolNew(  Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istImplPoolNewMT(Class)         istImplPoolNew(  Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )


#define istDefinePoolFactory(Class, Deleter, Threading)\
    template<class C, class T, class D> friend class ist::TPoolFactory;\
    typedef ist::TPoolFactory<Class, Deleter, Threading> PoolT;\
private:\
    static PoolT& getPool()\
    {\
        static PoolT s_pool(#Class);\
        return s_pool;\
    }

#define istDeclPoolFactory(Class, Deleter, Threading)\
    template<class C, class T, class D> friend class ist::TPoolFactory;\
    typedef ist::TPoolFactory<Class, Deleter, Threading> PoolT;\
private:\
    static PoolT& getPool();

#define istImplPoolFactory(Class, Deleter, Threading)\
    Class::PoolT& Class::getPool()\
    {\
        static PoolT s_pool(#Class);\
        return s_pool;\
    }

#define istDefinePoolFactorySTD(Class, Del) istDefinePoolFactory(Class, Del, ist::PoolSingleThreaded)
#define istDefinePoolFactoryMTD(Class, Del) istDefinePoolFactory(Class, Del, ist::PoolMultiThreaded )
#define istDeclPoolFactorySTD(Class, Del)   istDeclPoolFactory(  Class, Del, ist::PoolSingleThreaded)
#define istDeclPoolFactoryMTD(Class, Del)   istDeclPoolFactory(  Class, Del, ist::PoolMultiThreaded )
#define istImplPoolFactorySTD(Class, Del)   istImplPoolFactory(  Class, Del, ist::PoolSingleThreaded)
#define istImplPoolFactoryMTD(Class, Del)   istImplPoolFactory(  Class, Del, ist::PoolMultiThreaded )
#define istDefinePoolFactoryST(Class)       istDefinePoolFactory(Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istDefinePoolFactoryMT(Class)       istDefinePoolFactory(Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )
#define istDeclPoolFactoryST(Class)         istDeclPoolFactory(  Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istDeclPoolFactoryMT(Class)         istDeclPoolFactory(  Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )
#define istImplPoolFactoryST(Class)         istImplPoolFactory(  Class, ist::Deleter_DoNothing, ist::PoolSingleThreaded)
#define istImplPoolFactoryMT(Class)         istImplPoolFactory(  Class, ist::Deleter_DoNothing, ist::PoolMultiThreaded )



namespace ist {

class PoolBase;

class istInterModule PoolManager
{
public:
    static void update();
    static void clear();
    static void addPool(PoolBase *p);

    static size_t getNumPool();
    static PoolBase* getPool(size_t i);

    static void printPoolStates();
};


class PoolBase
{
istNoncpyable(PoolBase);
public:
    PoolBase(const char *Class, size_t blocksize, size_t align);
    virtual ~PoolBase();
    virtual void update()=0;
    virtual void reserve(size_t size)=0;
    virtual void clear()=0;

    const char* getClassName() const;
    size_t getBlockSize() const;
    size_t getAlign() const;
    virtual size_t getNumBlocks() const=0;

private:
    const char *m_classname;
    size_t m_blocksize;
    size_t m_align;
};



struct PoolSingleThreaded { typedef DummyMutex MutexT; };
struct PoolMultiThreaded { typedef Mutex MutexT; };

struct Deleter_DoNothing
{
    template<class Obj, class Deleter>
    void operator()(ist::raw_vector< std::pair<int, Obj*> > &pool, Deleter del) const
    {
    }
};

template<int N>
struct Deleter_Delayed
{
    template<class T>
    struct second_is_null
    {
        bool operator()(const T &v) const { return v.second==NULL; }
    };

    template<class Obj, class Deleter>
    void operator()(ist::raw_vector<std::pair<int, Obj*>> &pool, Deleter del) const
    {
        size_t deleted = 0;
        size_t size = pool.size();
        for(size_t i=0; i<size; ++i) {
            if(pool[i].first++==N) {
                del(pool[i].second);
                pool[i].second = NULL;
                ++deleted;
            }
        }
        if(deleted>0) {
            pool.erase(std::remove_if(pool.begin(), pool.end(), second_is_null<std::pair<int, Obj*>>()), pool.end());
        }
    }
};



template<class Deleter, class ThreadingPolicy>
class TPoolAllocator : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename ThreadingPolicy::MutexT MutexT;
    typedef std::pair<int, void*> Pair;
    typedef ist::raw_vector<Pair> Pairs;

    TPoolAllocator(const char *Class, size_t blocksize, size_t align)
        : super(Class, blocksize, align)
    {
    }
    ~TPoolAllocator()
    {
        clear();
    }

    static void Free(void *p) { istAlignedFree(p); }

    virtual void update()
    {
        Deleter()(m_pool, &TPoolAllocator::Free);
    }

    virtual size_t getNumBlocks() const
    {
        return m_pool.size();
    }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(Pair(0, istAlignedMalloc(getBlockSize(), getAlign())));
        }
    }

    virtual void clear()
    {
        MutexT::ScopedLock lock(m_mutex);
        for(size_t i=0; i<m_pool.size(); ++i) {
            istAlignedFree(m_pool[i].second);
        }
        m_pool.clear();
    }


    void* allocate()
    {
        MutexT::ScopedLock lock(m_mutex);
        if(!m_pool.empty()) {
            void *ret = m_pool.back().second;
            m_pool.pop_back();
            return ret;
        }
        else {
            return istAlignedMalloc(getBlockSize(), getAlign());
        }
    }

    void recycle( void *p )
    {
        MutexT::ScopedLock lock(m_mutex);
        m_pool.push_back(Pair(0, p));
    }

private:
    Pairs m_pool;
    MutexT m_mutex;
};


template<class ObjType, class Deleter, class ThreadingPolicy>
class TPoolFactory : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename ThreadingPolicy::MutexT MutexT;
    typedef std::pair<int, ObjType*> Pair;
    typedef ist::raw_vector<Pair> Pairs;

    TPoolFactory(const char *Class)
        : super(Class, sizeof(ObjType), istAlignof(ObjType))
    {
    }

    ~TPoolFactory()
    {
        clear();
    }

    static void Delete(ObjType *p) { istDelete(p); }

    virtual void update()
    {
        Deleter()(m_pool, &TPoolFactory::Delete);
    }

    virtual size_t getNumBlocks() const { return m_pool.size(); }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(Pair(0, istNew(ObjType)()));
        }
    }

    virtual void clear()
    {
        MutexT::ScopedLock lock(m_mutex);
        for(size_t i=0; i<m_pool.size(); ++i) {
            istDelete(m_pool[i].second);
        }
        m_pool.clear();
    }

    ObjType* create()
    {
        MutexT::ScopedLock lock(m_mutex);
        if(!m_pool.empty()) {
            ObjType *ret = m_pool.back().second;
            m_pool.pop_back();
            return ret;
        }
        else {
            return istNew(ObjType)();
        }
    }

    void recycle(ObjType *p)
    {
        MutexT::ScopedLock lock(m_mutex);
        m_pool.push_back(Pair(0, p));
    }

private:
    Pairs m_pool;
    MutexT m_mutex;
};


} // namespace ist


#endif // ist_Base_PoolNew_h
