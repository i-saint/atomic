#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/stdex/ist_raw_vector.h"
#include "ist/Concurrency/Mutex.h"


#define istPoolUpdate() ist::PoolManager::update()
#define istPoolClear()  ist::PoolManager::clear()


#define istDefinePoolNew(Class, Traits)\
public:\
    typedef ist::TPoolAllocator<Traits> PoolT;\
    static void* operator new(size_t, size_t)       { return getPool().allocate(); }\
    static void  operator delete(void *p, size_t)   { getPool().recycle(p); }\
    static void* operator new[](size_t, size_t)     { istAssert(false); return NULL; }\
    static void  operator delete[](void *p, size_t) { istAssert(false); }\
private:\
    static PoolT& getPool()\
    {\
        static PoolT *s_pool = istNew(PoolT)(#Class, sizeof(Class), istAlignof(Class));\
        return *s_pool;\
    }

#define istDeclPoolNew(Class, Traits)\
public:\
    typedef ist::TPoolAllocator<Traits> PoolT;\
    static void* operator new(size_t, size_t);\
    static void  operator delete(void *p, size_t);\
    static void* operator new[](size_t, size_t);\
    static void  operator delete[](void *p, size_t);\
private:\
    static PoolT& getPool();

#define istImplPoolNew(Class, Traits)\
void* Class::operator new(size_t, size_t)       { return getPool().allocate(); }\
void  Class::operator delete(void *p, size_t)   { getPool().recycle(p); }\
void* Class::operator new[](size_t, size_t)     { istAssert(false); return NULL; }\
void  Class::operator delete[](void *p, size_t) { istAssert(false); }\
Class::PoolT& Class::getPool()\
{\
    static PoolT *s_pool = istNew(PoolT)(#Class, sizeof(Class), istAlignof(Class));\
    return *s_pool;\
}

#define istDefinePoolNewST(Class)       istDefinePoolNew(Class, ist::PoolAllocatorTraitsST<Class>)
#define istDefinePoolNewMT(Class)       istDefinePoolNew(Class, ist::PoolAllocatorTraitsMT<Class>)
#define istDeclPoolNewST(Class)         istDeclPoolNew(  Class, ist::PoolAllocatorTraitsST<Class>)
#define istDeclPoolNewMT(Class)         istDeclPoolNew(  Class, ist::PoolAllocatorTraitsMT<Class>)
#define istImplPoolNewST(Class)         istImplPoolNew(  Class, ist::PoolAllocatorTraitsST<Class>)
#define istImplPoolNewMT(Class)         istImplPoolNew(  Class, ist::PoolAllocatorTraitsMT<Class>)


#define istDefinePoolFactory(Class, Traits)\
    template<class T> friend class ist::PoolCreator;\
    typedef ist::TPoolFactory<Class, Traits> PoolT;\
private:\
    static PoolT& getPool()\
    {\
        static PoolT *s_pool = istNew(PoolT)(#Class);\
        return *s_pool;\
    }

#define istDeclPoolFactory(Class, Traits)\
    template<class T> friend class ist::PoolCreator;\
    typedef ist::TPoolFactory<Class, Traits> PoolT;\
private:\
    static PoolT& getPool();

#define istImplPoolFactory(Class, Traits)\
    Class::PoolT& Class::getPool()\
    {\
        static PoolT *s_pool = istNew(PoolT)(#Class);\
        return *s_pool;\
    }

#define istDefinePoolFactoryST(Class)   istDefinePoolFactory(Class, ist::PoolFactoryTraitsST<Class>)
#define istDefinePoolFactoryMT(Class)   istDefinePoolFactory(Class, ist::PoolFactoryTraitsMT<Class>)
#define istDeclPoolFactoryST(Class)     istDeclPoolFactory(  Class, ist::PoolFactoryTraitsST<Class>)
#define istDeclPoolFactoryMT(Class)     istDeclPoolFactory(  Class, ist::PoolFactoryTraitsMT<Class>)
#define istImplPoolFactoryST(Class)     istImplPoolFactory(  Class, ist::PoolFactoryTraitsST<Class>)
#define istImplPoolFactoryMT(Class)     istImplPoolFactory(  Class, ist::PoolFactoryTraitsMT<Class>)



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
struct PoolMultiThreaded { typedef SpinMutex MutexT; };

struct PoolUpdater_DoNothing
{
    template<class T, class C>
    void operator()(ist::raw_vector< std::pair<int, T*> > &, C ) const
    {
    }
};

template<int N>
struct PoolUpdater_DelayedDeleter
{
    template<class T>
    struct second_is_null
    {
        bool operator()(const T &v) const { return v.second==NULL; }
    };

    template<class T, class C>
    void operator()(ist::raw_vector<std::pair<int, T*>> &pool, C c) const
    {
        size_t deleted = 0;
        size_t size = pool.size();
        for(size_t i=0; i<size; ++i) {
            auto &p = pool[i];
            if(p.first++==N) {
                c.release(p.second);
                p.second = NULL;
                ++deleted;
            }
        }
        if(deleted>0) {
            pool.erase(
                std::remove_if(pool.begin(), pool.end(), second_is_null<std::pair<int, T*>>()),
                pool.end());
        }
    }
};

struct PoolAllocator
{
    void* allocate(size_t s, size_t a) const { return istAlignedMalloc(s, a); }
    void  release(void *p) const             { return istAlignedFree(p); }
};

template<class T>
struct PoolCreator
{
    T*   create() const      { return istNew(T)(); }
    void release(T *p) const { istDelete(p); }
};



template<class T>
struct PoolAllocatorTraitsST
{
    typedef PoolSingleThreaded    Threading;
    typedef PoolUpdater_DoNothing Updater;
    typedef PoolAllocator         Allocator;
};

template<class T>
struct PoolAllocatorTraitsMT
{
    typedef PoolMultiThreaded     Threading;
    typedef PoolUpdater_DoNothing Updater;
    typedef PoolAllocator         Allocator;
};



template<class Traits>
class TPoolAllocator : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename Traits::Threading::MutexT MutexT;
    typedef typename Traits::Updater Updater;
    typedef typename Traits::Allocator Allocator;
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

    static void Free(void *p) { Allocator().release(p); }

    virtual void update()
    {
        MutexT::ScopedLock lock(m_mutex);
        Updater()(m_pool, Allocator());
    }

    virtual size_t getNumBlocks() const
    {
        return m_pool.size();
    }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(Pair(0, Allocator().allocate(getBlockSize(), getAlign())));
        }
    }

    virtual void clear()
    {
        MutexT::ScopedLock lock(m_mutex);
        for(size_t i=0; i<m_pool.size(); ++i) {
            Allocator().release(m_pool[i].second);
        }
        m_pool.clear();
    }


    void* allocate()
    {
        {
            MutexT::ScopedLock lock(m_mutex);
            if(!m_pool.empty()) {
                void *ret = m_pool.back().second;
                m_pool.pop_back();
                return ret;
            }
        }
        return Allocator().allocate(getBlockSize(), getAlign());
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



// 適切な Threading, Updater, Creator の typedef を用意することでカスタム可能。
// 以下は汎用 traits

template<class T>
struct PoolFactoryTraitsST
{
    typedef PoolSingleThreaded    Threading;
    typedef PoolUpdater_DoNothing Updater;
    typedef PoolCreator<T>        Creator;
};

template<class T>
struct PoolFactoryTraitsMT
{
    typedef PoolMultiThreaded     Threading;
    typedef PoolUpdater_DoNothing Updater;
    typedef PoolCreator<T>        Creator;
};

template<class T, class Traits=PoolFactoryTraitsST<T> >
class TPoolFactory : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename Traits::Threading::MutexT MutexT;
    typedef typename Traits::Updater Updater;
    typedef typename Traits::Creator Creator;
    typedef std::pair<int, T*> Pair;
    typedef ist::raw_vector<Pair> Pairs;

    TPoolFactory(const char *classname)
        : super(classname, sizeof(T), istAlignof(T))
    {
    }

    ~TPoolFactory()
    {
        clear();
    }

    virtual void update()
    {
        MutexT::ScopedLock lock(m_mutex);
        Updater()(m_pool, Creator());
    }

    virtual size_t getNumBlocks() const { return m_pool.size(); }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(Pair(0, Creator().create()));
        }
    }

    virtual void clear()
    {
        MutexT::ScopedLock lock(m_mutex);
        for(size_t i=0; i<m_pool.size(); ++i) {
            Creator().release(m_pool[i].second);
        }
        m_pool.clear();
    }

    T* create()
    {
        {
            MutexT::ScopedLock lock(m_mutex);
            if(!m_pool.empty()) {
                T *ret = m_pool.back().second;
                m_pool.pop_back();
                return ret;
            }
        }
        return Creator().create();
    }

    void recycle(T *p)
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
