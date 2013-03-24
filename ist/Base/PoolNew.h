#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/stdex/ist_raw_vector.h"
#include "ist/Concurrency/Mutex.h"


#define istPoolUpdate()     ist::PoolManager::update()
#define istPoolRelease()    ist::PoolManager::release()


#define istImplPoolFunction(PoolType, FuncName, ...)\
    PoolType& FuncName()\
    {\
        static char s_mem[sizeof(PoolType)];\
        static PoolType *s_pool;\
        if(s_pool==NULL) {\
            ist::Mutex::ScopedLock lock(ist::PoolManager::getMutex());\
            if(s_pool==NULL) {\
                s_pool=istPlacementNew(PoolType, s_mem)(__VA_ARGS__);\
            }\
        }\
        return *s_pool;\
    }


#define istDefinePoolNew(Class, Traits)\
public:\
    typedef ist::TPoolAllocator<Traits> PoolT;\
    static void* operator new(size_t, size_t)       { return getPool().allocate(); }\
    static void  operator delete(void *p, size_t)   { getPool().recycle(p); }\
    static void* operator new[](size_t, size_t)     { istAssert(false); return NULL; }\
    static void  operator delete[](void *p, size_t) { istAssert(false); }\
private:\
    static istImplPoolFunction(PoolT, getPool, #Class, sizeof(Class), istAlignof(Class))

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
    static istImplPoolFunction(Class::PoolT, Class::getPool, #Class, sizeof(Class), istAlignof(Class))

#define istDefinePoolNewST(Class)       istDefinePoolNew(Class, ist::PoolTraitsST<Class>)
#define istDefinePoolNewMT(Class)       istDefinePoolNew(Class, ist::PoolTraitsMT<Class>)
#define istDeclPoolNewST(Class)         istDeclPoolNew(  Class, ist::PoolTraitsST<Class>)
#define istDeclPoolNewMT(Class)         istDeclPoolNew(  Class, ist::PoolTraitsMT<Class>)
#define istImplPoolNewST(Class)         istImplPoolNew(  Class, ist::PoolTraitsST<Class>)
#define istImplPoolNewMT(Class)         istImplPoolNew(  Class, ist::PoolTraitsMT<Class>)


#define istDefinePoolFactory(Class, Traits)\
    template<class T> friend class ist::PoolCreator;\
    typedef ist::TPoolFactory<Class, Traits> PoolT;\
private:\
    static istImplPoolFunction(PoolT, getPoolm #Class)

#define istDeclPoolFactory(Class, Traits)\
    template<class T> friend class ist::PoolCreator;\
    typedef ist::TPoolFactory<Class, Traits> PoolT;\
private:\
    static PoolT& getPool();

#define istImplPoolFactory(Class, Traits)\
    istImplPoolFunction(Class::PoolT, Class::getPool, #Class)

#define istDefinePoolFactoryST(Class)   istDefinePoolFactory(Class, ist::PoolTraitsST<Class>)
#define istDefinePoolFactoryMT(Class)   istDefinePoolFactory(Class, ist::PoolTraitsMT<Class>)
#define istDeclPoolFactoryST(Class)     istDeclPoolFactory(  Class, ist::PoolTraitsST<Class>)
#define istDeclPoolFactoryMT(Class)     istDeclPoolFactory(  Class, ist::PoolTraitsMT<Class>)
#define istImplPoolFactoryST(Class)     istImplPoolFactory(  Class, ist::PoolTraitsST<Class>)
#define istImplPoolFactoryMT(Class)     istImplPoolFactory(  Class, ist::PoolTraitsMT<Class>)



namespace ist {

class PoolBase;

class istInterModule PoolManager
{
istNoncpyable(PoolManager);
public:
    static void update();
    static void release();
    static void addPool(PoolBase *p);

    static size_t getNumPool();
    static PoolBase* getPool(size_t i);
    static Mutex& getMutex();

    static void printPoolStates();
};


class istInterModule PoolBase
{
istNoncpyable(PoolBase);
public:
    PoolBase(const char *classname);
    virtual ~PoolBase();
    virtual void release();
    virtual void update()=0;
    virtual void reserve(size_t size)=0;
    virtual void clear()=0;
    virtual size_t getNumBlocks() const=0;

    const char* getClassName() const;

private:
    const char *m_classname;
};



struct PoolSingleThreaded { typedef DummyMutex MutexT; };
struct PoolMultiThreaded { typedef SpinMutex MutexT; };

struct PoolUpdater_DoNothing
{
    template<class ObjT, class CreatorT, class MutexT>
    void operator()(ist::raw_vector<std::pair<int, ObjT*>> &, CreatorT , MutexT &) const
    {
    }
};

template<int Delay>
struct PoolUpdater_DelayedDeleter
{
    template<class T>
    struct second_is_null
    {
        bool operator()(const T &v) const { return v.second==NULL; }
    };

    template<class ObjT, class CreatorT, class MutexT>
    void operator()(ist::raw_vector<std::pair<int, ObjT*>> &pool, CreatorT c, MutexT &mutex) const
    {
        Mutex::ScopedLock lock(mutex);
        size_t deleted = 0;
        size_t size = pool.size();
        for(size_t i=0; i<size; ++i) {
            auto &p = pool[i];
            if(p.first++==Delay) {
                c.release(p.second);
                p.second = NULL;
                ++deleted;
            }
        }
        if(deleted>0) {
            pool.erase(
                std::remove_if(pool.begin(), pool.end(), second_is_null<std::pair<int, ObjT*>>()),
                pool.end());
        }
    }
};

struct PoolAllocator
{
    void* allocate(size_t s, size_t a) const { return istAlignedMalloc(s, a); }
    void  release(void *p) const             { return istAlignedFree(p); }
};

struct PoolCreator
{
    template<class T> T*   create() const      { return istNew(T)(); }
    template<class T> void release(T *p) const { istDelete(p); }
};



// 適切な Threading, Updater, Creator の typedef を用意することでカスタム可能。
// 以下は汎用 traits

template<class T>
struct PoolTraitsST
{
    typedef PoolSingleThreaded      Threading;
    typedef PoolUpdater_DoNothing   Updater;
    typedef PoolAllocator           Allocator;
    typedef PoolCreator             Creator;
};

template<class T>
struct PoolTraitsMT
{
    typedef PoolMultiThreaded       Threading;
    typedef PoolUpdater_DoNothing   Updater;
    typedef PoolAllocator           Allocator;
    typedef PoolCreator             Creator;
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
        : super(Class)
        , m_blocksize(blocksize)
        , m_align(align)
    {
    }
    ~TPoolAllocator()
    {
        clear();
    }

    size_t getBlockSize() const { return m_blocksize; }
    size_t getAlign() const     { return m_align; }

    virtual void update()
    {
        Updater()(m_pool, Allocator(), m_mutex);
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
    size_t m_blocksize;
    size_t m_align;
};




template<class T, class Traits=PoolTraitsST<T> >
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
        : super(classname)
    {
    }

    ~TPoolFactory()
    {
        clear();
    }

    virtual void update()
    {
        Updater()(m_pool, Creator(), m_mutex);
    }

    virtual size_t getNumBlocks() const { return m_pool.size(); }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(Pair(0, Creator().create<T>()));
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
        return Creator().create<T>();
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
