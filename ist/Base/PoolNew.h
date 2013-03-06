#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/stdex/ist_raw_vector.h"
#include "ist/Concurrency/Mutex.h"

#define istDefinePoolNew(ClassName, Threading)\
public:\
    typedef ist::TPoolAllocator<Threading> PoolT;\
    static void* operator new(size_t /*size*/)  { return getPool().allocate(); }\
    static void  operator delete(void *p)       { getPool().recycle(p); }\
    static void* operator new[](size_t /*size*/){ istAssert(false); return NULL; }\
    static void  operator delete[](void *p)     { istAssert(false); }\
private:\
    static PoolT& getPool()\
    {\
        static PoolT s_pool(#ClassName, sizeof(ClassName), istAlignof(ClassName));\
        return s_pool;\
    }

#define istDeclPoolNew(ClassName, Threading)\
public:\
    typedef ist::TPoolAllocator<Threading> PoolT;\
    static void* operator new(size_t /*size*/);\
    static void  operator delete(void *p);\
    static void* operator new[](size_t /*size*/);\
    static void  operator delete[](void *p);\
private:\
    static PoolT& getPool();

#define istImplPoolNew(ClassName, Threading)\
void* ClassName::operator new(size_t /*size*/)  { return getPool().allocate(); }\
void  ClassName::operator delete(void *p)       { getPool().recycle(p); }\
void* ClassName::operator new[](size_t /*size*/){ istAssert(false); return NULL; }\
void  ClassName::operator delete[](void *p)     { istAssert(false); }\
ClassName::PoolT& ClassName::getPool()\
{\
    static PoolT s_pool(#ClassName, sizeof(ClassName), istAlignof(ClassName));\
    return s_pool;\
}

#define istDefinePoolNewST(ClassName)   istDefinePoolNew(ClassName, ist::PoolSingleThreaded)
#define istDefinePoolNewMT(ClassName)   istDefinePoolNew(ClassName, ist::PoolMultiThreaded )
#define istDeclPoolNewST(ClassName)     istDeclPoolNew(ClassName, ist::PoolSingleThreaded)
#define istDeclPoolNewMT(ClassName)     istDeclPoolNew(ClassName, ist::PoolMultiThreaded )
#define istImplPoolNewST(ClassName)     istImplPoolNew(ClassName, ist::PoolSingleThreaded)
#define istImplPoolNewMT(ClassName)     istImplPoolNew(ClassName, ist::PoolMultiThreaded )


#define istDefinePoolFactory( ClassName, Threading)\
    template<class C, class T> friend class ist::TPoolFactory;\
    typedef ist::TPoolFactory<ClassName, Threading> PoolT;\
private:\
    static PoolT& getPool()\
    {\
        static PoolT s_pool(#ClassName);\
        return s_pool;\
    }

#define istDeclPoolFactory(ClassName, Threading)\
    template<class C, class T> friend class ist::TPoolFactory;\
    typedef ist::TPoolFactory<ClassName, Threading> PoolT;\
private:\
    static PoolT& getPool();

#define istImplPoolFactory(ClassName, Threading)\
    ClassName::PoolT& ClassName::getPool()\
    {\
        static PoolT s_pool(#ClassName);\
        return s_pool;\
    }

#define istDefinePoolFactoryST(ClassName)   istDefinePoolFactory(ClassName, ist::PoolSingleThreaded)
#define istDefinePoolFactoryMT(ClassName)   istDefinePoolFactory(ClassName, ist::PoolMultiThreaded )
#define istDeclPoolFactoryST(ClassName)     istDeclPoolFactory(ClassName, ist::PoolSingleThreaded)
#define istDeclPoolFactoryMT(ClassName)     istDeclPoolFactory(ClassName, ist::PoolMultiThreaded )
#define istImplPoolFactoryST(ClassName)     istImplPoolFactory(ClassName, ist::PoolSingleThreaded)
#define istImplPoolFactoryMT(ClassName)     istImplPoolFactory(ClassName, ist::PoolMultiThreaded )



namespace ist {

class PoolBase;

class istInterModule PoolManager
{
public:
    static void clear();
    static void addPool(PoolBase *p);

    static size_t getNumPool();
    static PoolBase* getPool(size_t i);

    static void printPoolStates();
};

struct PoolSingleThreaded
{
    typedef DummyMutex MutexT;
};

struct PoolMultiThreaded
{
    typedef Mutex MutexT;
};


class PoolBase
{
istNoncpyable(PoolBase);
public:
    PoolBase(const char *classname, size_t blocksize, size_t align);
    virtual ~PoolBase();
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

template<class ThreadingPolicy>
class TPoolAllocator : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename ThreadingPolicy::MutexT MutexT;

    TPoolAllocator(const char *classname, size_t blocksize, size_t align);
    ~TPoolAllocator();
    virtual size_t getNumBlocks() const { return m_pool.size(); }
    virtual void reserve(size_t size);
    virtual void clear();

    void* allocate(); // pool があればそれを、なければ新規にメモリ確保して返す
    void recycle(void *p); // 使わなくなったメモリを pool に戻す

private:
    ist::raw_vector<void*> m_pool;
    MutexT m_mutex;
};

typedef TPoolAllocator<PoolSingleThreaded> PoolAllocatorST;
typedef TPoolAllocator<PoolMultiThreaded>  PoolAllocatorMT;



template<class ObjType, class ThreadingPolicy>
class TPoolFactory : public PoolBase
{
typedef PoolBase super;
public:
    typedef typename ThreadingPolicy::MutexT MutexT;

    TPoolFactory(const char *classname)
        : super(classname, sizeof(ObjType), istAlignof(ObjType))
    {
    }

    ~TPoolFactory()
    {
        clear();
    }

    virtual size_t getNumBlocks() const { return m_pool.size(); }

    virtual void reserve(size_t size)
    {
        MutexT::ScopedLock lock(m_mutex);
        while(m_pool.size()<size) {
            m_pool.push_back(istNew(ObjType)());
        }
    }

    virtual void clear()
    {
        MutexT::ScopedLock lock(m_mutex);
        for(size_t i=0; i<m_pool.size(); ++i) {
            istDelete(m_pool[i]);
        }
        m_pool.clear();
    }

    ObjType* create()
    {
        MutexT::ScopedLock lock(m_mutex);
        if(!m_pool.empty()) {
            ObjType *ret = m_pool.back();
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
        m_pool.push_back(p);
    }

private:
    ist::raw_vector<ObjType*> m_pool;
    MutexT m_mutex;
};


} // namespace ist


#endif // ist_Base_PoolNew_h
