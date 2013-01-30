#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/Concurrency/Mutex.h"

#define istDefinePoolNew(PoolT, ClassName)\
public:\
    static void* operator new(size_t /*size*/)  { return getPool().allocate(); }\
    static void operator delete(void *p)        { getPool().recycle(p); }\
    static void* operator new[](size_t /*size*/){ istAssert(false); return NULL; }\
    static void operator delete[](void *p)      { istAssert(false); }\
private:\
    static PoolT& getPool()\
    {\
        static PoolT s_pool(#ClassName, sizeof(ClassName), istAlignof(ClassName));\
        return s_pool;\
    }\

#define istDefinePoolNewST(ClassName) istDefinePoolNew(ist::PoolST, ClassName)
#define istDefinePoolNewMT(ClassName) istDefinePoolNew(ist::PoolMT, ClassName)


namespace ist {

class IPool;

class istInterModule PoolNewManager
{
public:
    static void freeAll();
    static void addPool(IPool *p);
};

struct istInterModule PoolSingleThreaded
{
    typedef DummyMutex MutexT;
};

struct istInterModule PoolMultiThreaded
{
    typedef Mutex MutexT;
};


class istInterModule IPool
{
public:
    IPool();
    virtual ~IPool();
    virtual void freeAll()=0;
};

template<class ThreadingPolicy>
class istInterModule TPool : public IPool
{
public:
    typedef typename ThreadingPolicy::MutexT MutexT;

    TPool(const char *classname, size_t blocksize, size_t align);
    ~TPool();
    virtual void freeAll();

    void* allocate();
    void recycle(void *p);

private:
    const char *m_classname;
    MutexT m_mutex;
    size_t m_blocksize;
    size_t m_align;
    stl::vector<void*> m_pool;
};

typedef TPool<PoolSingleThreaded> PoolST;
typedef TPool<PoolMultiThreaded>  PoolMT;

} // namespace ist


#endif // ist_Base_PoolNew_h
