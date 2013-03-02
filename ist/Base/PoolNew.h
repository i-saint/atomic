#ifndef ist_Base_PoolNew_h
#define ist_Base_PoolNew_h

#include "ist/Config.h"
#include "ist/stdex/ist_raw_vector.h"
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

class PoolBase;

class istInterModule PoolNewManager
{
public:
    static void freeAll();
    static void addPool(PoolBase *p);

    static size_t getNumPool();
    static PoolBase* getPool(size_t i);

    static void printPoolStates();
};

struct istInterModule PoolSingleThreaded
{
    typedef DummyMutex MutexT;
};

struct istInterModule PoolMultiThreaded
{
    typedef Mutex MutexT;
};


class istInterModule PoolBase
{
istNoncpyable(PoolBase);
public:
    PoolBase(const char *classname, size_t blocksize, size_t align);
    virtual ~PoolBase();
    virtual void freeAll()=0;

    const char* getClassName() const;
    size_t getBlockSize() const;
    size_t getAlign() const;
    size_t getNumBlocks() const;

protected:
    ist::raw_vector<void*>& getPool();

private:
    istMemberPtrDecl_Noncopyable(Members) m;
};

template<class ThreadingPolicy>
class istInterModule TPool : public PoolBase
{
typedef PoolBase super;
public:
    TPool(const char *classname, size_t blocksize, size_t align);
    ~TPool();
    virtual void freeAll();

    void* allocate();
    void recycle(void *p);

private:
    typedef typename ThreadingPolicy::MutexT MutexT;
    istMemberPtrDecl_Noncopyable(Members) m;
};

typedef TPool<PoolSingleThreaded> PoolST;
typedef TPool<PoolMultiThreaded>  PoolMT;

} // namespace ist


#endif // ist_Base_PoolNew_h
