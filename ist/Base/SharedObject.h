#ifndef ist_Base_SharedObject_h_
#define ist_Base_SharedObject_h_

#include "ist/Base/Types.h"
#include "ist/Base/New.h"
#include "ist/Base/Assert.h"
#include "ist/Concurrency/Atomic.h"

namespace ist {

    class istInterModule SharedObject
    {
    public:
        SharedObject() : m_ref_counter(0) {}
        virtual ~SharedObject()     {}
        virtual void addRef()       { ++m_ref_counter; }
        virtual void setRef(int32 v){ m_ref_counter=v; }
        int32 getRef() const        { return m_ref_counter; }

        virtual void release()
        {
            int32 ref = --m_ref_counter;
            istAssert(ref>=0, "");
            if(ref==0) { onZeroRef(); }
        }

    protected:
        virtual void onZeroRef() { istDelete(this); }

    private:
        atomic_int32 m_ref_counter;
    };

    inline void intrusive_ptr_add_ref( SharedObject *obj ) { obj->addRef(); }
    inline void intrusive_ptr_release( SharedObject *obj ) { obj->release(); }

} // namespace ist

#define istSafeRelease(Obj)             if(Obj){Obj->release();Obj=NULL;}
#define istSafeAddRef(Obj)              if(Obj){Obj->addRef();}

#endif // ist_Base_SharedObject_h_
