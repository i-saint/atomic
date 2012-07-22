#ifndef __ist_Base_SharedObject_h_
#define __ist_Base_SharedObject_h_

#include "ist/Base/Types.h"
#include "ist/Base/New.h"
#include "ist/Concurrency/Atomic.h"

namespace ist {

    class SharedObject
    {
    public:
        virtual ~SharedObject() {}
        void incrementReferenceCount() { ++m_ref_counter; }
        void decrementReferenceCount() { if(--m_ref_counter==0) { onZeroReference(); } }

    protected:
        virtual void onZeroReference() { istDelete(this); }

    private:
        atomic_int32 m_ref_counter;
    };

    inline void intrusive_ptr_add_ref( SharedObject *obj ) { obj->incrementReferenceCount(); }
    inline void intrusive_ptr_release( SharedObject *obj ) { obj->decrementReferenceCount(); }

} // namespace ist

#endif // __ist_Base_SharedObject_h_
