#ifndef __ist_Base_SharedObject_h_
#define __ist_Base_SharedObject_h_

#include "Types.h"
#include "ThreadUtil.h"

namespace ist {

    class SharedObject
    {
    public:
        virtual ~SharedObject() {}
        void incrementReferenceCount() { ++m_ref_counter; }
        void decrementReferenceCount() { if(--m_ref_counter==0) { delete this; } }

    private:
        atomic_int32 m_ref_counter;
    };

    void intrusive_ptr_add_ref( SharedObject *obj ) { obj->incrementReferenceCount(); }
    void intrusive_ptr_release( SharedObject *obj ) { obj->decrementReferenceCount(); }

} // namespace ist

#endif // __ist_Base_SharedObject_h_
