#ifndef ist_Base_MemberPtr_h
#define ist_Base_MemberPtr_h

#include "ist/Config.h"

#define istNoncpyable(T)\
    private:\
        T(const T &other);\
        T& operator=(const T &other);


#define istMemberPtrDecl(M)\
    struct M;\
    class istInterModule member_ptr\
    {\
    public:\
        member_ptr();\
        ~member_ptr();\
        member_ptr(const member_ptr &other);\
        member_ptr& operator=(const member_ptr &other);\
        M&       operator*();\
        const M& operator*() const;\
        M*       operator->();\
        const M* operator->() const;\
    private:\
        M * istRestrict ptr;\
    };\
    friend class member_ptr;\
    member_ptr

#define istMemberPtrImpl(C,M)\
    istMemberPtrImpl_Noncopyable(C,M)\
    C::member_ptr::member_ptr(const member_ptr &other) : ptr(istNew(C::M)(*other.ptr)) {}\
    C::member_ptr& C::member_ptr::operator=(const member_ptr &other) { *ptr=*other.ptr; return *this; }



#define istMemberPtrDecl_Noncopyable(M)\
    struct M;\
    class istInterModule member_ptr\
    {\
    public:\
        member_ptr();\
        ~member_ptr();\
        M&       operator*();\
        const M& operator*() const;\
        M*       operator->();\
        const M* operator->() const;\
    private:\
        member_ptr(const member_ptr &other);\
        member_ptr& operator=(const member_ptr &other);\
        M * istRestrict ptr;\
    };\
    friend class member_ptr;\
    member_ptr

#define istMemberPtrImpl_Noncopyable(C,M)\
    C::member_ptr::member_ptr() : ptr(istNew(C::M)()) {}\
    C::member_ptr::~member_ptr() { istDelete(ptr); }\
    C::M&        C::member_ptr::operator*()        { return *ptr; }\
    const C::M&  C::member_ptr::operator*() const  { return *ptr; }\
    C::M*        C::member_ptr::operator->()       { return ptr; }\
    const C::M*  C::member_ptr::operator->() const { return ptr; }


#endif // ist_Base_MemberPtr_h
