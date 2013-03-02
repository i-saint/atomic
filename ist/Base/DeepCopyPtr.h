#ifndef ist_Base_DeepCopyPtr_h
#define ist_Base_DeepCopyPtr_h

#include "ist/Config.h"

#define istMemberPtrTemplate()\
    template<class T>\
    class istInterModule member_ptr\
    {\
    public:\
        member_ptr();\
        member_ptr(T *p);\
        ~member_ptr();\
        member_ptr(const member_ptr &other);\
        member_ptr& operator=(const member_ptr &other);\
        T&       operator*();\
        const T& operator*() const;\
        T*       operator->();\
        const T* operator->() const;\
    private:\
        T * istRestrict m_ptr;\
    };

#define istMemberPtrDecl(T) template<class T> friend class member_ptr; member_ptr<T>

#define istMemberPtrImpl_Noncopyable(T)\
    template<> member_ptr<T>::member_ptr() : m_ptr(istNew(T)()) {}\
    template<> member_ptr<T>::member_ptr(T *p) : m_ptr(p) {}\
    template<> member_ptr<T>::~member_ptr() { istDelete(m_ptr); }\
    template<> T&       member_ptr<T>::operator*()        { return *m_ptr; }\
    template<> const T& member_ptr<T>::operator*() const  { return *m_ptr; }\
    template<> T*       member_ptr<T>::operator->()       { return m_ptr; }\
    template<> const T* member_ptr<T>::operator->() const { return m_ptr; }

#define istMemberPtrImpl(T)\
    istMemberPtrImpl_Noncopyable(T)\
    template<> member_ptr<T>::member_ptr(const member_ptr &other) : m_ptr(istNew(T)(*other.m_ptr)) {}\
    template<> member_ptr<T>& member_ptr<T>::operator=(const member_ptr &other) { *m_ptr=*other.m_ptr; return *this; }

namespace ist {
    istMemberPtrTemplate();
} // namespace ist

#endif // ist_Base_DeepCopyPtr_h
