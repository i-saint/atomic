#ifndef __ist_Base_Generics_h__
#define __ist_Base_Generics_h__

namespace ist {

    template<class T> struct UnRef { typedef T type; };
    template<class T> struct UnRef<T&> { typedef T type; };

    template<class T> struct UnConst { typedef T type; };
    template<class T> struct UnConst<const T> { typedef T type; };

    template<class T>
    struct UnRefConst { typedef typename UnConst< typename UnRef<T>::type >::type type; };

    template<class T> struct IteratorToValueType { typedef typename T::value_type type; };
    template<class T> struct IteratorToValueType<T*> { typedef T type; };

} // namespace ist

#endif // __ist_Base_Generics_h__
