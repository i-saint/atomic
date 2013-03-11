#ifndef ist_stdex_helper_functions_h
#define ist_stdex_helper_functions_h

namespace ist {

template<class T> struct less_deref     { bool operator()(const T &a, const T &b) const { return  a <  b; } };
template<class T> struct less_deref<T*> { bool operator()(const T *a, const T *b) const { return *a < *b; } };

template<class T> struct equal_deref     { bool operator()(const T &a, const T &b) const { return  a ==  b; } };
template<class T> struct equal_deref<T*> { bool operator()(const T *a, const T *b) const { return *a == *b; } };

template<class T> struct less_id     { bool operator()(const T &a, const T &b) const { return  a.getID() <  b.getID(); } };
template<class T> struct less_id<T*> { bool operator()(const T *a, const T *b) const { return a->getID() < b->getID(); } };

} // namespace ist

#endif // ist_stdex_helper_functions_h
