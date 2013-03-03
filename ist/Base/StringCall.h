#include "Stringnize.h"

namespace ist {


#define RemoveCR(T) typename std::remove_const<typename std::remove_reference<T>::type>::type

template<class Ch, class R>
struct SC_Fn0
{
    typedef R (*F)();
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    bool operator()(F f, Str *r)
    {
        if(r) { Stringnize(f(), *r); }
        else  { f(); }
        return true;
    }
};

template<class Ch, class R, class C>
struct SC_MemFn0
{
    typedef R (C::*F)();
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    bool operator()(F f, C &o, Str *r)
    {
        if(r) { Stringnize((o.*f)(), *r); }
        else  { (o.*f)(); }
        return true;
    }
};

template<class Ch, class R, class C>
struct SC_ConstMemFn0
{
    typedef R (C::*F)() const;
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    bool operator()(F f, const C &o, Str *r)
    {
        if(r) { Stringnize((o.*f)(), *r); }
        else  { (o.*f)(); }
        return true;
    }
};


template<class Ch, class R, class A0>
struct SC_Fn1
{
    typedef R (*F)(A0);
    typedef RemoveCR(A0) A0T;
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    bool operator()(F f, Str *r, const Ch *a0)
    {
        A0T v0;
        if(!Parse(a0, v0)) { return false; }
        if(r) { Stringnize(f(v0), *r); }
        else  { f(); }
        return true;
    }
};

template<class Ch, class R, class C, class A0>
struct SC_MemFn1
{
    typedef R (C::*F)(A0);
    typedef RemoveCR(A0) A0T;
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    void operator()(F f, C &o, Str &r, const Ch *a0)
    {
        A0T v0;
        if(!Parse(a0, v0)) { return false; }
        if(r) { Stringnize((o.*f)(v0), *r); }
        else  { (o.*f)(); }
        return true;
    }
};

template<class Ch, class R, class C, class A0>
struct SC_ConstMemFn1
{
    typedef R (C::*F)(A0) const;
    typedef RemoveCR(A0) A0T;
    typedef std::basic_string<Ch, std::char_traits<Ch>, std::allocator<Ch> > Str;
    void operator()(F f, const C &o, Str &r, const Ch *a0)
    {
        A0T v0;
        if(!Parse(a0, v0)) { return false; }
        if(r) { Stringnize((o.*f)(v0), *r); }
        else  { (o.*f)(v0); }
        return true;
    }
};

#undef RemoveCR

} // namespace ist
