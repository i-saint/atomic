#ifndef ist_Debug_CommandlineCommand_h
#define ist_Debug_CommandlineCommand_h
#include "ist/Base/Types.h"
#include "ist/Base/New.h"
#include "ist/Base/Variant.h"

namespace ist {


class ICLCommand
{
public:
    virtual ~ICLCommand() {}
    virtual void release() { istDelete(this); }
    virtual uint32 getNumArgs() const=0;
    virtual void setArg(uint32 i, const char *arg)=0;
    virtual void clearArgs()=0;
    virtual void exec()=0;
};


bool CLParseArg(const char *str, bool &v);
bool CLParseArg(const char *str, int8 &v);
bool CLParseArg(const char *str, int16 &v);
bool CLParseArg(const char *str, int32 &v);
bool CLParseArg(const char *str, uint8 &v);
bool CLParseArg(const char *str, uint16 &v);
bool CLParseArg(const char *str, uint32 &v);
bool CLParseArg(const char *str, float32 &v);
bool CLParseArg(const char *str, vec2 &v);
bool CLParseArg(const char *str, vec3 &v);
bool CLParseArg(const char *str, vec4 &v);
bool CLParseArg(const char *str, ivec2 &v);
bool CLParseArg(const char *str, ivec3 &v);
bool CLParseArg(const char *str, ivec4 &v);
bool CLParseArg(const char *str, variant16 &v);


template<class V>
class CLCValue : public ICLCommand
{
public:
    CLCValue(V *v) : m_v(v) {}
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        V a = V();
        if(m_v && (!m_args[0] || CLParseArg(m_args[0], a))) {
            m_v = a;
        }
    }
private:
    V *m_v;
    const char *m_args[1];
};


template<class R>
class CLCFunction0 : public ICLCommand
{
public:
    typedef R (*Func)();

    CLCFunction0(Func f) : m_f(f) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void exec()
    {
        if(m_f) {
            m_f();
        }
    }
private:
    Func m_f;
};

template<class R, class C>
class CLCMemFn0 : public ICLCommand
{
public:
    typedef R (C::*Func)();

    CLCMemFn0(Func f, C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void exec()
    {
        if( m_f && m_obj )
        {
            (m_obj->*m_f)();
        }
    }
private:
    Func m_f;
    C *m_obj;
};

template<class R, class C>
class CLCConstMemFn0 : public ICLCommand
{
public:
    typedef R (C::*Func)() const;

    CLCConstMemFn0(Func f, const C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void exec()
    {
        if( m_f && m_obj )
        {
            (m_obj->*m_f)();
        }
    }
private:
    Func m_f;
    const C *m_obj;
};

template<class R, class A0>
class CLCFunction1 : public ICLCommand
{
public:
    typedef R (*Func)(A0);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLCFunction1(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            m_f(a0);
        }
    }
private:
    Func m_f;
    const char *m_args[1];
};

template<class R, class C, class A0>
class CLCMemFn1 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLCMemFn1(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[1];
};

template<class R, class C, class A0>
class CLCConstMemFn1 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLCConstMemFn1(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[1];
};

template<class R, class A0, class A1>
class CLCFunction2 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLCFunction2(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            )
        {
            m_f(a0, a1);
        }
    }
private:
    Func m_f;
    const char *m_args[2];
};

template<class R, class C, class A0, class A1>
class CLCMemFn2 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLCMemFn2(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            )
        {
            (m_obj->*m_f)(a0, a1);
        }
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[2];
};

template<class R, class C, class A0, class A1>
class CLCConstMemFn2 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLCConstMemFn2(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            )
        {
            (m_obj->*m_f)(a0, a1);
        }
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[2];
};

template<class R, class A0, class A1, class A2>
class CLCFunction3 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1, A2);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLCFunction3(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            )
        {
            m_f(a0, a1, a2);
        }
    }
private:
    Func m_f;
    const char *m_args[3];
};

template<class R, class C, class A0, class A1, class A2>
class CLCMemFn3 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLCMemFn3(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2);
        }
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[3];
};

template<class R, class C, class A0, class A1, class A2>
class CLCConstMemFn3 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLCConstMemFn3(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2);
        }
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[3];
};

template<class R, class A0, class A1, class A2, class A3>
class CLCFunction4 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1, A2, A3);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLCFunction4(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            && ( !m_args[3] || CLParseArg(m_args[3], a3) )
            )
        {
            m_f(a0, a1, a2, a3);
        }
    }
private:
    Func m_f;
    const char *m_args[4];
};

template<class R, class C, class A0, class A1, class A2, class A3>
class CLCMemFn4 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2, A3);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLCMemFn4(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
                        && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            && ( !m_args[3] || CLParseArg(m_args[3], a3) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2, a3);
        }
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[4];
};

template<class R, class C, class A0, class A1, class A2, class A3>
class CLCConstMemFn4 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2, A3) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLCConstMemFn4(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            && ( !m_args[1] || CLParseArg(m_args[1], a1) )
            && ( !m_args[2] || CLParseArg(m_args[2], a2) )
            && ( !m_args[3] || CLParseArg(m_args[3], a3) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2, a3);
        }
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[4];
};



template<class V>
CLCValue<V>* CreateCommand(V *v)
{ return istNew(istTypeJoin(CLCValue<V>))(v); }


template<class R>
CLCFunction0<R>* CreateCommand(R (*f)())
{ return istNew(istTypeJoin(CLCFunction0<R>))(f); }

template<class R, class C>
CLCMemFn0<R, C>* CreateCommand(R (C::*f)(), C *obj)
{ return istNew(istTypeJoin(CLCMemFn0<R, C>))(f, obj); }

template<class R, class C>
CLCConstMemFn0<R, C>* CreateCommand(R (C::*f)() const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn0<R, C>))(f, obj); }


template<class R, class A0>
CLCFunction1<R, A0>* CreateCLCommand(R (*f)(A0))
{ return istNew(istTypeJoin(CLCFunction1<R, A0>))(f); }

template<class R, class C, class A0>
CLCMemFn1<R, C, A0>* CreateCLCommand(R (C::*f)(A0), C *obj)
{ return istNew(istTypeJoin(CLCMemFn1<R, C, A0>))(f, obj); }

template<class R, class C, class A0>
CLCConstMemFn1<R, C, A0>* CreateCLCommand(R (C::*f)(A0) const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn1<R, C, A0>))(f, obj); }

template<class R, class A0, class A1>
CLCFunction2<R, A0, A1>* CreateCLCommand(R (*f)(A0, A1))
{ return istNew(istTypeJoin(CLCFunction2<R, A0, A1>))(f); }

template<class R, class C, class A0, class A1>
CLCMemFn2<R, C, A0, A1>* CreateCLCommand(R (C::*f)(A0, A1), C *obj)
{ return istNew(istTypeJoin(CLCMemFn2<R, C, A0, A1>))(f, obj); }

template<class R, class C, class A0, class A1>
CLCConstMemFn2<R, C, A0, A1>* CreateCLCommand(R (C::*f)(A0, A1) const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn2<R, C, A0, A1>))(f, obj); }

template<class R, class A0, class A1, class A2>
CLCFunction3<R, A0, A1, A2>* CreateCLCommand(R (*f)(A0, A1, A2))
{ return istNew(istTypeJoin(CLCFunction3<R, A0, A1, A2>))(f); }

template<class R, class C, class A0, class A1, class A2>
CLCMemFn3<R, C, A0, A1, A2>* CreateCLCommand(R (C::*f)(A0, A1, A2), C *obj)
{ return istNew(istTypeJoin(CLCMemFn3<R, C, A0, A1, A2>))(f, obj); }

template<class R, class C, class A0, class A1, class A2>
CLCConstMemFn3<R, C, A0, A1, A2>* CreateCLCommand(R (C::*f)(A0, A1, A2) const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn3<R, C, A0, A1, A2>))(f, obj); }

template<class R, class A0, class A1, class A2, class A3>
CLCFunction4<R, A0, A1, A2, A3>* CreateCLCommand(R (*f)(A0, A1, A2, A3))
{ return istNew(istTypeJoin(CLCFunction4<R, A0, A1, A2, A3>))(f); }

template<class R, class C, class A0, class A1, class A2, class A3>
CLCMemFn4<R, C, A0, A1, A2, A3>* CreateCLCommand(R (C::*f)(A0, A1, A2, A3), C *obj)
{ return istNew(istTypeJoin(CLCMemFn4<R, C, A0, A1, A2, A3>))(f, obj); }

template<class R, class C, class A0, class A1, class A2, class A3>
CLCConstMemFn4<R, C, A0, A1, A2, A3>* CreateCLCommand(R (C::*f)(A0, A1, A2, A3) const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn4<R, C, A0, A1, A2, A3>))(f, obj); }


} // namespace ist
#endif // ist_Debug_Commandline_h
