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
    virtual void exec()=0;
};


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
bool CLParseArg(const char *str, Variant16 &v);


template<class R>
class CLFunctionCommand0 : public ICLCommand
{
public:
    typedef R (*Func)();

    CLFunctionCommand0(Func f) : m_f(f) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
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
class CLMemFnCommand0 : public ICLCommand
{
public:
    typedef R (C::*Func)();

    CLMemFnCommand0(Func f, C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
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

template<class R, class C>
class CLCMemFnCommand0 : public ICLCommand
{
public:
    typedef R (C::*Func)() const;

    CLCMemFnCommand0(Func f, const C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
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

template<class R, class A0>
class CLFunctionCommand1 : public ICLCommand
{
public:
    typedef R (*Func)(A0);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLFunctionCommand1(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            m_f(a0);
        }
        clearArgs();
    }
private:
    Func m_f;
    const char *m_args[1];
};

template<class R, class C, class A0>
class CLMemFnCommand1 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLMemFnCommand1(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
        clearArgs();
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[1];
};

template<class R, class C, class A0>
class CLCMemFnCommand1 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;

    CLCMemFnCommand1(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || CLParseArg(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
        clearArgs();
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[1];
};

template<class R, class A0, class A1>
class CLFunctionCommand2 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLFunctionCommand2(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const char *m_args[2];
};

template<class R, class C, class A0, class A1>
class CLMemFnCommand2 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLMemFnCommand2(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[2];
};

template<class R, class C, class A0, class A1>
class CLCMemFnCommand2 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;

    CLCMemFnCommand2(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[2];
};

template<class R, class A0, class A1, class A2>
class CLFunctionCommand3 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1, A2);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLFunctionCommand3(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const char *m_args[3];
};

template<class R, class C, class A0, class A1, class A2>
class CLMemFnCommand3 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLMemFnCommand3(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[3];
};

template<class R, class C, class A0, class A1, class A2>
class CLCMemFnCommand3 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;

    CLCMemFnCommand3(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[3];
};

template<class R, class A0, class A1, class A2, class A3>
class CLFunctionCommand4 : public ICLCommand
{
public:
    typedef R (*Func)(A0, A1, A2, A3);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLFunctionCommand4(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const char *m_args[4];
};

template<class R, class C, class A0, class A1, class A2, class A3>
class CLMemFnCommand4 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2, A3);
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLMemFnCommand4(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[4];
};

template<class R, class C, class A0, class A1, class A2, class A3>
class CLCMemFnCommand4 : public ICLCommand
{
public:
    typedef R (C::*Func)(A0, A1, A2, A3) const;
    typedef typename std::remove_const<typename std::remove_reference<A0>::type>::type A0T;
    typedef typename std::remove_const<typename std::remove_reference<A1>::type>::type A1T;
    typedef typename std::remove_const<typename std::remove_reference<A2>::type>::type A2T;
    typedef typename std::remove_const<typename std::remove_reference<A3>::type>::type A3T;

    CLCMemFnCommand4(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
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
        clearArgs();
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[4];
};



template<class R>
ICLCommand* CreateCommand(R (*f)())
{ return istNew(CLFunctionCommand0<R>)(f); }

template<class R, class C>
ICLCommand* CreateCommand(R (C::*f)(), C *obj)
{ return istNew(CLMemFnCommand0<R>)(f, obj); }

template<class R, class C>
ICLCommand* CreateCommand(R (C::*f)() const, C *obj)
{ return istNew(CLCMemFnCommand0<R>)(f, obj); }

template<class R, class A0>
ICLCommand* CreateCLCommand(R (*f)(A0))
{ return istNew(istTypeJoin(CLFunctionCommand1<R, A0>))(f); }

template<class R, class C, class A0>
ICLCommand* CreateCLCommand(R (C::*f)(A0), C *obj)
{ return istNew(istTypeJoin(CLMemFnCommand1<R, C, A0>))(f, obj); }

template<class R, class C, class A0>
ICLCommand* CreateCLCommand(R (C::*f)(A0) const, C *obj)
{ return istNew(istTypeJoin(CLCMemFnCommand1<R, C, A0>))(f, obj); }

template<class R, class A0, class A1>
ICLCommand* CreateCLCommand(R (*f)(A0, A1))
{ return istNew(istTypeJoin(CLFunctionCommand2<R, A0, A1>))(f); }

template<class R, class C, class A0, class A1>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1), C *obj)
{ return istNew(istTypeJoin(CLMemFnCommand2<R, C, A0, A1>))(f, obj); }

template<class R, class C, class A0, class A1>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1) const, C *obj)
{ return istNew(istTypeJoin(CLCMemFnCommand2<R, C, A0, A1>))(f, obj); }

template<class R, class A0, class A1, class A2>
ICLCommand* CreateCLCommand(R (*f)(A0, A1, A2))
{ return istNew(istTypeJoin(CLFunctionCommand3<R, A0, A1, A2>))(f); }

template<class R, class C, class A0, class A1, class A2>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1, A2), C *obj)
{ return istNew(istTypeJoin(CLMemFnCommand3<R, C, A0, A1, A2>))(f, obj); }

template<class R, class C, class A0, class A1, class A2>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1, A2) const, C *obj)
{ return istNew(istTypeJoin(CLCMemFnCommand3<R, C, A0, A1, A2>))(f, obj); }

template<class R, class A0, class A1, class A2, class A3>
ICLCommand* CreateCLCommand(R (*f)(A0, A1, A2, A3))
{ return istNew(istTypeJoin(CLFunctionCommand4<R, A0, A1, A2, A3>))(f); }

template<class R, class C, class A0, class A1, class A2, class A3>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1, A2, A3), C *obj)
{ return istNew(istTypeJoin(CLMemFnCommand4<R, C, A0, A1, A2, A3>))(f, obj); }

template<class R, class C, class A0, class A1, class A2, class A3>
ICLCommand* CreateCLCommand(R (C::*f)(A0, A1, A2, A3) const, C *obj)
{ return istNew(istTypeJoin(CLCMemFnCommand4<R, C, A0, A1, A2, A3>))(f, obj); }


} // namespace ist
#endif // ist_Debug_Commandline_h
