#ifndef ist_Debug_CommandlineCommand_h
#define ist_Debug_CommandlineCommand_h
#include "ist/Base/Types.h"
#include "ist/Base/New.h"
#include "ist/Base/Stringnize.h"

namespace ist {


class ICLCommand
{
public:
    virtual ~ICLCommand() {}
    virtual void release() { istDelete(this); }
    virtual uint32 getNumArgs() const=0;
    virtual void setArg(uint32 i, const char *arg)=0;
    virtual void clearArgs()=0;
    virtual void stringnizeResult(stl::string &str)=0;
    virtual bool exec()=0;
};

#define RemoveCR(T) typename std::remove_const<typename std::remove_reference<T>::type>::type


template<class V>
class CLCValue : public ICLCommand
{
public:
    CLCValue(V *v) : m_v(v) {}
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        V a = V();
        if(m_v && (!m_args[0] || Parse(m_args[0], a))) {
            m_v = a;
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;

    CLCFunction0(Func f) : m_f(f) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        if(m_f) {
            m_r = m_f();
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    RT m_r;
};
template<>
class CLCFunction0<void> : public ICLCommand
{
public:
    typedef void (*Func)();

    CLCFunction0(Func f) : m_f(f) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        if(m_f) {
            m_f();
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
};

template<class R, class C>
class CLCMemFn0 : public ICLCommand
{
public:
    typedef R (C::*Func)();
    typedef RemoveCR(R) RT;

    CLCMemFn0(Func f, C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        if( m_f && m_obj )
        {
            m_r = (m_obj->*m_f)();
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    C *m_obj;
    RT m_r;
};
template<class C>
class CLCMemFn0<void, C> : public ICLCommand
{
public:
    typedef void (C::*Func)();

    CLCMemFn0(Func f, C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        if( m_f && m_obj )
        {
            (m_obj->*m_f)();
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;

    CLCConstMemFn0(Func f, const C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        if( m_f && m_obj )
        {
            m_r = (m_obj->*m_f)();
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    RT m_r;
};
template<class C>
class CLCConstMemFn0<void, C> : public ICLCommand
{
public:
    typedef void (C::*Func)() const;

    CLCConstMemFn0(Func f, const C *o) : m_f(f), m_obj(o) {}
    virtual uint32 getNumArgs() const { return 0; }
    virtual void setArg(uint32 i, const char *arg) {}
    virtual void clearArgs() {}
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        if( m_f && m_obj )
        {
            (m_obj->*m_f)();
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;

    CLCFunction1(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            m_r = m_f(a0);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    RT m_r;
    const char *m_args[1];
};
template<class A0>
class CLCFunction1<void, A0> : public ICLCommand
{
public:
    typedef void (*Func)(A0);
    typedef RemoveCR(A0) A0T;

    CLCFunction1(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            m_f(a0);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;

    CLCMemFn1(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            m_r = (m_obj->*m_f)(a0);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    C *m_obj;
    RT m_r;
    const char *m_args[1];
};
template<class C, class A0>
class CLCMemFn1<void, C, A0> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0);
    typedef RemoveCR(A0) A0T;

    CLCMemFn1(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;

    CLCConstMemFn1(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            m_r = (m_obj->*m_f)(a0);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    RT m_r;
    const char *m_args[1];
};
template<class C, class A0>
class CLCConstMemFn1<void, C, A0> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0) const;
    typedef RemoveCR(A0) A0T;

    CLCConstMemFn1(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            )
        {
            (m_obj->*m_f)(a0);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCFunction2(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            m_r = m_f(a0, a1);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    RT m_r;
    const char *m_args[2];
};
template<class A0, class A1>
class CLCFunction2<void, A0, A1> : public ICLCommand
{
public:
    typedef void (*Func)(A0, A1);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCFunction2(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            m_f(a0, a1);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCMemFn2(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    C *m_obj;
    RT m_r;
    const char *m_args[2];
};
template<class C, class A0, class A1>
class CLCMemFn2<void, C, A0, A1> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCMemFn2(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            (m_obj->*m_f)(a0, a1);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCConstMemFn2(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    RT m_r;
    const char *m_args[2];
};
template<class C, class A0, class A1>
class CLCConstMemFn2<void, C, A0, A1> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1) const;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;

    CLCConstMemFn2(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            )
        {
            (m_obj->*m_f)(a0, a1);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCFunction3(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            m_r = m_f(a0, a1, a2);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    RT m_r;
    const char *m_args[3];
};
template<class A0, class A1, class A2>
class CLCFunction3<void, A0, A1, A2> : public ICLCommand
{
public:
    typedef void (*Func)(A0, A1, A2);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCFunction3(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            m_f(a0, a1, a2);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCMemFn3(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1, a2);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    C *m_obj;
    RT m_r;
    const char *m_args[3];
};
template<class C, class A0, class A1, class A2>
class CLCMemFn3<void, C, A0, A1, A2> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1, A2);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCMemFn3(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCConstMemFn3(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1, a2);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    RT m_r;
    const char *m_args[3];
};
template<class C, class A0, class A1, class A2>
class CLCConstMemFn3<void, C, A0, A1, A2> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1, A2) const;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;

    CLCConstMemFn3(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCFunction4(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            m_r = m_f(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    RT m_r;
    const char *m_args[4];
};
template<class A0, class A1, class A2, class A3>
class CLCFunction4<void, A0, A1, A2, A3> : public ICLCommand
{
public:
    typedef void (*Func)(A0, A1, A2, A3);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCFunction4(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            m_f(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCMemFn4(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    C *m_obj;
    RT m_r;
    const char *m_args[4];
};
template<class C, class A0, class A1, class A2, class A3>
class CLCMemFn4<void, C, A0, A1, A2, A3> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1, A2, A3);
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCMemFn4(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
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
    typedef RemoveCR(R) RT;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCConstMemFn4(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { Stringnize(m_r, str); }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            m_r = (m_obj->*m_f)(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    RT m_r;
    const char *m_args[4];
};
template<class C, class A0, class A1, class A2, class A3>
class CLCConstMemFn4<void, C, A0, A1, A2, A3> : public ICLCommand
{
public:
    typedef void (C::*Func)(A0, A1, A2, A3) const;
    typedef RemoveCR(A0) A0T;
    typedef RemoveCR(A1) A1T;
    typedef RemoveCR(A2) A2T;
    typedef RemoveCR(A3) A3T;

    CLCConstMemFn4(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void stringnizeResult(stl::string &str) { str="void"; }
    virtual bool exec()
    {
        A0T a0 = A0T();
        A1T a1 = A1T();
        A2T a2 = A2T();
        A3T a3 = A3T();
        if( m_f && m_obj
            && ( !m_args[0] || Parse(m_args[0], a0) )
            && ( !m_args[1] || Parse(m_args[1], a1) )
            && ( !m_args[2] || Parse(m_args[2], a2) )
            && ( !m_args[3] || Parse(m_args[3], a3) )
            )
        {
            (m_obj->*m_f)(a0, a1, a2, a3);
        }
        else { return false; }
        return true;
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[4];
};

#undef RemoveCR


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
