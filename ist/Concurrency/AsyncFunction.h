#ifndef ist_Concurrency_AsyncFunction_h
#define ist_Concurrency_AsyncFunction_h

#include <functional>
#include "ist/Base/Generics.h"
#include "ist/Concurrency/TaskScheduler.h"

namespace ist {

// std::function<result_t, ()>、もしくはこの互換オブジェクト (result_t が typedef されている functor) を非同期実行
template<class Func, class ResultT=typename Func::result_type>
class AsyncFunction : public Task
{
public:
    AsyncFunction() : m_result() {}
    AsyncFunction(const Func &f) : m_func(f), m_result() {}
    ~AsyncFunction() { wait(); }

    void setFunction(const Func &f) { m_func=f; }
    void start() { TaskScheduler::getInstance()->enqueue(this); }
    Func& getFunction() { return m_func; }
    ResultT getResult() { wait(); return m_result; }

    virtual void exec() { m_result=m_func(); }

private:
    Func m_func;
    ResultT m_result;
};

// result_t==void 用の特殊化
template<class Func>
class AsyncFunction<Func, void> : public Task
{
public:
    AsyncFunction() {}
    AsyncFunction(const Func &f) : m_func(f) {}
    ~AsyncFunction() { wait(); }

    void setFunction(const Func &f) { m_func=f; }
    void start() { TaskScheduler::getInstance()->enqueue(this); }
    Func& getFunction() { return m_func; }

    virtual void exec() { m_func(); }

private:
    Func m_func;
};


// std::function は引数だけ差し替えて再利用ができないので、
// std::function の生成すら惜しい時用の軽量汎用関数オブジェクトを用意


template<class Ret=void, class Arg1=void, class Arg2=void, class Arg3=void>
class Function;

template<class Class, class Ret=void, class Arg1=void, class Arg2=void, class Arg3=void>
class Method;

template<class Class, class Ret=void, class Arg1=void, class Arg2=void, class Arg3=void>
class ConstMethod;

template<class Arg>
struct ArgHolder
{
    ArgHolder() {}
    ArgHolder(Arg v) : m_value(v) {}
    operator Arg() const { return m_value; }
    Arg m_value;
};
template<class Arg>
struct ArgHolder<const Arg>
{
    ArgHolder() {}
    ArgHolder(const Arg v) : m_value(v) {}
    operator Arg() const { return m_value; }
    Arg m_value;
};
template<class Arg>
struct ArgHolder<Arg&>
{
    ArgHolder() {}
    ArgHolder(Arg &v) : m_value(&v) {}
    operator Arg() const { return *m_value; }
    Arg *m_value;
};
template<class Arg>
struct ArgHolder<const Arg&>
{
    ArgHolder() {}
    ArgHolder(const Arg &v) : m_value(&v) {}
    operator Arg() const { return *m_value; }
    const Arg *m_value;
};


template<class Res>
class Function<Res>
{
public:
    typedef Res result_type;
    typedef Res (*Func)();
    Function() : m_func(NULL) {}
    Function(Func f) : m_func(f) {}
    Res operator()() const { return m_func(); }
private:
    Func m_func;
};
template<class Res, class Arg1>
class Function<Res, Arg1>
{
public:
    typedef Res result_type;
    typedef Res (*Func)(Arg1);
    typedef ArgHolder<Arg1> Arg1H;
    Function() : m_func(NULL){}
    Function(Func f, Arg1 a1) : m_func(f), m_arg1(a1) {}
    Res operator()() const { return m_func(m_arg1); }
private:
    Func m_func;
    Arg1H m_arg1;
};


template<class Class, class Res>
class Method<Class, Res>
{
public:
    typedef Res result_type;
    typedef Res (Class::*Func)();
    Method() : m_func(NULL), m_inst(NULL) {}
    Method(Func f, Class &i) : m_func(f), m_inst(&i) {}
    Res operator()() const { return (m_inst->*m_func)(); }
private:
    Func m_func;
    Class *m_inst;
};
template<class Class, class Res, class Arg1>
class Method<Class, Res, Arg1>
{
public:
    typedef Res result_type;
    typedef Res (Class::*Func)(Arg1);
    typedef ArgHolder<Arg1> Arg1H;
    Method() : m_func(NULL), m_inst(NULL) {}
    Method(Func f, Class &i, Arg1 a1) : m_func(f), m_inst(&i), m_arg1(a1) {}
    Res operator()() const { return (m_inst->*m_func)(m_arg1); }
private:
    Func m_func;
    Class *m_inst;
    Arg1H m_arg1;
};


template<class Class, class Res>
class ConstMethod<Class, Res>
{
public:
    typedef Res result_type;
    typedef Res (Class::*Func)() const;
    ConstMethod() : m_func(NULL), m_inst(NULL) {}
    ConstMethod(Func f, Class &i) : m_func(f), m_inst(&i) {}
    Res operator()() { return (m_inst->*m_func)(); }
private:
    Func m_func;
    Class *m_inst;
};
template<class Class, class Res, class Arg1>
class ConstMethod<Class, Res, Arg1>
{
public:
    typedef Res result_type;
    typedef Res (Class::*Func)(Arg1) const;
    typedef ArgHolder<Arg1> Arg1H;
    ConstMethod() : m_func(NULL), m_inst(NULL) {}
    ConstMethod(Func f, Class &i, Arg1 a1) : m_func(f), m_inst(&i), m_arg1(a1) {}
    Res operator()() { return (m_inst->*m_func)(m_arg1); }
private:
    Func m_func;
    Class *m_inst;
    Arg1H m_arg1;
};

} // namespace ist

#endif // ist_Concurrency_AsyncFunction_h
