#!/usr/local/bin/ruby


def gen_command(num_args)
    class_args = []
    args = []
    arg_typedefs = []
    tmp_arg_decls = []
    tmp_args = []
    arg_parse = []

    (0...num_args).each do |i|
        class_args << "class A#{i}"
        args << "A#{i}"
        arg_typedefs << "    typedef typename std::remove_const<typename std::remove_reference<A#{i}>::type>::type A#{i}T;"
        tmp_arg_decls << "        A#{i}T a#{i} = A#{i}T();"
        tmp_args << "a#{i}"
        arg_parse << "            && ( !m_args[#{i}] || CLParseArg(m_args[#{i}], a#{i}) )"
    end
    puts <<END
template<class R, #{class_args.join(", ")}>
class CLFunctionCommand#{num_args} : public ICLCommand
{
public:
    typedef R (*Func)(#{args.join(", ")});
#{arg_typedefs.join("\n")}

    CLFunctionCommand#{num_args}(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f
            #{arg_parse.join("\n")}
            )
        {
            m_f(#{tmp_args.join(", ")});
        }
        clearArgs();
    }
private:
    Func m_f;
    const char *m_args[#{num_args}];
};

template<class R, class C, #{class_args.join(", ")}>
class CLMemFnCommand#{num_args} : public ICLCommand
{
public:
    typedef R (C::*Func)(#{args.join(", ")});
#{arg_typedefs.join("\n")}

    CLMemFnCommand#{num_args}(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f && m_obj
            #{arg_parse.join("\n")}
            )
        {
            (m_obj->*m_f)(#{tmp_args.join(", ")});
        }
        clearArgs();
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[#{num_args}];
};

template<class R, class C, #{class_args.join(", ")}>
class CLCMemFnCommand#{num_args} : public ICLCommand
{
public:
    typedef R (C::*Func)(#{args.join(", ")}) const;
#{arg_typedefs.join("\n")}

    CLCMemFnCommand#{num_args}(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f && m_obj
#{arg_parse.join("\n")}
            )
        {
            (m_obj->*m_f)(#{tmp_args.join(", ")});
        }
        clearArgs();
    }
private:
    Func m_f;
    const C *m_obj;
    const char *m_args[#{num_args}];
};

END
end

def gen_creator(num_args)
    class_args = []
    args = []
    (0...num_args).each do |i|
        class_args << "class A#{i}"
        args << "A#{i}"
    end
    puts <<END
template<class R, #{class_args.join(", ")}>
ICLCommand* CreateCLCommand(R (*f)(#{args.join(", ")}))
{ return istNew(istTypeJoin(CLFunctionCommand#{num_args}<R, #{args.join(", ")}>))(f); }

template<class R, class C, #{class_args.join(", ")}>
ICLCommand* CreateCLCommand(R (C::*f)(#{args.join(", ")}), C *obj)
{ return istNew(istTypeJoin(CLMemFnCommand#{num_args}<R, C, #{args.join(", ")}>))(f, obj); }

template<class R, class C, #{class_args.join(", ")}>
ICLCommand* CreateCLCommand(R (C::*f)(#{args.join(", ")}) const, C *obj)
{ return istNew(istTypeJoin(CLCMemFnCommand#{num_args}<R, C, #{args.join(", ")}>))(f, obj); }

END
end

(1...5).each do |i| gen_command(i) end
(1...5).each do |i| gen_creator(i) end
