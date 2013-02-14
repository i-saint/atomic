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
class CLCFunction#{num_args} : public ICLCommand
{
public:
    typedef R (*Func)(#{args.join(", ")});
#{arg_typedefs.join("\n")}

    CLCFunction#{num_args}(Func f) : m_f(f) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f
            #{arg_parse.join("\n")}
            )
        {
            m_f(#{tmp_args.join(", ")});
        }
    }
private:
    Func m_f;
    const char *m_args[#{num_args}];
};

template<class R, class C, #{class_args.join(", ")}>
class CLCMemFn#{num_args} : public ICLCommand
{
public:
    typedef R (C::*Func)(#{args.join(", ")});
#{arg_typedefs.join("\n")}

    CLCMemFn#{num_args}(Func f, C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f && m_obj
            #{arg_parse.join("\n")}
            )
        {
            (m_obj->*m_f)(#{tmp_args.join(", ")});
        }
    }
private:
    Func m_f;
    C *m_obj;
    const char *m_args[#{num_args}];
};

template<class R, class C, #{class_args.join(", ")}>
class CLCConstMemFn#{num_args} : public ICLCommand
{
public:
    typedef R (C::*Func)(#{args.join(", ")}) const;
#{arg_typedefs.join("\n")}

    CLCConstMemFn#{num_args}(Func f, const C *o) : m_f(f), m_obj(o) { clearArgs(); }
    virtual uint32 getNumArgs() const { return _countof(m_args); }
    virtual void setArg(uint32 i, const char *arg) { m_args[i]=arg; }
    virtual void clearArgs() { std::fill_n(m_args, _countof(m_args), (char*)NULL); }
    virtual void exec()
    {
#{tmp_arg_decls.join("\n")}
        if( m_f && m_obj
#{arg_parse.join("\n")}
            )
        {
            (m_obj->*m_f)(#{tmp_args.join(", ")});
        }
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
CLCFunction#{num_args}<R, #{args.join(", ")}>* CreateCLCommand(R (*f)(#{args.join(", ")}))
{ return istNew(istTypeJoin(CLCFunction#{num_args}<R, #{args.join(", ")}>))(f); }

template<class R, class C, #{class_args.join(", ")}>
CLCMemFn#{num_args}<R, C, #{args.join(", ")}>* CreateCLCommand(R (C::*f)(#{args.join(", ")}), C *obj)
{ return istNew(istTypeJoin(CLCMemFn#{num_args}<R, C, #{args.join(", ")}>))(f, obj); }

template<class R, class C, #{class_args.join(", ")}>
CLCConstMemFn#{num_args}<R, C, #{args.join(", ")}>* CreateCLCommand(R (C::*f)(#{args.join(", ")}) const, C *obj)
{ return istNew(istTypeJoin(CLCConstMemFn#{num_args}<R, C, #{args.join(", ")}>))(f, obj); }

END
end

(1...5).each do |i| gen_command(i) end
(1...5).each do |i| gen_creator(i) end
