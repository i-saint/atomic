#!/usr/local/bin/ruby


def gen_callers(num_args)
    class_args = []
    short_class_args = []

    arg_typedefs = []

    func_args = []
    pass_args = []


    (0...num_args).each do |i|
        class_args << "class A#{i}"
        short_class_args << "A#{i}"

        arg_typedefs << "RemoveCR(A#{i})"

        func_args << "const VA#{i} &a#{i}"
        pass_args << "args.a#{i}"
    end

    class_args = class_args.join(", ")
    short_class_args = short_class_args.join(", ")

    arg_typedefs = arg_typedefs.join(", ")

    pass_args = pass_args.join(", ")

    puts <<END

template<class R, #{class_args}>
struct BC_Fn#{num_args}
{
    typedef R (*F)(#{short_class_args});
    typedef RemoveCR(R) RT;
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        if(r) { *(RT*)r=f(#{pass_args}); }
        else  {         f(#{pass_args}); }
    }
};
template<#{class_args}>
struct BC_Fn#{num_args}<void, #{short_class_args}>
{
    typedef void (*F)(#{short_class_args});
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        f(#{pass_args});
    }
};

template<class R, class C, #{class_args}>
struct BC_MemFn#{num_args}
{
    typedef R (C::*F)(#{short_class_args});
    typedef RemoveCR(R) RT;
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, C &o, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        if(r) { *(RT*)r=(o.*f)(#{pass_args}); }
        else  {         (o.*f)(#{pass_args}); }
    }
};
template<class C, #{class_args}>
struct BC_MemFn#{num_args}<void, C, #{short_class_args}>
{
    typedef void (C::*F)(#{short_class_args});
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, C &o, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        (o.*f)(#{pass_args});
    }
};

template<class R, class C, #{class_args}>
struct BC_ConstMemFn#{num_args}
{
    typedef R (C::*F)(#{short_class_args}) const;
    typedef RemoveCR(R) RT;
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, const C &o, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        if(r) { *(RT*)r=(o.*f)(#{pass_args}); }
        else  {         (o.*f)(#{pass_args}); }
    }
};
template<class C, #{class_args}>
struct BC_ConstMemFn#{num_args}<void, C, #{short_class_args}>
{
    typedef void (C::*F)(#{short_class_args}) const;
    typedef ArgList<#{arg_typedefs}> ArgListT;
    void operator()(F f, const C &o, void *r, const void *a)
    {
        ArgListT &args = *(ArgListT*)a;
        (o.*f)(#{pass_args});
    }
};
END
end


def gen_calls(num_args)
    class_args = []
    short_class_args = []


    (0...num_args).each do |i|
        class_args << "class A#{i}"
        short_class_args << "A#{i}"
    end

    class_args = class_args.join(", ")
    short_class_args = short_class_args.join(",")


    puts <<END

template<class R, #{class_args}>
inline void BinaryCall(R (*f)(#{short_class_args}), void *r, const void *a)
{ BC_Fn#{num_args}<R,#{short_class_args}>()(f, r, a); }

template<class R, #{class_args}>
inline void BinaryCall(R (*f)(#{short_class_args}), R &r, const ArgList<#{short_class_args}> &a)
{ BC_Fn#{num_args}<R,#{short_class_args}>()(f, &r, &a); }

template<class R, #{class_args}>
inline void BinaryCall(R (*f)(#{short_class_args}), const void *a)
{ BC_Fn#{num_args}<R,#{short_class_args}>()(f, NULL, a); }

template<class R, #{class_args}>
inline void BinaryCall(R (*f)(#{short_class_args}), const ArgList<#{short_class_args}> &a)
{ BC_Fn#{num_args}<R,#{short_class_args}>()(f, NULL, &a); }


template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}), C &o, void *r, const void *a)
{ BC_MemFn#{num_args}<R,C,#{short_class_args}>()(f, o, r, a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}), C &o, R &r, const ArgList<#{short_class_args}> &a)
{ BC_MemFn#{num_args}<R,C,#{short_class_args}>()(f, o, &r, &a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}), C &o, const void *a)
{ BC_MemFn#{num_args}<R,C,#{short_class_args}>()(f, o, NULL, a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}), C &o, const ArgList<#{short_class_args}> &a)
{ BC_MemFn#{num_args}<R,C,#{short_class_args}>()(f, o, NULL, &a); }


template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}) const, const C &o, void *r, const void *a)
{ BC_ConstMemFn#{num_args}<R,C,#{short_class_args}>()(f, o, r, a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}) const, const C &o, R &r, const ArgList<#{short_class_args}> &a)
{ BC_ConstMemFn#{num_args}<R,C,#{short_class_args}>()(f, o, &r, &a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}) const, const C &o, const void *a)
{ BC_ConstMemFn#{num_args}<R,C,#{short_class_args}>()(f, o, NULL, a); }

template<class R, class C, #{class_args}>
inline void BinaryCall(R (C::*f)(#{short_class_args}) const, const C &o, const ArgList<#{short_class_args}> &a)
{ BC_ConstMemFn#{num_args}<R,C,#{short_class_args}>()(f, o, NULL, &a); }

END
end

(1...5).each do |i| gen_callers(i) end
(1...5).each do |i| gen_calls(i) end