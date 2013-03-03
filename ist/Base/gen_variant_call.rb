#!/usr/local/bin/ruby


def gen_callers(num_args)
    class_args = []
    variant_args = []

    short_class_args = []
    short_variant_args = []

    arg_typedefs = []
    variant_typedefs = []

    func_args = []
    pass_args = []


    (0...num_args).each do |i|
        class_args << "class A#{i}"
        variant_args << "size_t SA#{i}"

        short_class_args << "A#{i}"
        short_variant_args << "SA#{i}"

        arg_typedefs << "typedef RemoveCR(A#{i}) A#{i}T;"
        variant_typedefs << "typedef TVariant<SA#{i}> VA#{i};"

        func_args << "const VA#{i} &a#{i}"
        pass_args << "const_cast<A#{i}T&>(a#{i}.cast<A#{i}T>())"
    end

    class_args = class_args.join(", ")
    variant_args = variant_args.join(", ")

    short_class_args = short_class_args.join(", ")
    short_variant_args = short_variant_args.join(", ")

    arg_typedefs = arg_typedefs.join("\n    ")
    variant_typedefs = variant_typedefs.join("\n    ")

    func_args = func_args.join(", ")
    pass_args = pass_args.join(", ")

    puts <<END
template<class R, #{class_args}, size_t SR, #{variant_args}>
struct VC_Fn#{num_args}
{
    typedef R (*F)(#{short_class_args});
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, VR *r, #{func_args})
    {
        if(r){ *r=f(#{pass_args}); }
        else {    f(#{pass_args}); }
    }
};
template<#{class_args}, size_t SR, #{variant_args}>
struct VC_Fn#{num_args}<void, #{short_class_args}, SR, #{short_variant_args}>
{
    typedef void (*F)(#{short_class_args});
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, VR *r, #{func_args})
    {
        f(#{pass_args});
    }
};

template<class R, class C, #{class_args}, size_t SR, #{variant_args}>
struct VC_MemFn#{num_args}
{
    typedef R (C::*F)(#{short_class_args});
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, C *o, VR *r, #{func_args})
    {
        if(r){ *r=(o->*f)(#{pass_args}); }
        else {    (o->*f)(#{pass_args}); }
    }
};
template<class C, #{class_args}, size_t SR, #{variant_args}>
struct VC_MemFn#{num_args}<void, C, #{short_class_args}, SR, #{short_variant_args}>
{
    typedef void (C::*F)(#{short_class_args});
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, C *o, VR *r, #{func_args})
    {
        (o->*f)(#{pass_args});
    }
};

template<class R, class C, #{class_args}, size_t SR, #{variant_args}>
struct VC_ConstMemFn#{num_args}
{
    typedef R (C::*F)(#{short_class_args}) const;
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, const C *o, VR *r, #{func_args})
    {
        if(r){ *r=(o->*f)(#{pass_args}); }
        else {    (o->*f)(#{pass_args}); }
    }
};
template<class C, #{class_args}, size_t SR, #{variant_args}>
struct VC_ConstMemFn#{num_args}<void, C, #{short_class_args}, SR, #{short_variant_args}>
{
    typedef void (C::*F)(#{short_class_args}) const;
    #{arg_typedefs}
    typedef TVariant<SR> VR;
    #{variant_typedefs}
    void operator()(F f, const C *o, VR *r, #{func_args})
    {
        (o->*f)(#{pass_args});
    }
};
END
end


def gen_calls(num_args)
    class_args = []
    variant_args = []

    short_class_args = []
    short_variant_args = []

    func_args = []
    pass_args = []

    array_variant_args = []
    array_pass_args = []


    (0...num_args).each do |i|
        class_args << "class A#{i}"
        variant_args << "size_t SA#{i}"

        short_class_args << "A#{i}"
        short_variant_args << "SA#{i}"

        func_args << "TVariant<SA#{i}> &a#{i}"
        pass_args << "a#{i}"

        array_variant_args << "SA"
        array_pass_args << "va[#{i}]"
    end

    class_args = class_args.join(", ")
    variant_args = variant_args.join(", ")

    short_class_args = short_class_args.join(",")
    short_variant_args = short_variant_args.join(",")

    func_args = func_args.join(", ")
    pass_args = pass_args.join(", ")

    array_variant_args = array_variant_args.join(",")
    array_pass_args = array_pass_args.join(", ")


    puts <<END
template<class R, #{class_args}, size_t SR, #{variant_args}>
inline void VariantCall(R (*f)(#{short_class_args}), TVariant<SR> *r, #{func_args})
{ VC_Fn#{num_args}<R,#{short_class_args},SR,#{short_variant_args}>()(f, r, #{pass_args}); }

template<class R, #{class_args}, size_t SR, size_t SA>
inline void VariantCall(R (*f)(#{short_class_args}), TVariant<SR> *r, const TVariant<SA> *va)
{ VC_Fn#{num_args}<R,#{short_class_args},SR,#{array_variant_args}>()(f, r, #{array_pass_args}); }

template<class R, #{class_args}, #{variant_args}>
inline void VariantCall(R (*f)(#{short_class_args}), #{func_args})
{ VC_Fn#{num_args}<R,#{short_class_args},4,#{short_variant_args}>()(f, NULL, #{pass_args}); }

template<class R, #{class_args}, size_t SA>
inline void VariantCall(R (*f)(#{short_class_args}), const TVariant<SA> *va)
{ VC_Fn#{num_args}<R,#{short_class_args},4,#{array_variant_args}>()(f, NULL, #{array_pass_args}); }


template<class R, class C, #{class_args}, size_t SR, #{variant_args}>
inline void VariantCall(R (C::*f)(#{short_class_args}), C *o, TVariant<SR> *r, #{func_args})
{ VC_MemFn#{num_args}<R,C,#{short_class_args},SR,#{short_variant_args}>()(f, o, r, #{pass_args}); }

template<class R, class C, #{class_args}, size_t SR, size_t SA>
inline void VariantCall(R (C::*f)(#{short_class_args}), C *o, TVariant<SR> *r, const TVariant<SA> *va)
{ VC_MemFn#{num_args}<R,C,#{short_class_args},SR,#{array_variant_args}>()(f, o, r, #{array_pass_args}); }

template<class R, class C, #{class_args}, #{variant_args}>
inline void VariantCall(R (C::*f)(#{short_class_args}), C *o, #{func_args})
{ VC_MemFn#{num_args}<R,C,#{short_class_args},4,#{short_variant_args}>()(f, o, NULL, #{pass_args}); }

template<class R, class C, #{class_args}, size_t SA>
inline void VariantCall(R (C::*f)(#{short_class_args}), C *o, const TVariant<SA> *va)
{ VC_MemFn#{num_args}<R,C,#{short_class_args},4,#{array_variant_args}>()(f, o, NULL, #{array_pass_args}); }


template<class R, class C, #{class_args}, size_t SR, #{variant_args}>
inline void VariantCall(R (C::*f)(#{short_class_args}) const, const C *o, TVariant<SR> *r, #{func_args})
{ VC_ConstMemFn#{num_args}<R,C,#{short_class_args},SR,#{short_variant_args}>()(f, o, r, #{pass_args}); }

template<class R, class C, #{class_args}, size_t SR, size_t SA>
inline void VariantCall(R (C::*f)(#{short_class_args}) const, const C *o, TVariant<SR> *r, const TVariant<SA> *va)
{ VC_ConstMemFn#{num_args}<R,C,#{short_class_args},SR,#{array_variant_args}>()(f, o, r, #{array_pass_args}); }

template<class R, class C, #{class_args}, #{variant_args}>
inline void VariantCall(R (C::*f)(#{short_class_args}) const, const C *o, #{func_args})
{ VC_ConstMemFn#{num_args}<R,C,#{short_class_args},4,#{short_variant_args}>()(f, o, NULL, #{pass_args}); }

template<class R, class C, #{class_args}, size_t SA>
inline void VariantCall(R (C::*f)(#{short_class_args}) const, const C *o, const TVariant<SA> *va)
{ VC_ConstMemFn#{num_args}<R,C,#{short_class_args},4,#{array_variant_args}>()(f, o, NULL, #{array_pass_args}); }



END
end

(1...5).each do |i| gen_callers(i) end
(1...5).each do |i| gen_calls(i) end