#ifndef ist_Base_EnumString_h
#define ist_Base_EnumString_h

namespace ist {

struct EnumStr
{
    int32 num;
    const char *str;
};

struct equal_num
{
    int32 num;
    equal_num(int32 n) : num(n) {}
    bool operator()(const EnumStr &es) const
    {
        return es.num==num;
    }
};

struct equal_str
{
    const char *str;
    equal_str(const char *s) : str(s) {}
    bool operator()(const EnumStr &es) const
    {
        return strcmp(es.str, str)==0;
    }
};


} // namespace ist

#endif // ist_Base_EnumString_h


#ifdef istSEnumBlock
#   undef istSEnumBlock
#   undef istSEnum
#   undef istSEnumEq
#endif // istSEnumBlock


#ifndef istStringnizeEnum

#define istSEnumBlock(TypeName, ...)\
    enum TypeName {\
        __VA_ARGS__\
    };\
    const char* Get##TypeName##String(TypeName num);\
    TypeName Get##TypeName##Num(const char *str);\
    void TypeName##EachPair(const std::function<void (const ist::EnumStr&)> &f);

#define istSEnum(Elem) Elem
#define istSEnumEq(Elem, Eq) Elem = Eq

#else // istStringnizeEnum
#define istSEnumBlock(TypeName, ...)\
    static const ist::EnumStr g_pairs_##TypeName[] = {\
        __VA_ARGS__\
    };\
    const char* Get##TypeName##String(TypeName num)\
    {\
        const ist::EnumStr *beg = g_pairs_##TypeName;\
        const ist::EnumStr *end = g_pairs_##TypeName+_countof(g_pairs_##TypeName);\
        const ist::EnumStr *res = std::find_if(beg, end, ist::equal_num(num));\
        return res==end ? NULL : res->str;\
    }\
    TypeName Get##TypeName##Num(const char *str)\
    {\
        const ist::EnumStr *beg = g_pairs_##TypeName;\
        const ist::EnumStr *end = g_pairs_##TypeName+_countof(g_pairs_##TypeName);\
        const ist::EnumStr *res = std::find_if(beg, end, ist::equal_str(str));\
        return TypeName(res==end ? 0 : res->num);\
    }\
    void TypeName##EachPair(const std::function<void (const ist::EnumStr&)> &f)\
    {\
        const ist::EnumStr *beg = g_pairs_##TypeName;\
        const ist::EnumStr *end = g_pairs_##TypeName+_countof(g_pairs_##TypeName);\
        std::for_each(beg, end, f);\
    }\

#define istSEnum(Elem)       {(int32)Elem, #Elem}
#define istSEnumEq(Elem, Eq) {(int32)Elem, #Elem}


#endif // istStringnizeEnum
