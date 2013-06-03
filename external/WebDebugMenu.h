// created by i-saint
// distributed under Creative Commons Attribution (CC BY) license.
// https://github.com/i-saint/WebDebugMenu

#ifndef WebDebugMenu_h
#define WebDebugMenu_h

#ifndef wdmDisable
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstdarg>
#include <stdint.h>

#if   defined(wdmDLL_Impl)
#   define wdmIntermodule __declspec(dllexport)
#elif defined(wdmStatic)
#   define wdmIntermodule
#else // wdmDynamic
#   define wdmIntermodule __declspec(dllimport)
#   ifdef _WIN64
#       pragma comment(lib, "WebDebugMenu64.lib")
#   else // _WIN64
#       pragma comment(lib, "WebDebugMenu.lib")
#   endif // _WIN64
#endif

// wdmEvent, wdmNode は dll を跨ぐ可能性がある
// このため、これらは STL のコンテナなどには触れないようにする

typedef uint32_t wdmID;
struct wdmEvent
{
    wdmID node;
    const char *command;
};

struct wdmConfig
{
    uint16_t port;
    uint16_t max_queue;
    uint16_t max_threads;
    uint32_t json_reserve_size;

    wdmConfig();
    bool load(const char *path);
};


class wdmNode
{
protected:
    virtual             ~wdmNode() {}
public:
    virtual void        release()=0;
    virtual wdmID       getID() const=0;
    virtual const char* getName() const=0;
    virtual size_t      getNumChildren() const=0;
    virtual wdmNode*    findChild(const char *name) const=0;
    virtual wdmNode*    getChild(size_t i) const=0;
    virtual void        addChild(const char *path, wdmNode *child)=0;
    virtual void        eraseChild(const char *path)=0;
    virtual void        setName(const char *name, size_t len=0)=0;
    virtual size_t      jsonize(char *out, size_t len, int recursion) const=0;
    virtual bool        handleEvent(const wdmEvent &evt)=0;
};

extern "C" {
    wdmIntermodule void             wdmInitialize();
    wdmIntermodule void             wdmFinalize();
    wdmIntermodule void             wdmFlush();

    wdmIntermodule void             wdmOpenBrowser();
    wdmIntermodule const wdmConfig* wdmGetConfig();

    // 内部実装用
    wdmIntermodule wdmID    _wdmGenerateID();
    wdmIntermodule wdmNode* _wdmGetRootNode();
    wdmIntermodule void     _wdmRegisterNode(wdmNode *node);
    wdmIntermodule void     _wdmUnregisterNode(wdmNode *node);
};



// 以下はユーザー側にのみ見え、WebDebugMenu.dll からは一切触れないコード
#pragma region wdmImplRegion

#ifdef _MSC_VER
#   define wdmSNPrintf    _snprintf
#   define wdmVSNPrintf   _vsnprintf
#else // _MSC_VER
#   define wdmSNPrintf    snprintf
#   define wdmVSNPrintf   vsnprintf
#endif // _MSC_VER

// std::remove_const はポインタの const は外さないので、外すものを用意
template<class T> struct wdmRemoveConst { typedef T type; };
template<class T> struct wdmRemoveConst<const T> { typedef T type; };
template<class T> struct wdmRemoveConst<const T*> { typedef T* type; };
template<class T> struct wdmRemoveConst<const T&> { typedef T& type; };
template<class T, size_t L> struct wdmRemoveConst<const T[L]> { typedef T type[L] ; };

template<class T> struct wdmRemoveConstReference { typedef T type; };
template<class T> struct wdmRemoveConstReference<const T> { typedef T type; };
template<class T> struct wdmRemoveConstReference<const T*> { typedef T* type; };
template<class T> struct wdmRemoveConstReference<const T&> { typedef T type; };
template<class T, size_t L> struct wdmRemoveConstReference<const T[L]> { typedef T type[L] ; };

typedef std::string wdmString;
template<class T> inline const char* wdmTypename();
template<class T> inline bool wdmParse(const char *text, T &value);
template<class T> inline size_t wdmToS(char *out, size_t len, T value);

inline bool wdmNextSeparator(const char *s, size_t &pos)
{
    int brace = 0;
    int bracket = 0;
    for(;; ++pos) {
        if     (s[pos]==',') { if(brace==0 && bracket==0) {return true;} }
        else if(s[pos]=='\0') { return false; }
        else if(s[pos]=='{') { ++brace; }
        else if(s[pos]=='}') { --brace; }
        else if(s[pos]=='[') { ++bracket; }
        else if(s[pos]==']') { --bracket; }
    }
    return false;
}

template<class T, size_t L>
struct wdmArrayParseImpl
{
    size_t operator()(const char *text, T (&value)[L])
    {
        size_t num_parsed = 0;
        size_t pos = 0;
        T tmp;
        if(text[pos++]!='[') { return 0; }
        for(;;) {
            if(wdmParse(text+pos, tmp)) {
                value[num_parsed++] = tmp;
                if(!wdmNextSeparator(text, pos)) { break; }
                ++pos;
            }
        }
        return num_parsed;
    }
};
template<class T, size_t L> inline size_t wdmParse(const char *text, T (&value)[L]) { return wdmArrayParseImpl<T,L>()(text, value); }

template<class T, size_t L>
struct wdmArrayToSImpl
{
    size_t operator()(char *out, size_t len, const T (&value)[L])
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "[");
        for(size_t i=0; i<L; ++i) {
            s += wdmToS(out+s, len-s, value[i]);
            if(i+1!=L) { s += wdmSNPrintf(out+s, len-s, ","); }
        }
        s += wdmSNPrintf(out+s, len-s, "]");
        return s;
    }
};
template<class T, size_t L> inline size_t wdmToS(char *out, size_t len, const T (&value)[L]) { return wdmArrayToSImpl<T,L>()(out, len, value); }

class wdmNodeBase : public wdmNode
{
protected:
    virtual ~wdmNodeBase()
    {
        _wdmUnregisterNode(this);
        for(size_t i=0; i<m_children.size(); ++i) { m_children[i]->release(); }
    }

public:
    struct wdmEqualName
    {
        const char *m_name;
        size_t m_len;
        wdmEqualName(const char *name, size_t len) : m_name(name), m_len(len) {}
        bool operator()(const wdmNode *a) const
        {
            const char *name = a->getName();
            size_t len = strlen(name);
            return len==m_len && strncmp(name, m_name, len)==0;
        }
    };
    static inline size_t wdmFindSeparator(const char *s)
    {
        size_t i=0;
        for(;; ++i) {
            if(s[i]=='/' || s[i]=='\0') { break; }
        }
        return i;
    }
    typedef std::vector<wdmNode*> node_cont;

    wdmNodeBase()
        : m_id(_wdmGenerateID())
    {
        _wdmRegisterNode(this);
    }

    virtual void release() { delete this; }

    virtual wdmID       getID() const               { return m_id; }
    virtual const char* getName() const             { return m_name.c_str(); }
    virtual size_t      getNumChildren() const      { return m_children.size(); }
    virtual wdmNode*    getChild(size_t i) const    { return m_children[i]; }

    virtual wdmNode* findChild(const char *path) const
    {
        size_t s = wdmFindSeparator(path);
        node_cont::const_iterator i=std::find_if(m_children.begin(), m_children.end(), wdmEqualName(path, s));
        wdmNode *c = i==m_children.end() ? NULL : *i;
        return path[s]=='/' ? c->findChild(path+s+1) : c;
    }

    virtual void addChild(const char *path, wdmNode *child)
    {
        size_t s = wdmFindSeparator(path);
        node_cont::const_iterator i=std::find_if(m_children.begin(), m_children.end(), wdmEqualName(path, s));
        wdmNode *n = i==m_children.end() ? NULL : *i;
        if(path[s]=='/') {
            if(n==NULL) {
                n = new wdmNodeBase();
                n->setName(path, s);
                m_children.push_back(n);
            }
            n->addChild(path+s+1, child);
        }
        else {
            // 同名ノードがある場合、古いのは削除
            if(n!=NULL) {
                n->release();
                m_children.erase(i);
            }
            child->setName(path);
            m_children.push_back(child);
        }
    }

    virtual void eraseChild(const char *path)
    {
        size_t s = wdmFindSeparator(path);
        node_cont::const_iterator i=std::find_if(m_children.begin(), m_children.end(), wdmEqualName(path, s));
        wdmNode *n = i==m_children.end() ? NULL : *i;
        if(path[s]=='/') {
            if(n!=NULL) {
                n->eraseChild(path+s+1);
            }
        }
        else {
            if(n!=NULL) {
                n->release();
                m_children.erase(i);
            }
        }
    }

    virtual void setName(const char *name, size_t len=0)
    {
        m_name = len!=0 ? wdmString(name, len) : name;
    }

    size_t jsonizeChildren(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "\"hasChildren\": %d", getNumChildren()!=0);
        if(recursion-- && getNumChildren()!=0) {
            s += wdmSNPrintf(out+s, len-s, ", \"children\": [");
            for(size_t i=0; i<getNumChildren(); ++i) {
                s += getChild(i)->jsonize(out+s, len-s, recursion);
                if(i+1!=getNumChildren()) { s += wdmSNPrintf(out+s, len-s, ", "); }
            }
            s += wdmSNPrintf(out+s, len-s, "]");
        }
        return s;
    }

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", ", getID(), getName());
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &/*evt*/)
    {
        return false;
    }

private:
    node_cont m_children;
    wdmID m_id;
    wdmString m_name;
};

template<class T> struct wdmIsReadOnly { static const bool value=std::is_const<T>::value; };
template<> struct wdmIsReadOnly<char*>   { static const bool value=true; };
template<> struct wdmIsReadOnly<wchar_t*>{ static const bool value=true; };

template<class T> struct wdmCanBeRanged
{
    typedef typename wdmRemoveConstReference<T>::type value_t;
    static const bool value = std::is_arithmetic<value_t>::value;
};

template<class T, bool valid=wdmCanBeRanged<T>::value>
struct wdmRange
{
    typedef typename wdmRemoveConstReference<T>::type value_t;
    value_t min_value;
    value_t max_value;
    bool enabled;

    wdmRange() : enabled(false), min_value(), max_value() {}
    wdmRange(T min_v, T max_v) : enabled(true), min_value(min_v), max_value(max_v) {}
    size_t jsonize(char *out, size_t len) const
    {
        size_t s = 0;
        if(enabled) {
            s += wdmSNPrintf(out+s, len-s, "\"range\":[");
            s += wdmToS(out+s, len-s, min_value);
            s += wdmSNPrintf(out+s, len-s, ", ");
            s += wdmToS(out+s, len-s, max_value);
            s += wdmSNPrintf(out+s, len-s, "], ");
        }
        return s;
    }
};
template<class T>
struct wdmRange<T, false>
{
    wdmRange() {}
    size_t jsonize(char *out, size_t len) const { return 0; }
};

template<class T, bool available=!wdmIsReadOnly<T>::value>
struct wdmHandleSet
{
    bool operator()(const wdmEvent &evt, T *value)
    {
        if(strncmp(evt.command, "set(", 4)==0) {
            wdmParse(evt.command+4, *value);
            return true;
        }
        return false;
    }
};
template<class T>
struct wdmHandleSet<T, false> {
    bool operator()(const wdmEvent &evt, const T *value) { return false; }
};

template<class T, bool available=!wdmIsReadOnly<T>::value && std::is_array<T>::value>
struct wdmHandleAt
{
    bool operator()(const wdmEvent &evt, T *value)
    {
        if(strncmp(evt.command, "at(", 3)==0) {
            int i = atoi(evt.command+3);
            size_t pos=3; wdmNextSeparator(evt.command, pos); ++pos;
            if(wdmParse(evt.command+pos, (*value)[i])) {
                return true;
            }
        }
        return false;
    }
};
template<class T>
struct wdmHandleAt<T, false> {
    bool operator()(const wdmEvent &, const T *) { return false; }
};

template<class T, class T2=T>
class wdmDataNode : public wdmNodeBase
{
typedef wdmNodeBase super;
public:
    typedef T arg_t;
    typedef typename wdmRemoveConstReference<T>::type value_t;
    typedef wdmRange<T2> range_t;

    wdmDataNode(arg_t *value, const range_t &range=range_t()) : m_range(range), m_value(value) {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"type\":\"%s\",", getID(), getName(), wdmTypename<value_t>());
#pragma warning(push)
#pragma warning(disable:4127)
        if(wdmIsReadOnly<arg_t>::value) {
            s += wdmSNPrintf(out+s, len-s, "\"readonly\":true, ");
        }
#pragma warning(pop)
        {
            s += wdmSNPrintf(out+s, len-s, "\"value\":");
            s += wdmToS(out+s, len-s, (const value_t&)*m_value);
            s += wdmSNPrintf(out+s, len-s, ", ");
        }
        s += m_range.jsonize(out+s, len-s);
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        return wdmHandleSet<arg_t>()(evt, m_value) || wdmHandleAt<arg_t>()(evt, m_value) || super::handleEvent(evt);
    }

private:
    range_t m_range;
    arg_t *m_value;
};


template<class T, size_t L, class T2=T>
class wdmArrayNode : public wdmNodeBase
{
typedef wdmNodeBase super;
public:
    typedef T (arg_t)[L];
    typedef typename wdmRemoveConstReference<T>::type value_t;
    typedef wdmRange<T2> range_t;

    wdmArrayNode(arg_t *value, const range_t &range=range_t()) : m_range(range), m_value(value) {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"type\":\"%s\", \"length\":%d, ", getID(), getName(), wdmTypename<value_t>(), (int)L);
        if(wdmIsReadOnly<arg_t>::value) {
            s += wdmSNPrintf(out+s, len-s, "\"readonly\":true, ");
        }
        {
            s += wdmSNPrintf(out+s, len-s, "\"value\":");
            s += wdmToS(out+s, len-s, *m_value);
            s += wdmSNPrintf(out+s, len-s, ", ");
        }
        s += m_range.jsonize(out+s, len-s);
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        return wdmHandleSet<arg_t>()(evt, m_value) || wdmHandleAt<arg_t>()(evt, m_value) || super::handleEvent(evt);
    }

private:
    range_t m_range;
    arg_t *m_value;
};

template<class T, class T2=T>
class wdmPropertyNode : public wdmNodeBase
{
typedef wdmNodeBase super;
public:
    typedef T arg_t;
    typedef typename wdmRemoveConstReference<T>::type value_t;
    typedef std::function<arg_t ()>     getter_t;
    typedef std::function<void (arg_t)> setter_t;
    typedef wdmRange<T2> range_t;

    wdmPropertyNode(getter_t getter=getter_t(), setter_t setter=setter_t(), const range_t &range=range_t())
        : m_getter(getter)
        , m_setter(setter)
        , m_range(range)
    {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"type\":\"%s\",", getID(), getName(), wdmTypename<value_t>());
        if(!m_setter) {
            s += wdmSNPrintf(out+s, len-s, "\"readonly\":true, ");
        }
        if(m_getter) {
            s += wdmSNPrintf(out+s, len-s, "\"value\":");
            arg_t t = m_getter();
            s += wdmToS(out+s, len-s, (const value_t&)t);
            s += wdmSNPrintf(out+s, len-s, ", ");
        }
        s += m_range.jsonize(out+s, len-s);
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        if(m_setter) {
            if(strncmp(evt.command, "set(", 4)==0) {
                value_t tmp;
                if(wdmParse(evt.command+4, tmp)) {
                    m_setter(tmp);
                    return true;
                }
            }
        }
        return super::handleEvent(evt);
    }

private:
    getter_t m_getter;
    setter_t m_setter;
    range_t m_range;
};

template<class R>
class wdmFunctionNode0 : public wdmNodeBase
{
typedef wdmNodeBase super;
public:
    typedef std::function<R ()>  func_t;

    wdmFunctionNode0(func_t func)
        : m_func(func)
    {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"callable\":true, \"argTypes\":[],", getID(), getName() );
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        if(strncmp(evt.command, "call(", 5)==0) {
            if(m_func) {
                m_func();
                return true;
            }
        }
        return super::handleEvent(evt);
    }

private:
    func_t m_func;
};

template<class R, class A0>
class wdmFunctionNode1 : public wdmNodeBase
{
    typedef wdmNodeBase super;
public:
    typedef std::function<R (A0)>  func_t;
    typedef typename wdmRemoveConstReference<A0>::type a0_t;

    wdmFunctionNode1(func_t func) : m_func(func), m_a0()
    {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"callable\":true, \"argTypes\":[\"%s\"],", getID(), getName(), wdmTypename<a0_t>() );
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        if(strncmp(evt.command, "arg(", 4)==0) {
            int i = atoi(evt.command+4);
            size_t pos=4; wdmNextSeparator(evt.command, pos); ++pos;
            switch(i) {
            case 0: wdmParse(evt.command+pos, m_a0); break;
            }
            return true;
        }
        else if(strncmp(evt.command, "call(", 5)==0) {
            if(m_func) {
                m_func(m_a0);
                return true;
            }
        }
        return super::handleEvent(evt);
    }

private:
    func_t m_func;
    a0_t m_a0;
};

template<class R, class A0, class A1>
class wdmFunctionNode2 : public wdmNodeBase
{
typedef wdmNodeBase super;
public:
    typedef std::function<R (A0, A1)>  func_t;
    typedef typename wdmRemoveConstReference<A0>::type a0_t;
    typedef typename wdmRemoveConstReference<A1>::type a1_t;

    wdmFunctionNode2(func_t func) : m_func(func) , m_a0(), m_a1()
    {}

    virtual size_t jsonize(char *out, size_t len, int recursion) const
    {
        size_t s = 0;
        s += wdmSNPrintf(out+s, len-s, "{\"id\":%d, \"name\":\"%s\", \"callable\":true, \"argTypes\":[\"%s\",\"%s\"],",
            getID(), getName(), wdmTypename<a0_t>(), wdmTypename<a1_t>() );
        s += jsonizeChildren(out+s, len-s, recursion);
        s += wdmSNPrintf(out+s, len-s, "}");
        return s;
    }

    virtual bool handleEvent(const wdmEvent &evt)
    {
        if(strncmp(evt.command, "arg(", 4)==0) {
            int i = atoi(evt.command+4);
            size_t pos=4; wdmNextSeparator(evt.command, pos); ++pos;
            switch(i) {
            case 0: wdmParse(evt.command+pos, m_a0); break;
            case 1: wdmParse(evt.command+pos, m_a1); break;
            }
            return true;
        }
        else if(strncmp(evt.command, "call(", 5)==0) {
            if(m_func) {
                m_func(m_a0, m_a1);
                return true;
            }
        }
        return super::handleEvent(evt);
    }

private:
    func_t m_func;
    a0_t m_a0;
    a1_t m_a1;
};

#pragma endregion

inline wdmString wdmFormat(const char *fmt, ...)
{
    char buf[1024];
    va_list vl;
    va_start(vl, fmt);
    wdmVSNPrintf(buf, sizeof(buf), fmt, vl);
    va_end(vl);
    return buf;
}

#define wdmScope(...) __VA_ARGS__

// data node
template<class T>
inline void wdmAddNode(const wdmString &path, T *value)
{
    _wdmGetRootNode()->addChild(path.c_str(), new wdmDataNode<T>(value));
}
template<class T, class T2>
inline void wdmAddNode(const wdmString &path, T *value, T2 min_value, T2 max_value)
{
    _wdmGetRootNode()->addChild(path.c_str(), new wdmDataNode<T, T2>(value, wdmRange<T2>(min_value, max_value)));
}

// array node
template<class T, size_t L>
inline void wdmAddNode(const wdmString &path, T (*value)[L])
{
    _wdmGetRootNode()->addChild(path.c_str(), new wdmArrayNode<T,L>(value));
}
template<class T, size_t L, class T2>
inline void wdmAddNode(const wdmString &path, T (*value)[L], T2 min_value, T2 max_value)
{
    _wdmGetRootNode()->addChild(path.c_str(), new wdmArrayNode<T,L,T2>(value, wdmRange<T2>(min_value, max_value)));
}

// property node
template<class T>
inline void wdmAddNode(const wdmString &path, T (*getter)(), void (*setter)(T))
{
    auto *n = new wdmPropertyNode<T>(std::function<T ()>(getter), std::bind(setter, std::placeholders::_1));
    _wdmGetRootNode()->addChild(path.c_str(), n);
}
template<class T, class T2>
inline void wdmAddNode(const wdmString &path, T (*getter)(), void (*setter)(T), T2 min_value, T2 max_value)
{
    auto *n = new wdmPropertyNode<T>(std::function<T ()>(getter), std::bind(setter, std::placeholders::_1), wdmRange<T>(min_value, max_value));
    _wdmGetRootNode()->addChild(path.c_str(), n);
}
template<class C, class C2, class T>
inline void wdmAddNode(const wdmString &path, C *_this, T (C2::*getter)() const, void (C2::*setter)(T))
{
    auto *n = new wdmPropertyNode<T>(std::bind(getter, _this), std::bind(setter, _this, std::placeholders::_1));
    _wdmGetRootNode()->addChild(path.c_str(), n);
}
template<class C, class C2, class T, class T2>
inline void wdmAddNode(const wdmString &path, C *_this, T (C2::*getter)() const, void (C2::*setter)(T), T2 min_value, T2 max_value)
{
    auto *n = new wdmPropertyNode<T>(std::bind(getter, _this), std::bind(setter, _this, std::placeholders::_1), wdmRange<T>(min_value, max_value));
    _wdmGetRootNode()->addChild(path.c_str(), n);
}
template<class C, class C2, class T>
inline void wdmAddNode(const wdmString &path, C *_this, T (C2::*getter)() const)
{
    auto *n = new wdmPropertyNode<T>(std::bind(getter, _this));
    _wdmGetRootNode()->addChild(path.c_str(), n);
}

// function node (0 args)
template<class R>
inline void wdmAddNode(const wdmString &path, R (*f)())
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode0<R>(std::function<R ()>(f)) );
}
template<class R, class C, class C2>
inline void wdmAddNode(const wdmString &path, R (C::*mf)(), C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode0<R>(std::bind(mf, _this)) );
}
template<class R, class C, class C2>
inline void wdmAddNode(const wdmString &path, R (C::*cmf)() const, const C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode0<R>(std::bind(cmf, _this)) );
}
// function node (1 args)
template<class R, class A0>
inline void wdmAddNode(const wdmString &path, R (*f)(A0))
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode1<R,A0>(std::bind(f, std::placeholders::_1)) );
}
template<class R, class C, class C2, class A0>
inline void wdmAddNode(const wdmString &path, R (C::*mf)(A0), C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode1<R,A0>(std::bind(mf, _this, std::placeholders::_1)) );
}
template<class R, class C, class C2, class A0>
inline void wdmAddNode(const wdmString &path, R (C::*cmf)(A0) const, const C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode1<R,A0>(std::bind(cmf, _this, std::placeholders::_1)) );
}
// function node (2 args)
template<class R, class A0, class A1>
inline void wdmAddNode(const wdmString &path, R (*f)(A0,A1))
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode2<R,A0,A1>(std::bind(f, std::placeholders::_1, std::placeholders::_2)) );
}
template<class R, class C, class C2, class A0, class A1>
inline void wdmAddNode(const wdmString &path, R (C::*mf)(A0,A1), C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode2<R,A0,A1>(std::bind(mf, _this, std::placeholders::_1, std::placeholders::_2)) );
}
template<class R, class C, class C2, class A0, class A1>
inline void wdmAddNode(const wdmString &path, R (C::*cmf)(A0,A1) const, const C2 *_this)
{
    _wdmGetRootNode()->addChild( path.c_str(), new wdmFunctionNode2<R,A0,A1>(std::bind(cmf, _this, std::placeholders::_1, std::placeholders::_2)) );
}


inline void wdmEraseNode(const wdmString &path)
{
    _wdmGetRootNode()->eraseChild(path.c_str());
}


#pragma region wdmImplRegion2
// built-in types
template<> inline const char* wdmTypename<int8_t >() { return "int8"; }
template<> inline const char* wdmTypename<int16_t>() { return "int16"; }
template<> inline const char* wdmTypename<int32_t>() { return "int32"; }
template<> inline const char* wdmTypename<int64_t>() { return "int64"; }
template<> inline const char* wdmTypename<uint8_t >(){ return "uint8"; }
template<> inline const char* wdmTypename<uint16_t>(){ return "uint16"; }
template<> inline const char* wdmTypename<uint32_t>(){ return "uint32"; }
template<> inline const char* wdmTypename<uint64_t>(){ return "uint64"; }
template<> inline const char* wdmTypename<bool>()    { return "bool"; }
template<> inline const char* wdmTypename<float>()   { return "float32"; }
template<> inline const char* wdmTypename<double>()  { return "float64"; }
template<> inline const char* wdmTypename<char>()    { return "char"; }
template<> inline const char* wdmTypename<char*>()   { return "char*"; }
template<> inline const char* wdmTypename<wchar_t>() { return "wchar"; }
template<> inline const char* wdmTypename<wchar_t*>(){ return "wchar*"; }

template<> inline bool wdmParse(const char *text, int8_t  &v)  { int32_t t;  if(sscanf(text, "%i", &t)==1){ v=(int8_t)t; return true; } return false; }
template<> inline bool wdmParse(const char *text, int16_t &v)  { int32_t t;  if(sscanf(text, "%i", &t)==1){ v=(int16_t)t; return true; } return false; }
template<> inline bool wdmParse(const char *text, int32_t &v)  { int32_t t;  if(sscanf(text, "%i", &t)==1){ v=t; return true; } return false; }
template<> inline bool wdmParse(const char *text, int64_t &v)  { int64_t t;  if(sscanf(text, "%lli", &t)==1){ v=t; return true; } return false; }
template<> inline bool wdmParse(const char *text, uint8_t  &v) { uint32_t t; if(sscanf(text, "%u", &t)==1){ v=(uint8_t)t; return true; } return false; }
template<> inline bool wdmParse(const char *text, uint16_t &v) { uint32_t t; if(sscanf(text, "%u", &t)==1){ v=(uint16_t)t; return true; } return false; }
template<> inline bool wdmParse(const char *text, uint32_t &v) { uint32_t t; if(sscanf(text, "%u", &t)==1){ v=t; return true; } return false; }
template<> inline bool wdmParse(const char *text, uint64_t &v) { uint64_t t; if(sscanf(text, "%llu", &t)==1){ v=t; return true; } return false; }
template<> inline bool wdmParse(const char *text, bool &v)     { int32_t t;  if(sscanf(text, "%i", &t)==1){ v=t!=0; return true; } return false; }
template<> inline bool wdmParse(const char *text, float &v)    { return sscanf(text, "%f", &v)==1; }
template<> inline bool wdmParse(const char *, char &)     { return false; }
template<> inline bool wdmParse(const char *, wchar_t &)  { return false; }
template<> inline bool wdmParse(const char *, char *&)    { return false; }
template<> inline bool wdmParse(const char *, wchar_t *&) { return false; }

template<> inline size_t wdmToS(char *out, size_t len, int8_t  v)  { return wdmSNPrintf(out, len, "%i", (int32_t)v); }
template<> inline size_t wdmToS(char *out, size_t len, int16_t v)  { return wdmSNPrintf(out, len, "%i", (int32_t)v); }
template<> inline size_t wdmToS(char *out, size_t len, int32_t v)  { return wdmSNPrintf(out, len, "%i", v); }
template<> inline size_t wdmToS(char *out, size_t len, int64_t v)  { return wdmSNPrintf(out, len, "%lli", v); }
template<> inline size_t wdmToS(char *out, size_t len, uint8_t  v) { return wdmSNPrintf(out, len, "%u", (uint32_t)v); }
template<> inline size_t wdmToS(char *out, size_t len, uint16_t v) { return wdmSNPrintf(out, len, "%u", (uint32_t)v); }
template<> inline size_t wdmToS(char *out, size_t len, uint32_t v) { return wdmSNPrintf(out, len, "%u", v); }
template<> inline size_t wdmToS(char *out, size_t len, uint64_t v) { return wdmSNPrintf(out, len, "%llu", v); }
template<> inline size_t wdmToS(char *out, size_t len, bool v)     { return wdmSNPrintf(out, len, "%i", (int32_t)v); }
template<> inline size_t wdmToS(char *out, size_t len, float v)    { return wdmSNPrintf(out, len, "%f", v); }
template<> inline size_t wdmToS(char *out, size_t len, double v)   { return wdmSNPrintf(out, len, "%lf", v); }
template<> inline size_t wdmToS(char *out, size_t len, const char * v)    { return wdmSNPrintf(out, len, "\"%s\"", v); }
template<> inline size_t wdmToS(char *out, size_t len, const wchar_t * v) {
    size_t required = wcstombs(NULL, v, 0);
    if(required+2<=len) {
        size_t pos = 0;
        out[pos++]='"';
        pos += wcstombs(out+pos, v, len);
        out[pos++]='"';
        return required+2;
    }
    else {
        return len;
    }
}
template<> inline size_t wdmToS(char *out, size_t len, char *v)    { return wdmToS<const char*>(out, len, v); }
template<> inline size_t wdmToS(char *out, size_t len, wchar_t *v) { return wdmToS<const wchar_t*>(out, len, v); }

template<size_t L> struct wdmArrayToSImpl<char,    L> { size_t operator()(char *out, size_t len, const char    (&value)[L]) { return wdmToS(out, len, (char*)value);    } };
template<size_t L> struct wdmArrayToSImpl<wchar_t, L> { size_t operator()(char *out, size_t len, const wchar_t (&value)[L]) { return wdmToS(out, len, (wchar_t*)value); } };

template<size_t L> struct wdmArrayParseImpl<char, L> {
    size_t operator()(const char *text, char (&value)[L]) {
        if(text[0]!='"') { return 0; }
        const char *beg = text+1;
        int l = 0;
        for(; beg[l]!='\0' && !(beg[l]=='"' && beg[l-1]!='\\'); ++l) {}
        size_t n = std::min<size_t>(l, L-1);
        strncpy(value, beg, n);
        value[n] = '\0';
        return 1;
    }
};
template<size_t L> struct wdmArrayParseImpl<wchar_t, L> {
    size_t operator()(const char *text, wchar_t (&value)[L]) {
        if(text[0]!='"') { return 0; }
        const char *beg = text+1;
        int l = 0;
        for(; beg[l]!='\0' && !(beg[l]=='"' && beg[l-1]!='\\'); ++l) {}
        size_t n = std::min<size_t>(l, L-1);
        mbstowcs(value, beg, n);
        value[n] = L'\0';
        return 1;
    }
};

// vector types
struct wdmInt32x2 { int v[2]; int& operator[](size_t i){return v[i];} };
struct wdmInt32x3 { int v[3]; int& operator[](size_t i){return v[i];} };
struct wdmInt32x4 { int v[4]; int& operator[](size_t i){return v[i];} };
struct wdmFloat32x2 { float v[2]; float& operator[](size_t i){return v[i];} };
struct wdmFloat32x3 { float v[3]; float& operator[](size_t i){return v[i];} };
struct wdmFloat32x4 { float v[4]; float& operator[](size_t i){return v[i];} };

template<> inline const char* wdmTypename<wdmInt32x2  >() { return "int32x2"; }
template<> inline const char* wdmTypename<wdmInt32x3  >() { return "int32x3"; }
template<> inline const char* wdmTypename<wdmInt32x4  >() { return "int32x4"; }
template<> inline const char* wdmTypename<wdmFloat32x2>() { return "float32x2"; }
template<> inline const char* wdmTypename<wdmFloat32x3>() { return "float32x3"; }
template<> inline const char* wdmTypename<wdmFloat32x4>() { return "float32x4"; }

template<> inline bool wdmParse(const char *text, wdmInt32x2   &v) { return sscanf(text, "[%d,%d]", &v[0],&v[1])==2; }
template<> inline bool wdmParse(const char *text, wdmInt32x3   &v) { return sscanf(text, "[%d,%d,%d]", &v[0],&v[1],&v[2])==3; }
template<> inline bool wdmParse(const char *text, wdmInt32x4   &v) { return sscanf(text, "[%d,%d,%d,%d]", &v[0],&v[1],&v[2],&v[3])==4; }
template<> inline bool wdmParse(const char *text, wdmFloat32x2 &v) { return sscanf(text, "[%f,%f]", &v[0],&v[1])==2; }
template<> inline bool wdmParse(const char *text, wdmFloat32x3 &v) { return sscanf(text, "[%f,%f,%f]", &v[0],&v[1],&v[2])==3; }
template<> inline bool wdmParse(const char *text, wdmFloat32x4 &v) { return sscanf(text, "[%f,%f,%f,%f]", &v[0],&v[1],&v[2],&v[3])==4; }

template<> inline size_t wdmToS(char *text, size_t len, wdmInt32x2   v) { return wdmSNPrintf(text, len, "[%d,%d]", v[0],v[1]); }
template<> inline size_t wdmToS(char *text, size_t len, wdmInt32x3   v) { return wdmSNPrintf(text, len, "[%d,%d,%d]", v[0],v[1],v[2]); }
template<> inline size_t wdmToS(char *text, size_t len, wdmInt32x4   v) { return wdmSNPrintf(text, len, "[%d,%d,%d,%d]", v[0],v[1],v[2],v[3]); }
template<> inline size_t wdmToS(char *text, size_t len, wdmFloat32x2 v) { return wdmSNPrintf(text, len, "[%f,%f]", v[0],v[1]); }
template<> inline size_t wdmToS(char *text, size_t len, wdmFloat32x3 v) { return wdmSNPrintf(text, len, "[%f,%f,%f]", v[0],v[1],v[2]); }
template<> inline size_t wdmToS(char *text, size_t len, wdmFloat32x4 v) { return wdmSNPrintf(text, len, "[%f,%f,%f,%f]", v[0],v[1],v[2],v[3]); }

template<> struct wdmCanBeRanged<char>   { static const bool value=false; };
template<> struct wdmCanBeRanged<wchar_t>{ static const bool value=false; };


// SSE
#if defined(_INCLUDED_MM2) || defined(_XMMINTRIN_H_INCLUDED)
template<> inline const char* wdmTypename<__m128i>() { return wdmTypename<wdmInt32x4  >(); }
template<> inline const char* wdmTypename<__m128>()  { return wdmTypename<wdmFloat32x4>(); }
template<> inline bool wdmParse(const char *text, __m128i &v) { return wdmParse<wdmInt32x4  >(text, (wdmInt32x4&)v); }
template<> inline bool wdmParse(const char *text, __m128 &v)  { return wdmParse<wdmFloat32x4>(text, (wdmFloat32x4&)v); }
template<> inline size_t wdmToS(char *text, size_t len, __m128i v) { return wdmToS<wdmInt32x4  >(text, len, (const wdmInt32x4&)v); }
template<> inline size_t wdmToS(char *text, size_t len, __m128 v)  { return wdmToS<wdmFloat32x4>(text, len, (const wdmFloat32x4&)v); }
#endif // defined(_INCLUDED_MM2) || defined(_XMMINTRIN_H_INCLUDED)

// glm
#ifdef glm_glm
template<> inline const char* wdmTypename<glm::ivec2>() { return wdmTypename<wdmInt32x2>(); }
template<> inline const char* wdmTypename<glm::ivec3>() { return wdmTypename<wdmInt32x3>(); }
template<> inline const char* wdmTypename<glm::ivec4>() { return wdmTypename<wdmInt32x4>(); }
template<> inline const char* wdmTypename<glm::vec2 >() { return wdmTypename<wdmFloat32x2>(); }
template<> inline const char* wdmTypename<glm::vec3 >() { return wdmTypename<wdmFloat32x3>(); }
template<> inline const char* wdmTypename<glm::vec4 >() { return wdmTypename<wdmFloat32x4>(); }
template<> inline bool wdmParse(const char *text, glm::ivec2 &v) { return wdmParse<wdmInt32x2  >(text, (wdmInt32x2&)v); }
template<> inline bool wdmParse(const char *text, glm::ivec3 &v) { return wdmParse<wdmInt32x3  >(text, (wdmInt32x3&)v); }
template<> inline bool wdmParse(const char *text, glm::ivec4 &v) { return wdmParse<wdmInt32x4  >(text, (wdmInt32x4&)v); }
template<> inline bool wdmParse(const char *text, glm::vec2  &v) { return wdmParse<wdmFloat32x2>(text, (wdmFloat32x2&)v); }
template<> inline bool wdmParse(const char *text, glm::vec3  &v) { return wdmParse<wdmFloat32x3>(text, (wdmFloat32x3&)v); }
template<> inline bool wdmParse(const char *text, glm::vec4  &v) { return wdmParse<wdmFloat32x4>(text, (wdmFloat32x4&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::ivec2 v) { return wdmToS<wdmInt32x2  >(text, len, (const wdmInt32x2&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::ivec3 v) { return wdmToS<wdmInt32x3  >(text, len, (const wdmInt32x3&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::ivec4 v) { return wdmToS<wdmInt32x4  >(text, len, (const wdmInt32x4&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::vec2  v) { return wdmToS<wdmFloat32x2>(text, len, (const wdmFloat32x2&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::vec3  v) { return wdmToS<wdmFloat32x3>(text, len, (const wdmFloat32x3&)v); }
template<> inline size_t wdmToS(char *text, size_t len, glm::vec4  v) { return wdmToS<wdmFloat32x4>(text, len, (const wdmFloat32x4&)v); }
#endif // glm_glm
#pragma endregion

// xnamath
#ifdef __XNAMATH_H__
template<> inline const char* wdmTypename<XMFLOAT2>() { return wdmTypename<wdmFloat32x2>(); }
template<> inline const char* wdmTypename<XMFLOAT3>() { return wdmTypename<wdmFloat32x3>(); }
template<> inline const char* wdmTypename<XMFLOAT4>() { return wdmTypename<wdmFloat32x4>(); }
template<> inline bool wdmParse(const char *text, XMFLOAT2 &v) { return wdmParse<wdmFloat32x2>(text, (wdmFloat32x2&)v); }
template<> inline bool wdmParse(const char *text, XMFLOAT3 &v) { return wdmParse<wdmFloat32x3>(text, (wdmFloat32x3&)v); }
template<> inline bool wdmParse(const char *text, XMFLOAT4 &v) { return wdmParse<wdmFloat32x4>(text, (wdmFloat32x4&)v); }
template<> inline size_t wdmToS(char *text, size_t len, XMFLOAT2 v) { return wdmToS<wdmFloat32x2>(text, len, (const wdmFloat32x2&)v); }
template<> inline size_t wdmToS(char *text, size_t len, XMFLOAT3 v) { return wdmToS<wdmFloat32x3>(text, len, (const wdmFloat32x3&)v); }
template<> inline size_t wdmToS(char *text, size_t len, XMFLOAT4 v) { return wdmToS<wdmFloat32x4>(text, len, (const wdmFloat32x4&)v); }
#endif // __XNAMATH_H__

#else // wdmDisable

#define wdmInitialize(...)
#define wdmFinalize(...)
#define wdmOpenBrowser(...)
#define wdmGetConfig(...)
#define wdmFlush(...)
#define wdmScope(...)
#define wdmAddNode(...)
#define wdmEraseNode(...)

#endif // wdmDisable
#endif // WebDebugMenu_h
