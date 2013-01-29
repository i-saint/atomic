#ifndef __ist_Debug_ParamTree__
#define __ist_Debug_ParamTree__

#include "ist/Base/New.h"
#include "ist/Math/Misc.h" // clamp

#include <functional>


namespace ist {



class IParamNode;

class istInterModule IParamNode
{
istMakeDestructable;
public:
    enum EventCode {
        Event_Up,
        Event_Down,
        Event_Forward,
        Event_Backward,
        Event_Action,
        Event_Cancel,
        Event_Focus,
        Event_Defocus,
    };
    enum OptionCode {
        Option_None,
        Option_x10,
        Option_x100,
        Option_x01,
        Option_x001,
    };

public:
    virtual ~IParamNode() {}
    virtual void        release() { istDelete(this); }
    virtual void        setName(const char *name, uint32 len=0)=0;
    virtual void        setOpened(bool v)=0;
    virtual void        setParent(IParamNode *parent)=0;

    virtual const char* getName() const=0; // dll 跨ぐ可能性を考えると stl::string は返したくない
    virtual int32       getSelection() const=0;
    virtual bool        isOpened() const=0;
    virtual bool        isSelected() const;

    virtual IParamNode* getParent() const=0;
    virtual uint32      getChildrenCount() const=0;
    virtual IParamNode* getChild(uint32 i) const=0;
    virtual IParamNode* getChildByPath(const char *path) const=0;
    virtual void        addChild(IParamNode *node)=0;
    virtual void        addChildByPath(const char *path, IParamNode *node)=0;
    virtual void        eraseChild(IParamNode *node)=0;
    virtual void        releaseChildByPath(const char *path)=0;

    virtual uint32      printName(char *buf, uint32 buf_size) const=0;
    virtual uint32      printValue(char *buf, uint32 buf_size) const=0;
    virtual bool        handleEvent(EventCode e, OptionCode o=Option_None)=0;
};


// ここから下の class はモジュールを跨いではならない。
// モジュールを跨いでいいのは IParamNode だけとする。

template<class T> uint32 TPrintValue(char *buf, uint32 buf_size, T value);
template<> uint32 TPrintValue<float32>(char *buf, uint32 buf_size, float32 value);
template<> uint32 TPrintValue<float64>(char *buf, uint32 buf_size, float64 value);
template<> uint32 TPrintValue<  int8>(char *buf, uint32 buf_size,   int8 value);
template<> uint32 TPrintValue< uint8>(char *buf, uint32 buf_size,  uint8 value);
template<> uint32 TPrintValue< int16>(char *buf, uint32 buf_size,  int16 value);
template<> uint32 TPrintValue<uint16>(char *buf, uint32 buf_size, uint16 value);
template<> uint32 TPrintValue< int32>(char *buf, uint32 buf_size,  int32 value);
template<> uint32 TPrintValue<uint32>(char *buf, uint32 buf_size, uint32 value);
template<> uint32 TPrintValue< int64>(char *buf, uint32 buf_size,  int64 value);
template<> uint32 TPrintValue<uint64>(char *buf, uint32 buf_size, uint64 value);
template<> uint32 TPrintValue<  bool>(char *buf, uint32 buf_size,   bool value);
template<class T> class ITValueUpdater;
class INodeFunctor;


// ParamNodeBase から呼ばれる関数オブジェクト的なもの
class INodeFunctor
{
public:
    virtual ~INodeFunctor() {}
    virtual void release() { istDelete(this); }
    virtual void exec()=0;
};

// TParamNode から操作され、実際に値を変更する役割を担う
// 値はポインタ経由で直接弄るかもしれないし、Hoge::getValue()/Hoge::setValue() のようなメンバ関数経由で弄るかもしれない。
// そのへんの差異をこの class が吸収する。
template<class T>
class ITValueUpdater
{
public:
    virtual ~ITValueUpdater() {}
    virtual void release() { istDelete(this); }
    virtual T getValue() const=0;
    virtual void setValue(T v)=0;
};


class ParamNodeBase : public IParamNode
{
istMakeDestructable;
public:
    ParamNodeBase(const char *name="", INodeFunctor *functor=NULL);
    virtual ~ParamNodeBase();
    virtual void release();
    virtual void setName(const char *name, uint32 len=0);
    virtual void setFunctor(INodeFunctor *func);
    virtual void setOpened(bool v);
    virtual void setParent(IParamNode *parent);

    virtual const char* getName() const;
    virtual INodeFunctor* getFunctor() const;
    virtual int32       getSelection() const;
    virtual bool        isOpened() const;
    virtual IParamNode* getParent() const;
    virtual uint32      getChildrenCount() const;
    virtual IParamNode* getChild(uint32 i) const;
    virtual IParamNode* getChildByPath(const char *path) const;
    virtual void addChild(IParamNode *node);
    virtual void addChildByPath(const char *path, IParamNode *node);
    virtual void eraseChild(IParamNode *node);
    virtual void releaseChildByPath(const char *path);

    virtual bool handleAction(OptionCode o);
    virtual bool handleForward(OptionCode o);
    virtual bool handleBackward(OptionCode o);
    virtual bool handleFocus();
    virtual bool handleDefocus();
    virtual bool handleEvent(EventCode e, OptionCode o);

    virtual uint32 printName(char *buf, uint32 buf_size) const;
    virtual uint32 printValue(char *buf, uint32 buf_size) const;

    IParamNode* getSelectedItem();

private:
    stl::vector<IParamNode*> m_children;
    stl::string m_name;
    IParamNode *m_parent;
    INodeFunctor *m_functor;
    int32 m_selection;
    bool m_opened;
};


template<class T>
class istInterModule TParamNode : public ParamNodeBase
{
typedef ParamNodeBase super;
public:
    typedef T ValueT;
    typedef ITValueUpdater<T> UpdaterT;

    TParamNode() : m_param(NULL), m_min(), m_max(), m_step()
    {}

    TParamNode(const char *name, UpdaterT *p, ValueT _min, ValueT _max, ValueT step, INodeFunctor *functor=NULL)
        : super(name, functor), m_param(NULL), m_min(), m_max(), m_step()
    {
        setUpdater(p, _min, _max, step);
    }

    ~TParamNode()
    {
        if(m_param) { m_param->release(); }
    }

    void setUpdater(UpdaterT *p, ValueT _min, ValueT _max, ValueT step)
    {
        m_param = p;
        m_min = _min;
        m_max = _max;
        m_step = step;
    }

    void setValue(ValueT v) { m_param->setValue(v); }
    ValueT getValue() const { return m_param->getValue(); }
    ValueT getMin() const   { return m_min; }
    ValueT getMax() const   { return m_max; }
    ValueT getStep() const  { return m_step; }

    virtual bool handleForward(OptionCode o)
    {
        setValue(clamp(getValue()+getStep(), getMin(), getMax()));
        return true;
    }

    virtual bool handleBackward(OptionCode o)
    {
        setValue(clamp(getValue()-getStep(), getMin(), getMax()));
        return true;
    }

    virtual uint32 printValue(char *buf, uint32 buf_size) const
    {
        return TPrintValue(buf, buf_size, getValue());
    }

private:
    UpdaterT *m_param;
    ValueT m_min;
    ValueT m_max;
    ValueT m_step;
};
typedef TParamNode<float32> ParamNodeF32;
typedef TParamNode<float64> ParamNodeF64;
typedef TParamNode<int8>    ParamNodeI8;
typedef TParamNode<uint8>   ParamNodeU8;
typedef TParamNode<int16>   ParamNodeI16;
typedef TParamNode<uint16>  ParamNodeU16;
typedef TParamNode<int32>   ParamNodeI32;
typedef TParamNode<uint32>  ParamNodeU32;
typedef TParamNode<int64>   ParamNodeI64;
typedef TParamNode<uint64>  ParamNodeU64;

template<>
class istInterModule TParamNode<bool> : public ParamNodeBase
{
typedef ParamNodeBase super;
public:
    typedef bool ValueT;
    typedef ITValueUpdater<bool> UpdaterT;

    TParamNode() : m_param(NULL)
    {}

    TParamNode(const char *name, UpdaterT *p, INodeFunctor *functor=NULL)
        : super(name, functor), m_param(NULL)
    {
        setUpdater(p);
    }

    ~TParamNode()
    {
        istSafeRelease(m_param);
    }

    virtual void setUpdater(UpdaterT *p)
    {
        m_param = p;
    }

    void setValue(ValueT v) { m_param->setValue(v); }
    ValueT getValue() const { return m_param->getValue(); }

    virtual bool handleForward(OptionCode o)
    {
        setValue(true);
        return true;
    }

    virtual bool handleBackward(OptionCode o)
    {
        setValue(false);
        return true;
    }

    virtual uint32 printValue(char *buf, uint32 buf_size) const
    {
        return TPrintValue(buf, buf_size, getValue());
    }

private:
    UpdaterT *m_param;
};
typedef TParamNode<bool>    ParamNodeBool;



// 関数オブジェクトをそのまま呼ぶ
// ex: TParamNodeFunction(std::bind(&Hoge::doSomething, hoge))
template<class Func=std::function<void ()>>
class TNodeFunctor : public INodeFunctor
{
public:
    TNodeFunctor(Func v) : m_func(v) {}
    virtual void exec() { m_func(); }
private:
    Func m_func;
};


// ポインタを直接弄る系
template<class T>
class TValueUpdaterP : public ITValueUpdater<T>
{
public:
    TValueUpdaterP(T *value) : m_value(value) {}
    virtual T getValue() const { return *m_value; }
    virtual void setValue(T v) { *m_value=v; }
private:
    T *m_value;
};

// メンバ関数で弄る系
// TValueUpdaterMF(std::bind(&Hoge::getValue, hoge), std::bind(&Hoge::setValue, hoge, std::placeholders::_1) )
// みたいに生成するが、そのままだとめんどくさすぎるので便利系マクロを用意して使うことになると思われる。
template<class T, class Getter=std::function<T ()>, class Setter=std::function<void (T)>>
class TValueUpdaterM : public ITValueUpdater<T>
{
public:
    TValueUpdaterM(Getter getter, Setter setter) : m_getter(getter), m_setter(setter) {}
    virtual T getValue() const { return m_getter(); }
    virtual void setValue(T v) { m_setter(v); }
private:
    Getter m_getter;
    Setter m_setter;
};

} // namespace ist
#endif // __ist_Debug_ParamTree__
