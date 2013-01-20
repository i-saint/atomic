#ifndef __ist_Debug_ParamTree__
#define __ist_Debug_ParamTree__

#include "ist/Base/New.h"
#include "ist/Math/Misc.h" // clamp


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
        virtual void        eraseChild(IParamNode *node)=0;

        virtual const char* getName() const=0; // dll 跨ぐ可能性を考えると stl::string は返したくない
        virtual int32       getSelection() const=0;
        virtual bool        isOpened() const=0;

        virtual IParamNode* getParent() const=0;
        virtual uint32      getChildrenCount() const=0;
        virtual IParamNode* getChild(uint32 i) const=0;
        virtual IParamNode* getChildByPath(const char *path) const=0;
        virtual void        addChild(IParamNode *node)=0;
        virtual void        addChildByPath(const char *path, IParamNode *node)=0;

        virtual uint32      printName(char *buf, uint32 buf_size) const=0;
        virtual uint32      printValue(char *buf, uint32 buf_size) const=0;
        virtual bool        handleEvent(EventCode e, OptionCode o=Option_None)=0;
    };


    // ここから下の class はモジュールを跨いではならない。
    // モジュールを跨いでいいのは IParamNode だけとする。

    class istInterModule ParamNodeBase : public IParamNode
    {
    istMakeDestructable;
    public:
        virtual ~ParamNodeBase();
        virtual void release();
        virtual void setName(const char *name, uint32 len=0);
        virtual void setOpened(bool v);
        virtual void setParent(IParamNode *parent);
        virtual void eraseChild(IParamNode *node);

        virtual const char* getName() const;
        virtual int32       getSelection() const;
        virtual bool        isOpened() const;
        virtual IParamNode* getParent() const;
        virtual uint32      getChildrenCount() const;
        virtual IParamNode* getChild(uint32 i) const;
        virtual IParamNode* getChildByPath(const char *path) const;
        virtual void addChild(IParamNode *node);
        virtual void addChildByPath(const char *path, IParamNode *node);


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
        IParamNode *m_parent;
        stl::string m_name;
        int32 m_selection;
        bool m_opened;
    };

    template<class T>
    class istInterModule TParamNode : public ParamNodeBase
    {
    public:
        typedef T ValueT;

        TParamNode() : m_param(NULL), m_min(), m_max(), m_step()
        {}

        void SetValue(ValueT *p, ValueT _min, ValueT _max, ValueT step)
        {
            m_param = p;
            m_min = _min;
            m_max = _max;
            m_step = step;
        }

        ValueT& GetValue() const{ return *m_param; }
        ValueT GetMin() const   { return m_min; }
        ValueT GetMax() const   { return m_max; }
        ValueT GetStep() const  { return m_step; }

        virtual bool handleForward(OptionCode o)
        {
            *m_param = clamp(*m_param+m_step, m_min, m_max);
            return true;
        }

        virtual bool handleBackward(OptionCode o)
        {
            *m_param = clamp(*m_param-m_step, m_min, m_max);
            return true;
        }

    private:
        ValueT *m_param;
        ValueT m_min;
        ValueT m_max;
        ValueT m_step;
    };
    typedef TParamNode<float32>    ParamNodeF32;
    typedef TParamNode<int32>      ParamNodeI32;
    typedef TParamNode<uint32>     ParamNodeU32;

    class ParamNodeBool : public ParamNodeBase
    {
    public:
        ParamNodeBool() : m_param(NULL)
        {}

        void SetValue(bool *p)
        {
            m_param = p;
        }

        virtual bool handleForward(OptionCode o)
        {
            *m_param = true;
            return true;
        }

        virtual bool handleBackward(OptionCode o)
        {
            *m_param = false;
            return true;
        }

    private:
        bool *m_param;
    };


    template<class T, class U, class Base=ParamNodeBase>
    class TMethodParamNode : public Base
    {
    public:
        typedef T MethodT;
        typedef T ClassT;

        TMethodParamNode() : m_method(NULL), m_obj(NULL)
        {}

        virtual bool handleAction()
        {
            (m_obj->*m_method)();
            return true;
        }

    private:
        MethodT m_method;
        ClassT *m_obj;
    };

} // namespace ist
#endif // __ist_Debug_ParamTree__
