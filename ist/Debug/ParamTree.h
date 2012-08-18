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
        enum Event {
            Event_Up,
            Event_Down,
            Event_Forward,
            Event_Backward,
            Event_Action,
            Event_Cancel,
            Event_Focus,
            Event_Defocus,
        };
        enum Option {
            Option_None,
            Option_x10,
            Option_x100,
            Option_x01,
            Option_x001,
        };

    public:
        virtual const char* getName() const=0; // dll 跨ぐ可能性を考えると stl::string は返したくない
        virtual int32 getSelection() const=0;
        virtual bool isOpened() const=0;
        virtual IParamNode* getParent() const=0;
        virtual size_t getChildrenCount() const=0;
        virtual IParamNode* getChild(size_t i) const=0;
        virtual void addChild(IParamNode *node)=0;
        virtual void destroy()=0;
        virtual bool handleEvent(Event e, Option o=Option_None)=0;

        virtual size_t printName(char *buf, size_t buf_size) const=0;
        virtual size_t printValue(char *buf, size_t buf_size) const=0;

    public:
        virtual ~IParamNode() {}
        virtual void setOpened(bool v)=0;
        virtual void setParent(IParamNode *parent)=0;
        virtual void eraseChild(IParamNode *node)=0;
    };


    // ここから下の class はモジュールを跨いではならない。
    // モジュールを跨いでいいのは IParamNode だけとする。

    istInterModule IParamNode* GetChildByPath(IParamNode *node, const char *name);

    class istInterModule ParamNodeBase : public IParamNode
    {
    istMakeDestructable;
    public:
        virtual const char* getName() const         { return m_name.c_str(); }
        virtual int32 getSelection() const          { return m_selection; }
        virtual bool isOpened() const               { return m_opened; }
        virtual IParamNode* getParent() const       { return m_parent; }
        virtual size_t getChildrenCount() const     { return m_children.size(); }
        virtual IParamNode* getChild(size_t i) const{ return m_children[i]; }

        virtual void addChild(IParamNode *node)
        {
            node->setParent(this);
            m_children.push_back(node);
        }

        virtual void destroy()
        {
            getParent()->eraseChild(this);
            istDelete(this);
        }


        virtual bool handleAction(Option o)     { return false; }
        virtual bool handleForward(Option o)    { return false; }
        virtual bool handleBackward(Option o)   { return false; }
        virtual bool handleFocus()              { return false; }
        virtual bool handleDefocus()            { return false; }
        virtual bool handleEvent(Event e, Option o)
        {
            IParamNode *selected=getSelectedItem();
            if(selected) {
                if(selected->isOpened()) { return selected->handleEvent(e); }
            }
            if(isOpened()) {
                switch(e) {
                case Event_Up:
                    if(selected) { selected->handleEvent(Event_Defocus); }
                    m_selection = stl::max<int32>(m_selection-1, 0);
                    if(selected) { selected->handleEvent(Event_Focus); }
                    return true;
                case Event_Down:
                    if(selected) { selected->handleEvent(Event_Defocus); }
                    m_selection = stl::min<int32>(m_selection+1, stl::max<int32>(m_children.size()-1, 0));
                    if(selected) { selected->handleEvent(Event_Focus); }
                    return true;
                case Event_Forward:
                    if(selected) { selected->handleEvent(e, o); }
                    return true;
                case Event_Backward:
                    if(selected) { selected->handleEvent(e, o); }
                    return true;
                case Event_Action:
                    if(selected) { selected->handleEvent(e, o); }
                    return true;
                case Event_Cancel:
                    setOpened(false);
                    return true;
                }
            }
            else {
                switch(e) {
                case Event_Forward: return handleForward(o);
                case Event_Backward:return handleBackward(o);
                case Event_Focus:   return handleFocus();
                case Event_Defocus: return handleDefocus();
                case Event_Action:
                    if(!m_children.empty()) { setOpened(true); }
                    return handleAction(o);
                }
            }
            return false;
        }

        virtual size_t printValue(char *buf, size_t buf_size) const {}

        IParamNode* getSelectedItem()
        {
            if(m_selection<(int32)m_children.size()) {
                return m_children[m_selection];
            }
            return NULL;
        }

    protected:
        virtual ~ParamNodeBase();
        virtual void setOpened(bool v)              { m_opened=v; }
        virtual void setParent(IParamNode *parent)  { m_parent=parent; }
        virtual void eraseChild(IParamNode *node)   { m_children.erase(stl::find(m_children.begin(), m_children.end(), node)); }

    private:
        stl::vector<IParamNode*> m_children;
        IParamNode *m_parent;
        stl::string m_name;
        int32 m_selection;
        bool m_opened;
    };

    template<class T>
    class istInterModule ArithmeticParamNode : public ParamNodeBase
    {
    public:
        typedef T ValueT;

        ArithmeticParamNode() : m_param(NULL), m_min(), m_max(), m_step()
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

        virtual bool handleForward(Option o)
        {
            *m_param = clamp(*m_param+m_step, m_min, m_max);
            return true;
        }

        virtual bool handleBackward(Option o)
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
    typedef ArithmeticParamNode<float32>    Float32ParamNode;
    typedef ArithmeticParamNode<int32>      Int32ParamNode;
    typedef ArithmeticParamNode<uint32>     Uint32ParamNode;

    class BoolParamNode : public ParamNodeBase
    {
    public:
        BoolParamNode() : m_param(NULL)
        {}

        void SetValue(bool *p)
        {
            m_param = p;
        }

        virtual bool handleForward(Option o)
        {
            *m_param = true;
            return true;
        }

        virtual bool handleBackward(Option o)
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
