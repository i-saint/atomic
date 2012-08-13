#ifndef __ist_Debug_ParamTree__
#define __ist_Debug_ParamTree__

#include "ist/Base/SharedObject.h"

namespace ist {

    class IDebugParamNode;
    typedef boost::intrusive_ptr<IDebugParamNode> IDebugParamNodePtr;

    class istInterModule IDebugParamNode : public SharedObject
    {
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

        virtual ~IDebugParamNode() {}
        virtual const char* getName() const=0; // dll 跨ぐ可能性を考えると std::string は返したくない
        virtual bool handleEvent(Event e)=0;
    };


    // ここから下の class は他のモジュールに見せてはならない

    class DebugParamNodeBase : public IDebugParamNode
    {
    public:
        virtual bool handleEvent(Event e);

        const char* getName() const { return m_name.c_str(); }

    private:
        std::vector<IDebugParamNodePtr> m_children;
        std::string m_name;
        int32 m_selection;
        bool m_is_toplevel;
    };


    template<class T>
    T clamp(T v, T minmum, T maximum)
    {
        return std::min<T>(std::max<T>(v, minmum), maximum);
    }

    template<class T>
    class ArithmeticParamNode : public DebugParamNodeBase
    {
        typedef DebugParamNodeBase super;
    public:
        typedef T ValueT;

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

        virtual bool handleEvent(Event e)
        {
            switch(e) {
            case Event_Forward:
                *m_param = clamp(*m_param+m_step, m_min, m_max);
                return true;
            case Event_Backward:
                *m_param = clamp(*m_param-m_step, m_min, m_max);
                return true;
            default: return super::handleEvent(e);
            }
            return false;
        }

    private:
        ValueT *m_param;
        ValueT m_min;
        ValueT m_max;
        ValueT m_step;
    };
    typedef ArithmeticParamNode<float32>    FloatParamNode;
    typedef ArithmeticParamNode<int32>      Int32ParamNode;
    typedef ArithmeticParamNode<uint32>     Uint32ParamNode;


    template<class T, class U, class Base=DebugParamNodeBase>
    class MethodParamNode : public Base
    {
    public:
        typedef T MethodT;
        typedef T ClassT;
        virtual bool handleEvent(Event e)
        {
            switch(e) {
            case Event_Action:
                (m_obj->*m_method)();
                return true;
            default: return super::handleEvent(e);
            }
            return false;
        }


    private:
        MethodT m_method;
        ClassT *m_obj;
    };

} // namespace ist
#endif // __ist_Debug_ParamTree__
