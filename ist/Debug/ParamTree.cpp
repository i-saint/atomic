#include "istPCH.h"
#include "ist/Base.h"
#include "ParamTree.h"

namespace ist {


template<> uint32 TPrintValue<float32>(char *buf, uint32 buf_size, float32 value)   { return istsnprintf(buf, buf_size, "%.3f",  value); }
template<> uint32 TPrintValue<float64>(char *buf, uint32 buf_size, float64 value)   { return istsnprintf(buf, buf_size, "%.3lf", value); }
template<> uint32 TPrintValue<  int8>(char *buf, uint32 buf_size,   int8 value)     { return istsnprintf(buf, buf_size, "%d", static_cast< int32>(value)); }
template<> uint32 TPrintValue< uint8>(char *buf, uint32 buf_size,  uint8 value)     { return istsnprintf(buf, buf_size, "%u", static_cast<uint32>(value)); }
template<> uint32 TPrintValue< int16>(char *buf, uint32 buf_size,  int16 value)     { return istsnprintf(buf, buf_size, "%d", static_cast< int32>(value)); }
template<> uint32 TPrintValue<uint16>(char *buf, uint32 buf_size, uint16 value)     { return istsnprintf(buf, buf_size, "%u", static_cast<uint32>(value)); }
template<> uint32 TPrintValue< int32>(char *buf, uint32 buf_size,  int32 value)     { return istsnprintf(buf, buf_size, "%d", value); }
template<> uint32 TPrintValue<uint32>(char *buf, uint32 buf_size, uint32 value)     { return istsnprintf(buf, buf_size, "%u", value); }
template<> uint32 TPrintValue< int64>(char *buf, uint32 buf_size,  int64 value)     { return istsnprintf(buf, buf_size, "%lld", value); }
template<> uint32 TPrintValue<uint64>(char *buf, uint32 buf_size, uint64 value)     { return istsnprintf(buf, buf_size, "%llu", value); }
template<> uint32 TPrintValue<  bool>(char *buf, uint32 buf_size,   bool value)     { return istsnprintf(buf, buf_size, "%s", (value ? "true" : "false")); }


bool IParamNode::isSelected() const
{
    if(IParamNode *parent = getParent()) {
        return parent->getChild(parent->getSelection())==this;
    }
    return false;
}


ParamNodeBase::ParamNodeBase(const char *name, INodeFunctor *functor)
    : m_parent(NULL)
    , m_functor(functor)
    , m_selection(0)
    , m_opened(false)
{
    setName(name);
}

ParamNodeBase::~ParamNodeBase()
{
    istSafeRelease(m_functor);
    for(size_t i=0; i<m_children.size(); ++i) {
        istSafeRelease(m_children[i]);
    }
    if(IParamNode *parent=getParent()) {
        parent->eraseChild(this);
    }
}

void ParamNodeBase::release()
{
    istDelete(this);
}

void ParamNodeBase::setName( const char *name, uint32 len )
{
    if(len==0) {
        m_name = name;
    }
    else {
        m_name = stl::string(name, len);
    }
}

void ParamNodeBase::setFunctor( INodeFunctor *func )    { m_functor=func; }
void ParamNodeBase::setOpened(bool v)                   { if(v==false && !getParent()) { return; } m_opened=v; }
void ParamNodeBase::setParent(IParamNode *parent)       { m_parent=parent; }

const char*     ParamNodeBase::getName() const          { return m_name.c_str(); }
INodeFunctor*   ParamNodeBase::getFunctor() const       { return m_functor; }
int32           ParamNodeBase::getSelection() const     { return m_selection; }
bool            ParamNodeBase::isOpened() const         { return m_opened; }
IParamNode*     ParamNodeBase::getParent() const        { return m_parent; }
uint32          ParamNodeBase::getChildrenCount() const { return static_cast<uint32>(m_children.size()); }
IParamNode*     ParamNodeBase::getChild(uint32 i) const { return i<m_children.size() ? m_children[i] : NULL; }

IParamNode* ParamNodeBase::getChildByPath( const char *path ) const
{
    uint32 len = 0;
    bool leaf = false;
    for(;;) {
        if(path[len]=='\0') { leaf=true; break; }
        if(path[len]=='/')  { break; }
        ++len;
    }
    for(size_t i=0; i<m_children.size(); ++i) {
        if(strncmp(path, m_children[i]->getName(), len)==0) {
            return leaf ? m_children[i] : m_children[i]->getChildByPath(path+(len+1));
        }
    }
    return NULL;
}

void ParamNodeBase::addChild(IParamNode *node)
{
    node->setParent(this);
    m_children.push_back(node);
}

void ParamNodeBase::addChildByPath( const char *path, IParamNode *node )
{
    uint32 len = 0;
    bool leaf = false;
    for(;;) {
        if(path[len]=='\0') { leaf=true; break; }
        if(path[len]=='/')  { break; }
        ++len;
    }

    if(leaf) {
        node->setName(path);
        addChild(node);
    }
    else {
        for(size_t i=0; i<m_children.size(); ++i) {
            if(strncmp(path, m_children[i]->getName(), len)==0) {
                m_children[i]->addChildByPath(path+(len+1), node);
                return;
            }
        }

        // 枝が見つからなかったので追加
        ParamNodeBase *n = istNew(ParamNodeBase);
        n->setName(path, len);
        addChild(n);
        n->addChildByPath(path+(len+1), node);
    }
}

void ParamNodeBase::eraseChild(IParamNode *node)
{
    m_children.erase(stl::find(m_children.begin(), m_children.end(), node));
}

void ParamNodeBase::releaseChildByPath(const char *path)
{
    uint32 len = 0;
    bool leaf = false;
    for(;;) {
        if(path[len]=='\0') { leaf=true; break; }
        if(path[len]=='/')  { break; }
        ++len;
    }
    for(size_t i=0; i<m_children.size(); ++i) {
        if(strncmp(path, m_children[i]->getName(), len)==0) {
            if(leaf) { m_children[i]->release(); }
            else { m_children[i]->releaseChildByPath(path+(len+1)); }
        }
    }
}


bool ParamNodeBase::handleAction(OptionCode o)
{
    if(m_functor) {
        m_functor->exec();
        return true;
    }
    return false;
}
bool ParamNodeBase::handleForward(OptionCode o)    { return false; }
bool ParamNodeBase::handleBackward(OptionCode o)   { return false; }
bool ParamNodeBase::handleFocus()                  { return false; }
bool ParamNodeBase::handleDefocus()                { return false; }
bool ParamNodeBase::handleEvent(EventCode e, OptionCode o)
{
    IParamNode *selected = getSelectedItem();
    if(selected) {
        if(selected->isOpened()) { return selected->handleEvent(e, o); }
    }
    if(isOpened()) {
        switch(e) {
        case Event_Up:
            {
                int32 next_s = std::max<int32>(m_selection-1, 0);
                IParamNode *next = getChild(next_s);
                if(next && selected!=next) {
                    m_selection = next_s;
                    selected->handleEvent(Event_Defocus);
                    next->handleEvent(Event_Focus);
                }
                return true;
            }
        case Event_Down:
            {
                int32 next_s = std::min<int32>(m_selection+1, m_children.size());
                IParamNode *next = getChild(next_s);
                if(next && selected!=next) {
                    m_selection = next_s;
                    selected->handleEvent(Event_Defocus);
                    next->handleEvent(Event_Focus);
                }
                return true;
            }
        case Event_Forward:
            {
                if(selected) { selected->handleEvent(e, o); }
                return true;
            }
        case Event_Backward:
            {
                if(selected) { selected->handleEvent(e, o); }
                return true;
            }
        case Event_Action:
            {
                if(selected) { selected->handleEvent(e, o); }
                return true;
            }
        case Event_Cancel:
            {
                setOpened(false);
                return true;
            }
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

uint32 ParamNodeBase::printName( char *buf, uint32 buf_size ) const
{
    return istsnprintf(buf, buf_size, "%s", getName());
}

uint32 ParamNodeBase::printValue(char *buf, uint32 buf_size) const
{
    return istsnprintf(buf, buf_size, "");
}

IParamNode* ParamNodeBase::getSelectedItem()
{
    if(m_selection < static_cast<int32>(m_children.size())) {
        return m_children[m_selection];
    }
    return NULL;
}


} // namespace ist
