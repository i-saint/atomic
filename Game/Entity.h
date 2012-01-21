#ifndef __atomic_Game_Entity__
#define __atomic_Game_Entity__

#include "EntityClass.h"

#define atomicImplementEntity(class_name, category_id, class_id)\
class class_name;                                               \
template<> struct EntityTraits<class_name>                      \
{                                                               \
    enum {                                                      \
    CATEGORY_ID = category_id,                                  \
    CLASS_ID = class_id,                                        \
    };                                                          \
};                                                              \
template<> IEntity* EntitySet::createEntity<class_name>()    \
{                                                               \
    class_name *t = istNew(class_name)();                      \
    typedef EntityTraits<class_name> traits;                    \
    addEntity(traits::CATEGORY_ID, traits::CLASS_ID, t);        \
    return t;                                                   \
}



namespace atomic {


class IEntity;
template<class T> struct EntityTraits;




class IEntity
{
friend class EntitySet;
private:
    EntityHandle m_ehandle;

    bool isValidHandle(EntityHandle h);
    void setHandle(uint32 h) { m_ehandle=h; }

public:
    // コンストラクタではメンバ変数初期化以外の処理を行なってはいけない。他は initialize() で行う。
    // (ID がコンストラクタの後に決まるため、子オブジェクトの処理順などを適切に行うにはこうする必要がある)
    IEntity() : m_ehandle(0) {}
    virtual ~IEntity() {}
    uint32 getHandle() const    { return m_ehandle; }
    virtual void initialize()   {}
    virtual void finalize()     {}

    virtual void update(float32 dt)=0;
    virtual void asyncupdate(float32 dt){}
    virtual void draw()                 {}

    virtual bool call(uint32 call_id, const variant &v) { return false; }
    virtual bool query(uint32 query_id, variant &v) const { return false; }
};

class EntitySet : public IAtomicGameModule
{
public:
    typedef stl::vector<EntityHandle> HandleCont;
    typedef stl::vector<IEntity*> EntityCont;

private:
    HandleCont m_vacant[ECID_END][ESID_MAX];
    EntityCont m_entities[ECID_END][ESID_MAX];
    EntityCont m_new_entities;
    HandleCont m_all;

    void addEntity(uint32 categoryid, uint32 classid, IEntity *e);

public:
    EntitySet();
    ~EntitySet();

    void frameBegin();
    void update(float32 dt);
    void asyncupdate(float32 dt);
    void draw();
    void frameEnd();

    IEntity* getEntity(EntityHandle h);
    void deleteEntity(EntityHandle h);
    template<class T> IEntity* createEntity();
};



} // namespace atomic
#endif // __atomic_Game_Entity__
