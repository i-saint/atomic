#ifndef __atomic_Game_Entity__
#define __atomic_Game_Entity__


#define atomicImplementEntity(class_name, category_id, class_id)\
class class_name;                                               \
template<> struct EntityTraits<class_name>                      \
{                                                               \
    enum {                                                      \
    CATEGORY_ID = category_id,                                  \
    CLASS_ID = class_id,                                        \
    };                                                          \
};                                                              \
template<> class_name* EntitySet::createEntity<class_name>()    \
{                                                               \
    class_name *t = IST_NEW(class_name)();                      \
    typedef EntityTraits<class_name> traits;                    \
    addEntity(traits::CATEGORY_ID, traits::CLASS_ID, t);        \
    return t;                                                   \
}



namespace atomic {

    enum ENTITY_CATEGORY_ID
    {
        ECID_UNKNOWN,
        ECID_PLAYER,
        ECID_ENEMY,
        ECID_OBSTRUCT,
        ECID_BULLET,
        ECID_VFX,

        ECID_END,
    };

    enum ENTITY_PLAYER_CLASS_ID
    {
        ESID_PLAYER,
        ESID_PLAYER_END,
    };

    enum ENTITY_ENEMY_CLASS_ID
    {
        ESID_ENEMY_CUBE,
        ESID_ENEMY_SPHERE,
        ESID_ENEMY_END,
    };

    enum ENTITY_OBSTACLE_CLASS_ID
    {
        ESID_OBSTACLE_CUBE,
        ESID_OBSTACLE_SPHERE,
        ESID_OBSTACLE_END,
    };

    enum ENTITY_BULLET_CLASS_ID
    {
        ESID_BULLET,
        ESID_BULLET_END,
    };

    enum ENTITY_VFX_CLASS_ID
    {
        ESID_VFX_END,
    };

    enum {
        ESID_MAX = 16
    };
    BOOST_STATIC_ASSERT(ESID_MAX >= ESID_PLAYER_END);
    BOOST_STATIC_ASSERT(ESID_MAX >= ESID_ENEMY_END);
    BOOST_STATIC_ASSERT(ESID_MAX >= ESID_OBSTACLE_END);
    BOOST_STATIC_ASSERT(ESID_MAX >= ESID_BULLET_END);
    BOOST_STATIC_ASSERT(ESID_MAX >= ESID_VFX_END);


    class IEntity;
    template<class T> struct EntityTraits;


    // EntityHandle: 上位 4 bit がカテゴリ (ENTITY_CATEGORY_ID)、その次 8 bit がカテゴリ内種別 (ENTITY_*_CLASS_ID)、それ以下は ID のフィールド
    typedef uint32 EntityHandle;
    inline uint32 EntityGetCategory(EntityHandle e) { return (e & 0xF0000000) >> 28; }
    inline uint32 EntityGetClass(EntityHandle e)    { return (e & 0x0FF00000) >> 20; }
    inline uint32 EntityGetID(EntityHandle e)       { return (e & 0x000FFFFF) >>  0; }
    inline uint32 EntityCreateHandle(uint32 cid, uint32 sid, uint32 id) { return (cid<<28) | (sid<<20) | id; }



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
        uint32      getHandle() const   { return m_ehandle; }

        virtual void initialize() {}
        virtual void finalize() {}
        virtual void update(float32 dt)=0;
        virtual void updateAsync(float32 dt)=0;
        virtual void draw()=0;

        virtual bool call(uint32 call_id, const variant &v) { return false; }
        virtual bool query(uint32 query_id, variant &v) const { return false; }
    };

    class EntitySet
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

        void update(float32 dt);
        void sync();
        void draw();

        IEntity* getEntity(EntityHandle h);
        void deleteEntity(EntityHandle h);
        template<class T> T* createEntity();
    };



} // namespace atomic
#endif // __atomic_Game_Entity__
