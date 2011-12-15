#ifndef __atomic_Game_Character__
#define __atomic_Game_Character__

namespace atomic {

    // EntityHandle: 上位 4 bit がカテゴリ、その次 8 bit がカテゴリ内の種別、それ以下は ID のフィールド
    enum {
        ENTITY_PLAYER   = 0,
        ENTITY_ENEMY    = 1,
        ENTITY_OBSTRUCT = 2,
        ENTITY_BULLET   = 3,
    };
    typedef uint32 EntityHandle;
    inline uint32 EntityGetCategory(EntityHandle e) { return (e & 0xF0000000) >> 28; }
    inline uint32 EntityGetSpecies(EntityHandle e)  { return (e & 0x0FF00000) >> 20; }
    inline uint32 EntityGetID(EntityHandle e)       { return (e & 0x000FFFFF) >>  0; }


    enum {
        CID_PLAYER_BEGIN,
        CID_PLAYER,
        CID_PLAYER_END,

        CID_ENEMY_BEGIN = CID_PLAYER_END,
        CID_ENEMY_CUBE,
        CID_ENEMY_SPHERE,
        CID_ENEMY_END,

        CID_OBSTACLE_BEGIN = CID_ENEMY_END,
        CID_OBSTACLE_CUBE,
        CID_OBSTACLE_SPHERE,
        CID_OBSTACLE_END,

        CID_BULLET_BEGIN = CID_OBSTACLE_END,
        CID_BULLET,
        CID_BULLET_END,

        CID_ROUTINE_BEGIN = CID_BULLET_END,
        CID_ROUTINE_HOMING_PLAYER,
        CID_ROUTINE_END,

        CID_END = CID_ROUTINE_END,
    };

    class Entity;
    class Routine;


    class Routine
    {
    public:
        virtual void update(float32 dt);
        virtual void updateAsync(float32 dt);
    };


    class CharacterSet
    {
    private:
    public:
    };

    class Entity
    {
        EntityHandle m_ehandle;

    protected:
        void setHandle(uint32 h) { m_ehandle=h; }

    public:
        Entity() : m_ehandle(0) {}
        uint32      getHandle() const   { return m_ehandle; }

        virtual void update(float32 dt)=0;
        virtual void updateAsync(float32 dt)=0;
    };

    class Enemy_Base : public Entity
    {
    private:
        mat4    m_transform;
        Routine *m_routine;
        uint32  m_handle;
        float32 m_health;

    public:
        Enemy_Base() : m_transform(), m_routine(NULL), m_handle(0), m_health(1.0f) {}

        virtual uint32 getClassID() const=0;

        float32     getHealth() const   { return m_health; }
        const mat4& getTransform() const{ return m_transform; }
        Routine*    getRoutine()        { return m_routine; }

        void setHelth(float32 v)        { m_health=v; }
        void setTransform(const mat4& v){ m_transform=v; }
        void setRoutine(Routine *v)     { m_routine=v; }

        virtual void update(float32 dt)
        {
            if(m_routine) { m_routine->update(dt); }
        }

        virtual void updateAsync(float32 dt)
        {
            if(m_routine) { m_routine->updateAsync(dt); }
        }
    };

    class Attr_DoubleAxisRotation
    {
    private:
        vec4 m_pos;
        vec4 m_scale;
        vec4 m_axis1;
        vec4 m_axis2;
        float32 m_rot1;
        float32 m_rot2;

    public:
        Attr_DoubleAxisRotation() : m_rot1(0.0f), m_rot2(0.0f) {}

        const vec4& getPosition() const { return m_pos; }
        const vec4& getScale() const    { return m_scale; }
        const vec4& getAxis1() const    { return m_axis1; }
        const vec4& getAxis2() const    { return m_axis2; }
        float32 getRotation1() const    { return m_rot1; }
        float32 getRotation2() const    { return m_rot2; }

        void setPosition(const vec4& v) { m_pos=v; }
        void setScale(const vec4& v)    { m_scale=v; }
        void setAxis1(const vec4& v)    { m_axis1=v; }
        void setAxis2(const vec4& v)    { m_axis2=v; }
        void setRotation1(float32 v)    { m_rot1=v; }
        void setRotation2(float32 v)    { m_rot2=v; }

        mat4 computeMatrix()
        {
            mat4 mat;
            mat = glm::translate(mat, reinterpret_cast<const vec3&>(m_pos));
            mat = glm::rotate(mat, m_rot2, reinterpret_cast<const vec3&>(m_axis2));
            mat = glm::rotate(mat, m_rot1, reinterpret_cast<const vec3&>(m_axis1));
            mat = glm::scale(mat, reinterpret_cast<const vec3&>(m_scale));
        }
    };


    class Enemy_Cube : public Enemy_Base, public Attr_DoubleAxisRotation
    {
    typedef Enemy_Base super;
    typedef Attr_DoubleAxisRotation transform;
    private:
    public:
        virtual uint32 getClassID() const { return CID_ENEMY_CUBE; }

        virtual void updateAsync(float32 dt)
        {
            super::updateAsync(dt);
            setTransform(computeMatrix());
        }
    };

    class Enemy_Sphere : public Enemy_Base, public Attr_DoubleAxisRotation
    {
    typedef Enemy_Base super;
    typedef Attr_DoubleAxisRotation transform;
    private:
    public:
        virtual uint32 getClassID() const { return CID_ENEMY_SPHERE; }

        virtual void updateAsync(float32 dt)
        {
            super::updateAsync(dt);
            setTransform(computeMatrix());
        }
    };

} // namespace atomic
#endif // __atomic_Game_Character__
