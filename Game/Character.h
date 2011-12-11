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

    class Entity;
    class Routine;

    class Entity
    {
    private:
    public:

    };

    class Routine
    {

    };


    class CharacterSet
    {
    private:
    public:
    };



    class Enemy_Base
    {
    private:
        uint32  m_handle;
        float32 m_health;
        mat4    m_transform;

    protected:
        void setHandle(uint32 h) { m_handle=h; }

    public:
        Enemy_Base() : m_handle(0), m_health(1.0f), m_transform() {}

        uint32      getHandle() const   { return m_classid; }
        float32     getHealth() const   { return m_health; }
        const mat4& getTransform() const{ return m_transform; }

        void setHelth(float32 v)        { m_heath=v; }
        void setTransform(const mat4& v){ m_transform=v; }
    };

    class Enemy_Cube
    {
    private:
    public:
    };

} // namespace atomic
#endif // __atomic_Game_Character__
