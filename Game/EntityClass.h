#ifndef __atomic_Game_EntityClass__
#define __atomic_Game_EntityClass__

namespace atomic {
    enum ENTITY_CATEGORY_ID
    {
        ECID_UNKNOWN,
        ECID_PLAYER,
        ECID_ENEMY,
        ECID_OBSTRUCT,
        ECID_BULLET,
        ECID_LEVEL,
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
        ESID_BULLET_SIMPLE,
        ESID_BULLET_END,
    };

    enum ENTITY_LEVEL_CLASS_ID
    {
        ESID_LEVEL_TEST,
        ESID_LEVEL_END,
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

    // EntityHandle: 上位 4 bit がカテゴリ (ENTITY_CATEGORY_ID)、その次 8 bit がカテゴリ内種別 (ENTITY_*_CLASS_ID)、それ以下は ID のフィールド
    inline uint32 EntityGetCategory(EntityHandle e) { return (e & 0xF0000000) >> 28; }
    inline uint32 EntityGetClass(EntityHandle e)    { return (e & 0x0FF00000) >> 20; }
    inline uint32 EntityGetID(EntityHandle e)       { return (e & 0x000FFFFF) >>  0; }
    inline uint32 EntityCreateHandle(uint32 cid, uint32 sid, uint32 id) { return (cid<<28) | (sid<<20) | id; }


    class Player;

    class Enemy_CubeBasic;
    class Enemy_SphereBasic;

    class Bullet_Simple;

    class Level_Test;

} // namespace atomic
#endif //__atomic_Game_EntityClass__
