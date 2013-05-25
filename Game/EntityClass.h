﻿#ifndef atomic_Game_EntityClass_h
#define atomic_Game_EntityClass_h

namespace atomic {

enum EntityCategoryID
{
    ECA_Unknown,
    ECA_Player,
    ECA_Enemy,
    ECA_Bullet,
    ECA_Obstacle,
    ECA_Level,
    ECA_VFX,

    ECA_End,
};

enum EntityClassID
{
    EC_Unknown,

    EC_Player_Begin = ECA_Player<<9,
    EC_Player,
    EC_Barrier,
    EC_Player_End,

    EC_Enemy_Begin = ECA_Enemy<<9,
    EC_Enemy_Test,
    EC_Enemy_End,

    EC_Bullet_Begin = ECA_Bullet<<9,
    EC_Bullet_Simple,
    EC_Bullet_Particle,
    EC_Bullet_Laser,
    EC_Bullet_End,

    EC_Obstacle_Begin = ECA_Obstacle<<9,
    EC_Obstacle_End,

    EC_Level_Begin = ECA_Level<<9,
    EC_Level_Test,
    EC_Level_End,

    EC_VFX_Begin = ECA_VFX<<9,
    EC_VFX_Scintilla,
    EC_VFX_End,

    EC_End,
};

// EntityHandle は
// 上位 3bit がカテゴリ (EntityCategoryID)
// カテゴリの 3bit も含めた上位 12 bit が class ID (EntityClassID)
// それ以下が entity table のインデックス
inline uint32 EntityGetCategory(EntityHandle e)     { return (e & 0xE0000000) >> 29; }
inline uint32 EntityGetClassID(EntityHandle e)      { return (e & 0xFFF00000) >> 20; }
inline uint32 EntityGetClassIndex(EntityHandle e)   { return (e & 0x1FF00000) >> 20; }
inline uint32 EntityGetIndex(EntityHandle e)        { return (e & 0x000FFFFF) >>  0; }
inline EntityHandle EntityCreateHandle(uint32 classid, uint32 index) { return (classid<<20) | index; }


class IEntity;
typedef IEntity* (*EntityCreator)();
EntityCreator* GetEntityCreatorTable(EntityClassID entity_classid);
void AddEntityCreator(EntityClassID entity_classid, EntityCreator creator);
IEntity* CreateEntity(EntityClassID entity_classid);

template<class EntityType> IEntity* CreateEntity();
template<class EntityType> class AddEntityTable;

#define atomicImplementEntity(Class) \
    template<> IEntity* CreateEntity<Class>() { return istNew(Class)(); } \
    template<> struct AddEntityTable<Class> {\
        AddEntityTable() { AddEntityCreator(EC_##Class, &CreateEntity<Class>); }\
    };\
    AddEntityTable<Class> g_add_entity_creator_##Class;

#define atomicImplementEntity_Pooled(Class) \
    template<> IEntity* CreateEntity<Class>() { return Class::create(); } \
    template<> struct AddEntityTable<Class> {\
        AddEntityTable() { AddEntityCreator(EC_##Class, &CreateEntity<Class>); }\
    };\
    AddEntityTable<Class> g_add_entity_creator_##Class;


} // namespace atomic
#endif //atomic_Game_EntityClass_h
