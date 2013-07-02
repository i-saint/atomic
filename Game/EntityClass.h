#ifndef atm_Game_EntityClass_h
#define atm_Game_EntityClass_h

namespace atm {

enum EntityCategoryID
{
    ECA_Unknown,
    ECA_Player,
    ECA_Enemy,
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
    EC_HatchSmall,
    EC_HatchLarge,
    EC_CarrierSmall,
    EC_CarrierLarge,
    EC_HomingMine,
    EC_LaserMissile,
    EC_BreakableParts,
    EC_BreakableCore,
    EC_SmallFighter,
    EC_MediumFighter,
    EC_LargeFighter,
    EC_Shell,
    EC_Tortoise,
    EC_Zab,
    EC_Nucleus,
    EC_BulletTurret,
    EC_LaserTurret,
    EC_Boss1,
    EC_Boss2,
    EC_Boss3,
    EC_Boss4,
    EC_Boss5,
    EC_Enemy_End,

    EC_Obstacle_Begin = ECA_Obstacle<<9,
    EC_PointLightEntity,
    EC_DirectionalLightEntity,
    EC_SpotLightEntity,
    EC_BoxLightEntity,
    EC_CylinderLightEntity,
    EC_GroundBlock,
    EC_FluidFilter,
    EC_FluidFan,
    EC_LevelLayer,
    EC_GearParts,
    EC_GearSmall,
    EC_GearLarge,
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

#define atmImplementEntity(Class) \
    template<> IEntity* CreateEntity<Class>() { return istNew(Class)(); } \
    template<> struct AddEntityTable<Class> {\
        AddEntityTable() { AddEntityCreator(EC_##Class, &CreateEntity<Class>); }\
    };\
    AddEntityTable<Class> g_add_entity_creator_##Class;

#define atmImplementEntity_Pooled(Class) \
    template<> IEntity* CreateEntity<Class>() { return Class::create(); } \
    template<> struct AddEntityTable<Class> {\
        AddEntityTable() { AddEntityCreator(EC_##Class, &CreateEntity<Class>); }\
    };\
    AddEntityTable<Class> g_add_entity_creator_##Class;


} // namespace atm
#endif //atm_Game_EntityClass_h
