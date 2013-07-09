#ifndef atm_Game_EntityClass_h
#define atm_Game_EntityClass_h

namespace atm {

istSEnumBlock(EntityCategoryID,
    istSEnum(ECA_Unknown),
    istSEnum(ECA_Player),
    istSEnum(ECA_Enemy),
    istSEnum(ECA_Obstacle),
    istSEnum(ECA_Level),
    istSEnum(ECA_End),
)

istSEnumBlock(EntityClassID,
    istSEnum(EC_Unknown),

    istSEnumEq(EC_Player_Begin, ECA_Player<<9),
    istSEnum(EC_Player),
    istSEnum(EC_Barrier),
    istSEnum(EC_Player_End),

    istSEnumEq(EC_Enemy_Begin, ECA_Enemy<<9),
    istSEnum(EC_Enemy_Test),
    // zako
    istSEnum(EC_Electron),
    istSEnum(EC_Proton),
    istSEnum(EC_Positron),
    istSEnum(EC_Antiproton),
    istSEnum(EC_SmallNucleus),
    istSEnum(EC_MediumNucleus),
    istSEnum(EC_LargeNucleus),
    istSEnum(EC_HomingMine),
    istSEnum(EC_LaserMissile),
    istSEnum(EC_SmallFighter),
    istSEnum(EC_MediumFighter),
    istSEnum(EC_LargeFighter),
    istSEnum(EC_Shell),
    istSEnum(EC_Tortoise),
    istSEnum(EC_Zab),
    istSEnum(EC_BulletTurret),
    istSEnum(EC_LaserTurret),
    istSEnum(EC_HatchSmall),
    istSEnum(EC_HatchLarge),
    istSEnum(EC_CarrierSmall),
    istSEnum(EC_CarrierLarge),
    istSEnum(EC_BreakableParts),
    istSEnum(EC_BreakableCore),
    // boss
    istSEnum(EC_Nova),
    istSEnum(EC_Nebula),
    istSEnum(EC_Pulsar),
    istSEnum(EC_Gravitron),
    istSEnum(EC_Photon),

    istSEnum(EC_Enemy_End),

    istSEnumEq(EC_Obstacle_Begin, ECA_Obstacle<<9),
    istSEnum(EC_PointLightEntity),
    istSEnum(EC_DirectionalLightEntity),
    istSEnum(EC_SpotLightEntity),
    istSEnum(EC_BoxLightEntity),
    istSEnum(EC_CylinderLightEntity),
    istSEnum(EC_GroundBlock),
    istSEnum(EC_FluidFilter),
    istSEnum(EC_FluidFan),
    istSEnum(EC_LevelLayer),
    istSEnum(EC_GearParts),
    istSEnum(EC_GearSmall),
    istSEnum(EC_GearLarge),
    istSEnum(EC_Obstacle_End),

    istSEnumEq(EC_Level_Begin, ECA_Level<<9),
    istSEnum(EC_Level_Test),
    istSEnum(EC_Level_Stage1),
    istSEnum(EC_Level_Stage2),
    istSEnum(EC_Level_Stage3),
    istSEnum(EC_Level_Stage4),
    istSEnum(EC_Level_Stage5),
    istSEnum(EC_Level_Stage6),
    istSEnum(EC_Level_End),

    istSEnum(EC_End),
)


#ifndef atm_Game_EntityClass_detail
#define atm_Game_EntityClass_detail

// EntityHandle は
// 上位 3bit がカテゴリ (EntityCategoryID)
// カテゴリの 3bit も含めた上位 12 bit が class ID (EntityClassID)
// それ以下が entity table のインデックス
inline uint32 EntityGetCategory(EntityHandle e)     { return (e & 0xE0000000) >> 29; }
inline uint32 EntityGetClassID(EntityHandle e)      { return (e & 0xFFF00000) >> 20; }
inline uint32 EntityGetClassIndex(EntityHandle e)   { return (e & 0x1FF00000) >> 20; }
inline uint32 EntityGetIndex(EntityHandle e)        { return (e & 0x000FFFFF) >>  0; }
inline EntityHandle EntityCreateHandle(uint32 classid, uint32 index) { return (classid<<20) | index; }

enum DeployFlags {
    DF_None  = 0,
    DF_RTS   = 1, // RTS モードでもエディタでもデプロイ可
    DF_Editor= 2, // エディタでのみデプロイ可
};
struct EntityClassInfo
{
    DeployFlags deploy;
    float32     cost;

    EntityClassInfo(DeployFlags df=DF_None, float32 c=10.0f) : deploy(df), cost(c) {}
};

class IEntity;
typedef IEntity* (*EntityCreator)();
void AddEntityCreator(EntityClassID entity_classid, EntityCreator creator, const EntityClassInfo &eci=EntityClassInfo());
IEntity* CreateEntity(EntityClassID entity_classid);
EntityClassInfo* GetEntityClassInfo(EntityClassID entity_classid);

template<class EntityType> IEntity* CreateEntity();
template<class EntityType> class AddEntityTable;

#define atmImplementEntity(Class, ...) \
    template<> IEntity* CreateEntity<Class>() { return istNew(Class)(); } \
    template<> struct AddEntityTable<Class> {\
        AddEntityTable() { AddEntityCreator(EC_##Class, &CreateEntity<Class>, EntityClassInfo(__VA_ARGS__)); }\
    };\
    AddEntityTable<Class> g_add_entity_creator_##Class;

#endif // atm_Game_EntityClass_detail

} // namespace atm
#endif //atm_Game_EntityClass_h
