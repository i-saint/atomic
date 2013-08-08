#ifndef atm_features_h
#define atm_features_h

#define atm_version 10
#define atm_version_string "0.1"
#define atm_replay_version 2
#define atm_crash_report_url "http://primitive-games.jp/atomic/crash_report/post"
#define atm_patch_url        "http://primitive-games.jp/atomic/update/"
#define atm_molecule_url     "http://primitive-games.jp/atomic/molecule/"

#define atm_MaxPlayerNum 4

#ifdef ist_env_Master
#   define atm_NetworkTimeout 3
#else
#   define atm_NetworkTimeout 180
#endif 
#define atm_Leveleditor_DefaultPort 10050
#define atm_GameServer_DefaultPort 10051


#define atm_enable_LevelEditorServer
#define atm_enable_WebGL
//// C84 版は coop 省略
//#define atm_enable_GameServer
//#define atm_enable_GameClient
#define atm_enable_EntityDLL
//#define atm_enable_DistanceField

#ifdef ist_env_Master
#   define dpDisable
#   define wdmDisable
#else // ist_env_Master
#   define ist_enable_Assert
#   define atm_enable_StrictHandleCheck
#   define atm_enable_DebugLog
#   define atm_enable_SyncLock
#   define atm_enable_GBufferViewer
#   define atm_enable_ShaderLiveEdit
#   define atm_enable_StateSave
#endif // ist_env_Master

#endif //atm_features_h
