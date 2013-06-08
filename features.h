#ifndef atm_features_h
#define atm_features_h

#define atm_version 1
#define atm_version_string "0.01"
#define atm_replay_version 1
#define atm_crash_report_url "http://primitive-games.jp/atomic/crash_report/post"
#define atm_patch_url        "http://primitive-games.jp/atomic/update/"

#define atm_MaxPlayerNum 4

#ifdef ist_env_Master
#   define atm_NetworkTimeout 3
#else
#   define atm_NetworkTimeout 180
#endif 
#define atm_Leveleditor_DefaultPort 10050
#define atm_GameServer_DefaultPort 10051


#define atm_enable_LevelEditorServer
#define atm_enable_GameServer
#define atm_enable_GameClient
//#define atm_enable_BinaryEntityData

#ifdef ist_env_Master
#else // ist_env_Master
#   define ist_enable_Assert
#   define atm_enable_debug_menu
#   define atm_enable_debug_console
#   define atm_enable_strict_handle_check
#   define atm_enable_debug_log
#   define atm_enable_sync_lock
#   define atm_enable_gbuffer_viewer
#   define atm_enable_shader_live_edit
#endif // ist_env_Master

//#define atm_enable_distance_field

#endif //atm_features_h
