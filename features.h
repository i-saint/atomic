#ifndef atomic_features_h
#define atomic_features_h

#define atomic_version 1
#define atomic_version_string "0.01"
#define atomic_replay_version 1
#define atomic_crash_report_url "http://primitive-games.jp/atomic/crash_report/post"
#define atomic_patch_url        "http://primitive-games.jp/atomic/update/"

#define atomic_MaxPlayerNum 4

#define atomic_enable_LevelEditorServer
#define atomic_enable_GameServer
#define atomic_enable_GameClient

#ifdef ist_env_Master
#else // ist_env_Master
#   define ist_enable_Assert
#   define atomic_enable_debug_menu
#   define atomic_enable_debug_console
#   define atomic_enable_strict_handle_check
#   define atomic_enable_debug_log
#   define atomic_enable_sync_lock
#   define atomic_enable_gbuffer_viewer
#   define atomic_enable_shader_live_edit
#endif // ist_env_Master

//#define atomic_enable_distance_field

#endif //atomic_features_h
