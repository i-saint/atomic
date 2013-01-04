#ifndef __atomic_features__
#define __atomic_features__

#define __atomic_version__ 1
#define __atomic_version_string__ "0.01"
#define __atomic_replay_version__ 1



#ifdef _MASTER
#else // _MASTER
#   define ist_enable_assert
#   define atomic_enable_debug_feature
#   define atomic_enable_debug_console
#   define atomic_enable_debug_strict_handle_check
#   define atomic_enable_debug_log
#   define atomic_enable_sync_lock
#endif // _MASTER

//#define atomic_enable_distance_field

#endif //__atomic_features__
