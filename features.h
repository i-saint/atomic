#ifndef __atomic_features__
#define __atomic_features__

#define __atomic_version__ 1
#define __atomic_version_string__ "0.01"
#define __atomic_replay_version__ 1


#define __ist_with_OpenGL__
//#define __ist_with_DirectX11__
#define __ist_with_OpenCL__
#define __ist_with_zlib__
#define __ist_with_oggvorbis__

#ifdef _MASTER
#else // _MASTER
    #define __ist_enable_assert__
    #define __atomic_enable_debug_feature__
    #define __atomic_enable_debug_console__
    #define __atomic_enable_debug_strict_handle_check__
#endif // _MASTER

//#define __atomic_enable_distance_field__

#endif //__atomic_features__
