#ifndef __ist_Config_h__

#define __ist_enable_assert__
#define __ist_enable_graphics_assert__

#define __ist_with_EASTL__
#define __ist_with_OpenGL__
//#define __ist_with_OpenGLES__
//#define __ist_with_DirectX11__
#define __ist_with_zlib__
#define __ist_with_png__
//#define __ist_with_jpeg__
#define __ist_with_gli__ // dds ファイル対応
#define __ist_with_OpenAL__
#define __ist_with_oggvorbis__

#if defined(_WIN64)
#   define istWindows
#   define istWin64
#   define istx86
#   define istx86_64
#elif defined(_WIN32)
#   define istWindows
#   define istWin32
#   define istx86
#elif defined(__ANDROID__)
#   define istAndroid
#   define istARM
#else
#   error
#endif


#endif // __ist_Config_h__
