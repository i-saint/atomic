#ifndef __ist_Base_Decl_h__
#define __ist_Base_Decl_h__

#ifdef istWindows

#if defined(istImplementModule)
#   define istInterModule __declspec(dllexport)
#elif defined(istUseModule)
#   define istInterModule __declspec(dllimport)
#else
#   define istInterModule
#endif // istImplementModule
#define istForceInline __forceinline
#define istThreadLocal __declspec(thread)

#else // istWindows

#define istInterModule
#define istForceInline inline
#define istThreadLocal __thread

#endif // istWindows

#endif // __ist_Base_Decl_h__
