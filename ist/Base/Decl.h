#ifndef __ist_Base_Decl_h__
#define __ist_Base_Decl_h__

#ifdef istWindows

#define istForceInline __forceinline
#define istThreadLocal __declspec(thread)

#else // istWindows

#define istForceInline inline
#define istThreadLocal __thread

#endif // istWindows

#endif // __ist_Base_Decl_h__
