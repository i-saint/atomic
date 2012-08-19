#ifndef __ist_Base_Decl_h__
#define __ist_Base_Decl_h__

#ifdef __ist_env_Windows__

#define istForceInline __forceinline
#define istThreadLocal __declspec(thread)

#else // __ist_env_Windows__

#define istForceInline inline
#define istThreadLocal __thread

#endif // __ist_env_Windows__

#endif // __ist_Base_Decl_h__
