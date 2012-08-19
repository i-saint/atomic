#ifndef __ist_Concurrency_ThreadCommon_h__
#define __ist_Concurrency_ThreadCommon_h__

#ifdef __ist_env_Windows__
#   include <windows.h>
#else // __ist_env_Windows__
#   include <pthread.h>
#endif // __ist_env_Windows__

#endif // __ist_Concurrency_ThreadCommon_h__
