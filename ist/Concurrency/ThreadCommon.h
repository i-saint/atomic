#ifndef __ist_Concurrency_ThreadCommon_h__
#define __ist_Concurrency_ThreadCommon_h__

#ifdef ist_env_Windows
#   include <windows.h>
#else // ist_env_Windows
#   include <pthread.h>
#endif // ist_env_Windows

#endif // __ist_Concurrency_ThreadCommon_h__
