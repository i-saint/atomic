#ifndef __ist_Concurrency_ThreadUtil_h__
#define __ist_Concurrency_ThreadUtil_h__

#include "ist/Base/Types.h"
#ifdef _WIN32
#include <intrin.h>
#endif

namespace ist {

void SetThreadName(const char *name);
void SetThreadName( uint32 dwThreadID, const char* name);

} // namespace ist

#endif // __ist_Concurrency_ThreadUtil_h__
