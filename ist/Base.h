#ifndef __ist_Base__
#define __ist_Base__

namespace ist {
    typedef char                int8;
    typedef short               int16;
    typedef int                 int32;
    typedef long long           int64;
    typedef unsigned char       uint8;
    typedef unsigned short      uint16;
    typedef unsigned int        uint32;
    typedef unsigned long long  uint64;
    typedef float               float32;
    typedef double              float64;
} // namespace ist

#include "Base/Allocator.h"
#include "Base/New.h"
#include "Base/Assert.h"
#include "Base/BinaryStream.h"
#include "Base/TaskScheduler.h"
#include "Base/FileLoader.h"

#endif // __ist_Base__