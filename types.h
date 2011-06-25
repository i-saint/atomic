#ifndef __atomic_Types__
#define __atomic_Types__

#include "ist/ist.h"
using namespace ist::graphics;

namespace atomic
{

typedef char                int8;
typedef short               int16;
typedef int                 int32;
typedef long long           int64;
typedef unsigned char       uint8;
typedef unsigned short      uint16;
typedef unsigned int        uint32;
typedef unsigned long long  uint64;

typedef float   float32;
typedef double  float64;


struct __declspec(align(16)) quadword { char c[16]; };



using ist::Task;
using ist::TaskScheduler;


typedef unsigned int id_t;
enum {
    CHR_FRACTION    = 0x80000000,
    CHR_BULLET      = 0x40000000,
    CHR_GROUND      = 0x20000000,
    INVALID_ID = 0xffffffff,
};




typedef eastl::allocator PermanentAllocator;
//typedef stack_allocator< stack_allocator<eastl::allocator> > FrameScopedAllocator; // ‚ ‚Æ‚Å‚±‚¿‚ç‚É
typedef eastl::allocator FrameScopedAllocator;



} // namespace atomic
#endif // __atomic_Types__
