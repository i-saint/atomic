#include "stdafx.h"
#include "Allocator.h"

void bad_alloc_hander_generic(const void* allocator_ptr)
{
    DebugBreak();
}
