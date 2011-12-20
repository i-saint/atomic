#include "stdafx.h"
#include "../Sound.h"
#include "../Base.h"
#include "Internal.h"

namespace ist {
namespace sound {

bool IntializeSound()
{
    g_device = alcOpenDevice(0);
    IST_PRINT("alcOpenDevice() succeeded\n");
    if(!g_device) {
        IST_PRINT("alcOpenDevice() failed");
        return false;
    }

    g_context = alcCreateContext(g_device, 0);
    IST_PRINT("alcCreateContext() succeeded\n");
    if(!g_context) {
        alcCloseDevice(g_device); g_device=NULL;
        return false;
        IST_PRINT("alcCreateContext() failed");
    }

    alcMakeContextCurrent(g_context);
    IST_PRINT("alcMakeContextCurrent() succeeded\n");
    return true;
}

void FinalizeSound()
{
    alcDestroyContext(g_context);
    alcCloseDevice(g_device);
}

} // namespace sound
} // namespace ist
