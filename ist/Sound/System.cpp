#include "stdafx.h"
#include "../Sound.h"
#include "../Base.h"
#include "Internal.h"

namespace ist {
namespace sound {

bool IntializeSound()
{
    g_device = alcOpenDevice(NULL);
    if(!g_device) {
        IST_PRINT("alcOpenDevice() failed");
        return false;
    }

    g_context = alcCreateContext(g_device, 0);
    if(!g_context) {
        alcCloseDevice(g_device); g_device=NULL;
        return false;
        IST_PRINT("alcCreateContext() failed");
    }

    alcMakeContextCurrent(g_context);
    return true;
}

void FinalizeSound()
{
    alcDestroyContext(g_context);
    alcCloseDevice(g_device);
}

} // namespace sound
} // namespace ist
