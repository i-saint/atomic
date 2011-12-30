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
        istPrint("alcOpenDevice() failed");
        return false;
    }

    g_context = alcCreateContext(g_device, NULL);
    if(!g_context) {
        alcCloseDevice(g_device); g_device=NULL;
        istPrint("alcCreateContext() failed");
        return false;
    }

    alcMakeContextCurrent(g_context);
    return true;
}

void FinalizeSound()
{
    alcMakeContextCurrent(0);
    alcDestroyContext(g_context);
    alcCloseDevice(g_device);
}

} // namespace sound
} // namespace ist
