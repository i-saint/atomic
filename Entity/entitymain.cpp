#include "stdafx.h"

#ifndef atmMainImpl
istImplementOperatorNewDelete();
#endif // atmMainImpl

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    if(fdwReason==DLL_PROCESS_ATTACH) {
    }
    else if(fdwReason==DLL_PROCESS_DETACH) {
    }
    return TRUE;
}
