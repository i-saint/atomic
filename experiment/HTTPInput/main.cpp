#include <windows.h>
#include "HTTPInput.h"

bool HookKeyboardMouse();
bool HookMMJoustick();
bool HookDirectInput8();
bool HookXInput();

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved)
{
    if(fdwReason==DLL_PROCESS_ATTACH) {
        HookKeyboardMouse();
        HookMMJoustick();
        HookDirectInput8();
        HookXInput();
    }
    else if(fdwReason==DLL_PROCESS_DETACH) {
        StopHTTPInputServer();
    }
    return TRUE;
}