#include <windows.h>
#include <mmsystem.h>

typedef MMRESULT (WINAPI *joyGetPosExT)(UINT uJoyID, LPJOYINFOEX pji);

MMRESULT WINAPI fake_joyGetPosEx(UINT uJoyID, LPJOYINFOEX pji)
{
    return JOYERR_NOERROR;
}

bool HookMMJoustick()
{
    return false;
}
