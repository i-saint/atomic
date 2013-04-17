#include <windows.h>
#include <dinput.h>
#include "HTTPInput.h"

typedef HRESULT (WINAPI *DirectInput8CreateT)(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter);
typedef HRESULT (__stdcall *CreateDeviceT)(IDirectInput8*, REFGUID, LPDIRECTINPUTDEVICE8A*, LPUNKNOWN);
typedef HRESULT (__stdcall *GetDeviceStateT)(IDirectInputDevice8*, DWORD, LPVOID);

DirectInput8CreateT orig_DirectInput8Create;
CreateDeviceT orig_CreateDevice;
GetDeviceStateT orig_GetDeviceState;

HRESULT __stdcall fake_GetDeviceState(IDirectInputDevice8 *dev, DWORD size, LPVOID data)
{
    HRESULT r = orig_GetDeviceState(dev, size, data);
    if(SUCCEEDED(r)) {
        const HTTPInputData *input = GetHTTPInputData();
        if(size==sizeof(DIJOYSTATE)) {
            DIJOYSTATE &state = *(DIJOYSTATE*)data;
            state.lX = abs(input->pad.x1+INT16_MIN)>abs(state.lX+INT16_MIN) ? input->pad.x1 : state.lX;
            state.lY = abs(input->pad.y1+INT16_MIN)>abs(state.lY+INT16_MIN) ? input->pad.y1 : state.lY;
            for(int i=0; i<32; ++i) {
                state.rgbButtons[i] |= (state.rgbButtons[i]&0x80) || (input->pad.buttons & 1<<i) ? 0x80 : 0;
            }
        }
        else if(size==sizeof(DIJOYSTATE2)) {
            DIJOYSTATE2 &state = *(DIJOYSTATE2*)data;
            state.lX = abs(input->pad.x1+INT16_MIN)>abs(state.lX+INT16_MIN) ? input->pad.x1 : state.lX;
            state.lY = abs(input->pad.y1+INT16_MIN)>abs(state.lY+INT16_MIN) ? input->pad.y1 : state.lY;
            for(int i=0; i<32; ++i) {
                state.rgbButtons[i] |= (state.rgbButtons[i]&0x80) || (input->pad.buttons & 1<<i) ? 0x80 : 0;
            }
        }
    }
    return r;
}

HRESULT __stdcall fake_CreateDevice(IDirectInput8 *di8, REFGUID guid, LPDIRECTINPUTDEVICE8A *out, LPUNKNOWN unk)
{
    HRESULT r = orig_CreateDevice(di8, guid, out, unk);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*out))[0];
        if(orig_GetDeviceState==NULL) { (void*&)orig_GetDeviceState=vftable[9]; }
        vftable[9] = &fake_GetDeviceState;
    }
    return r;
}

HRESULT WINAPI fake_DirectInput8Create(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter)
{
    HRESULT r = orig_DirectInput8Create(hinst, dwVersion, riidltf, ppvOut, punkOuter);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*ppvOut))[0];
        if(orig_CreateDevice==NULL) { (void*&)orig_CreateDevice=vftable[3]; }
        vftable[3] = &fake_CreateDevice;
        StartHTTPInputServer();
    }
    return r;
}

bool HookDirectInput8()
{
    bool ret = false;
    EachImportFunctionInEveryModule(
        [](const char *dllname) {
            return _stricmp(dllname, "dinput8.dll")==0;
        },
        [&](const char *funcname, void *&func) {
            if(strcmp(funcname, "DirectInput8Create")==0) {
                (void*&)orig_DirectInput8Create = func;
                ForceWrite<void*>(func, fake_DirectInput8Create);
                ret = true;
            }
        },
        [](DWORD, void *&func) {}
        );
    return ret;
}
