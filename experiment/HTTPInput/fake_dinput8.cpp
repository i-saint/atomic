#include <windows.h>
#include <dinput.h>
#include "HTTPInput.h"
#include "HTTPInput_Internal.h"

typedef HRESULT (WINAPI *DirectInput8CreateT)(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter);
typedef HRESULT (__stdcall *CreateDeviceT)(IDirectInput8*, REFGUID, LPDIRECTINPUTDEVICE8A*, LPUNKNOWN);
typedef HRESULT (__stdcall *GetDeviceStateT)(IDirectInputDevice8*, DWORD, LPVOID);

static DirectInput8CreateT orig_DirectInput8Create;
static CreateDeviceT orig_CreateDevice;
static GetDeviceStateT orig_GetDeviceState;

static HRESULT __stdcall fake_GetDeviceState(IDirectInputDevice8 *dev, DWORD size, LPVOID data)
{
    HRESULT r = orig_GetDeviceState(dev, size, data);
    if(HTTPInput_GetConfig()->override_dinput8) {
        if(FAILED(r)) { memset(data, 0, size); }
        const HTTPInputData::Pad &vpad = HTTPInput_GetData()->pad[0];
        if(size==sizeof(DIJOYSTATE)) {
            DIJOYSTATE &state = *(DIJOYSTATE*)data;
            state.lX = abs(vpad.x1+INT16_MIN)>abs(state.lX+INT16_MIN) ? vpad.x1 : state.lX;
            state.lY = abs(vpad.y1+INT16_MIN)>abs(state.lY+INT16_MIN) ? vpad.y1 : state.lY;
            for(int32 i=0; i<HTTPInputData::Pad::MaxButtons; ++i) {
                state.rgbButtons[i] |= vpad.buttons[i];
            }
        }
        else if(size==sizeof(DIJOYSTATE2)) {
            DIJOYSTATE2 &state = *(DIJOYSTATE2*)data;
            state.lX = abs(vpad.x1+INT16_MIN)>abs(state.lX+INT16_MIN) ? vpad.x1 : state.lX;
            state.lY = abs(vpad.y1+INT16_MIN)>abs(state.lY+INT16_MIN) ? vpad.y1 : state.lY;
            LONG trigger = vpad.trigger1 > vpad.trigger2 ? -vpad.trigger1 : vpad.trigger2;
            state.lZ = abs(trigger)>0x100 ? INT16_MAX+trigger : state.lZ;
            for(int32 i=0; i<HTTPInputData::Pad::MaxButtons; ++i) {
                state.rgbButtons[i] |= vpad.buttons[i];
            }
        }
        return DI_OK;
    }
    return r;
}

static HRESULT __stdcall fake_CreateDevice(IDirectInput8 *di8, REFGUID guid, LPDIRECTINPUTDEVICE8A *out, LPUNKNOWN unk)
{
    HRESULT r = orig_CreateDevice(di8, guid, out, unk);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*out))[0];
        if(orig_GetDeviceState==NULL) { (void*&)orig_GetDeviceState=vftable[9]; }
        vftable[9] = &fake_GetDeviceState;
    }
    return r;
}

static HRESULT WINAPI fake_DirectInput8Create(HINSTANCE hinst, DWORD dwVersion, REFIID riidltf, LPVOID * ppvOut, LPUNKNOWN punkOuter)
{
    HRESULT r = orig_DirectInput8Create(hinst, dwVersion, riidltf, ppvOut, punkOuter);
    if(SUCCEEDED(r)) {
        void **&vftable = ((void***)(*ppvOut))[0];
        if(orig_CreateDevice==NULL) { (void*&)orig_CreateDevice=vftable[3]; }
        vftable[3] = &fake_CreateDevice;
        HTTPInput_StartServer();
    }
    return r;
}

static FuncInfo g_dinput8_funcs[] = {
    {"DirectInput8Create", 0, (void*)&fake_DirectInput8Create, (void**)&orig_DirectInput8Create},
};
OverrideInfo g_dinput8_overrides = {"dinput8.dll", _countof(g_dinput8_funcs), g_dinput8_funcs};
