#ifndef HTTPInput_h
#define HTTPInput_h

typedef char            int8;
typedef short           int16;
typedef int             int32;
typedef unsigned char   uint8;
typedef unsigned short  uint16;
typedef unsigned int    uint32;

struct HTTPInputData
{
    struct Keyboard
    {
        char keys[256];
    };

    struct Mouse
    {
        int32 x,y;
        uint32 buttons;
    };

    struct Pad
    {
        int32 x1,y1;
        int32 x2,y2;
        uint32 buttons;
    };

    Keyboard key;
    Mouse    mouse;
    Pad      pad;
};

bool StartHTTPInputServer();
bool StopHTTPInputServer();
const HTTPInputData* GetHTTPInputData();

template<class F>
inline void EnumerateDLLImports(HMODULE module, const char *dllname, const F &f)
{
    if(module==NULL) { return; }

    size_t ImageBase = (size_t)module;
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)ImageBase;
    if(pDosHeader->e_magic!=IMAGE_DOS_SIGNATURE) { return; }

    PIMAGE_NT_HEADERS pNTHeader = (PIMAGE_NT_HEADERS)(ImageBase + pDosHeader->e_lfanew);
    DWORD RVAImports = pNTHeader->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress;
    if(RVAImports==0) { return; }

    IMAGE_IMPORT_DESCRIPTOR *pImportDesc = (IMAGE_IMPORT_DESCRIPTOR*)(ImageBase + RVAImports);
    while(pImportDesc->Name!=0) {
        const char *pDLLName = (const char*)(ImageBase+pImportDesc->Name);
        if(dllname==NULL || stricmp(pDLLName, dllname)==0) {
            IMAGE_IMPORT_BY_NAME **ppSymbolNames = (IMAGE_IMPORT_BY_NAME**)(ImageBase+pImportDesc->Characteristics);
            void **ppFuncs = (void**)(ImageBase+pImportDesc->FirstThunk);
            for(int i=0; ; ++i) {
                if(ppSymbolNames[i]==NULL) { break; }
                char *pName = (char*)(ImageBase+(size_t)ppSymbolNames[i]->Name);
                f(pName, ppFuncs[i]);
            }
        }
        ++pImportDesc;
    }
    return;
}

#endif // HTTPInput_h
