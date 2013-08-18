#include "istPCH.h"
#include "Hook.h"

#ifdef ist_env_Windows
#pragma comment(lib, "psapi.lib")
#include <windows.h>
#include <psapi.h>

namespace ist {


// target: 関数ポインタ。対象関数を hotpatch して元の関数へのポインタを返す
void* Hotpatch( void *target, const void *replacement )
{
    DWORD old;
    BYTE *f = (BYTE*)target;
    void *orig_func = f+2;
    ::VirtualProtect(f-5, 7, PAGE_EXECUTE_READWRITE, &old);
    f[-5]=0xE9; // jmp
    *((ptrdiff_t*)(f-4)) = (ptrdiff_t)replacement-(ptrdiff_t)f;
    f[0]=0xEB; f[1]=0xF9; // short jmp -7
    ::VirtualProtect(f-5, 7, old, &old);
    return orig_func;
}



// 関数ポインタ addr から required byte を含む instruction のサイズを求める
size_t GuessInstructionSize(void *addr, size_t required)
{
    // 不完全につき、未対応の instruction があれば適宜追加すべし
    // 関数の頭 5 byte 以内で実行されるものは多くが mov,sub,push あたりなのでこれだけでも多くに対応はできるハズ
    size_t ret = 0;
    BYTE *data = (BYTE*)addr;
    for(; ret<required; ) {
        switch(data[ret]) {
            // push
        case 0x55: ret+=1; break;
        case 0x68:
            ret+=1;
            switch(data[ret]) {
            case 0x6C: ret+=4; break;
            default:   ret+=1; break;
            }
            break;
        case 0x6A: ret+=5; break;
        case 0xFF: ret+=3; break;

            // mov
        case 0x8B:
            ret+=1;
            switch(data[ret]) {
            case 0x44: ret+=3; break;
            case 0x45: ret+=2; break;
            default:   ret+=1; break;
            }
            break;
        case 0xB8: ret+=5; break;

            // sub
        case 0x81: 
            ret+=1;
            switch(data[ret]) {
            case 0xEC: ret+=5; break;
            default:   ret+=1; break;
            }
            break;
        case 0x83:
            ret+=1;
            switch(data[ret]) {
            case 0xEC: ret+=2; break;
            default:   ret+=1; break;
            }
            break;
        default: ret+=1; break;
        }
    }
    return ret;
}

// target: 関数ポインタ。対象関数を上書きしつつ元のコードは退避して、元の関数へのポインタを返す
void* UglyHotpatch( void *target, const void *replacement )
{
    // 元コードの退避先
    BYTE *before = (BYTE*)::VirtualAlloc(NULL, 64, MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE);

    BYTE *f = (BYTE*)target;
    DWORD old;
    ::VirtualProtect(f, 32, PAGE_EXECUTE_READWRITE, &old);

    // 元のコードをコピー & 最後にコピー本へ jmp するコードを付加 (==これを call すれば上書き前の動作をするハズ)
    size_t slice = GuessInstructionSize(f, 5);
    memcpy(before, f, slice);
    before[slice]=0xE9; // jmp
    *(DWORD*)(before+slice+1) = (ptrdiff_t)(f+slice)-(ptrdiff_t)(before+slice) - 5;

    // 関数の先頭を hook 関数への jmp に書き換える
    f[0]=0xE9; // jmp
    *(DWORD*)(f+1) = (ptrdiff_t)replacement-(ptrdiff_t)f - 5;
    ::VirtualProtect(f, 32, old, &old);

    return before;
}

} // namespace ist
#endif // ist_env_Windows
