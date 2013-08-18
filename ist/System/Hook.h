#ifndef ist_System_Hook_h
#define ist_System_Hook_h
#ifdef ist_env_Windows

namespace ist {

// target: 関数ポインタ。対象関数を hotpatch して元の関数へのポインタを返す
void* Hotpatch( void *target, const void *replacement );
void* UglyHotpatch( void *target, const void *replacement );

} // namespace ist
#endif // ist_env_Windows
#endif // ist_System_Hook_h
