#ifndef ist_Debug_Callstack_h
#define ist_Debug_Callstack_h
namespace ist {

istInterModule bool InitializeDebugSymbol();
istInterModule void FinalizeDebugSymbol();

istInterModule int GetCallstack(void **callstack, int callstack_size, int skip_size=0);
istInterModule stl::string AddressToSymbolName(void *address);

// utility
istInterModule stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head=0, int clamp_tail=0, const char *indent="");

// 指定のアドレスが現在のモジュールの static 領域内であれば true
istInterModule bool IsStaticMemory(void *addr);
// 指定アドレスが現在のスレッドの stack 領域内であれば true
istInterModule bool IsStackMemory(void *addr);
// 指定のアドレスが heap 領域内であれば true
istInterModule bool IsHeapMemory(void *addr);

// 呼び出し元がメンバ関数の場合、その this を返す
// !! Debug() ビルド以外で使うのは危険です。全く無関係なポインタを返してきたりします !!
istInterModule void* GetThisOfCaller();

} // namespace ist
#endif // ist_Debug_Callstack_h
