#ifndef ist_Debug_Callstack_h
#define ist_Debug_Callstack_h
namespace ist {

    istInterModule bool InitializeDebugSymbol();
    istInterModule void FinalizeDebugSymbol();

    istInterModule int GetCallstack(void **callstack, int callstack_size, int skip_size=0);
    istInterModule stl::string AddressToSymbolName(void *address);

    // utility
    istInterModule stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head=0, int clamp_tail=0, const char *indent="");

} // namespace ist
#endif // ist_Debug_Callstack_h
