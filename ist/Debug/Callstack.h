#ifndef __ist_Debug_Callstack_h__
#define __ist_Debug_Callstack_h__
namespace ist {

    istInterModule bool InitializeSymbol();
    istInterModule void FinalizeSymbol();

    istInterModule int GetCallstack(void **callstack, int callstack_size, int skip_size=0);
    istInterModule stl::string AddressToSymbolName(void *address);

    // utility
    istInterModule stl::string CallstackToSymbolNames(void **callstack, int callstack_size, int clamp_head=0, int clamp_tail=0, const char *indent="");

} // namespace ist
#endif // __ist_Debug_Callstack_h__
