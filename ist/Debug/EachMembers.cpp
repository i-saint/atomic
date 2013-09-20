#include "istPCH.h"
#include "EachMembers.h"

#define _NO_CVCONST_H
#include <dbghelp.h>


namespace ist {


enum BasicType
{
    btNoType = 0,
    btVoid = 1,
    btChar = 2,
    btWChar = 3,
    btInt = 6,
    btUInt = 7,
    btFloat = 8,
    btBCD = 9,
    btBool = 10,
    btLong = 13,
    btULong = 14,
    btCurrency = 25,
    btDate = 26,
    btVariant = 27,
    btComplex = 28,
    btBit = 29,
    btBSTR = 30,
    btHresult = 31
};

struct EMVContext
{
    HANDLE hprocess;
    ULONG64 modbase;
    MemberInfo current;
    std::vector<MemberInfo> members;
    std::string tmp_name;

    EMVContext()
        : hprocess(::GetCurrentProcess())
        , modbase((ULONG64)::GetModuleHandleA(nullptr))
    {
    }
};

bool GetSymbolName(EMVContext &ctx, DWORD t)
{
    WCHAR *wname = NULL;
    if(::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_SYMNAME, &wname )) {
        size_t num = 0;
        char out[MAX_SYM_NAME];
        ::wcstombs_s(&num, out, wname, _countof(out));
        ctx.tmp_name = out;
        ::LocalFree(wname);
        return true;
    }
    return false;
}

bool GetSymbolTypeNameImpl(EMVContext &ctx, DWORD t, std::string &ret)
{
    DWORD tag = 0;
    DWORD basetype = 0;
    if(!::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_SYMTAG, &tag)) {
        return false;
    }

    if(tag==SymTagArrayType) {
        DWORD count = 0;
        ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_COUNT, &count);
        char a[128];
        sprintf(a, "[%d]", count);
        ret += a;

        DWORD tid = 0;
        ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_TYPEID, &tid);
        return GetSymbolTypeNameImpl(ctx, tid, ret);
    }
    else if(tag==SymTagPointerType) {
        ret = "*"+ret;

        DWORD tid = 0;
        ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_TYPEID, &tid);
        return GetSymbolTypeNameImpl(ctx, tid, ret);
    }
    else if(tag==SymTagBaseType) {
        ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_BASETYPE, &basetype);
        ULONG64 length = 0;
        ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_LENGTH, &length);
        std::string type;
        switch(basetype) {
        case btChar:  type="char"; break;
        case btWChar: type="wchar"; break;
        case btBool:  type="bool"; break;
        case btInt:   type="int"; break;
        case btUInt:  type="uint"; break;
        case btFloat: type="float"; break;
        }
        switch(basetype) {
        case btInt:
        case btUInt:
        case btFloat:
            char bits[32];
            sprintf(bits, "%d", length*8);
            type+=bits;
            break;
        }
        ret = type+ret;
    }
    else { // user defined type
        WCHAR *wname = nullptr;
        if(::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_SYMNAME, &wname )) {
            char name[MAX_SYM_NAME];
            size_t num = 0;
            ::wcstombs_s(&num, name, wname, _countof(name));
            ::LocalFree(wname);
            ret+=name;
        }
    }
    return true;
}
bool GetSymbolTypeName(EMVContext &ctx, DWORD t)
{
    ctx.tmp_name.clear();
    return GetSymbolTypeNameImpl(ctx, t, ctx.tmp_name);
}

void EachMembers(EMVContext &ctx, DWORD t, const MemberInfoCallback &f)
{
    DWORD tag = 0;
    if(!::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_SYMTAG, &tag)) {
        return;
    }

    if(tag==SymTagData) {
        DWORD offset = 0;
        DWORD tid = 0;
        if( ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_OFFSET, &offset) &&
            ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_TYPEID, &tid) )
        {
            ctx.current.value = (void*)((size_t)ctx.current.base_pointer+offset);
            GetSymbolTypeName(ctx, tid);
            ctx.current.type_name = ctx.tmp_name;
            GetSymbolName(ctx, t);
            ctx.current.value_name = ctx.tmp_name;
            ctx.current.type = MT_Variable;
            ctx.members.push_back(ctx.current);
            //f(ctx.mi);
        }
    }
    else if(tag==SymTagBaseClass) {
        void *base_prev = ctx.current.base_pointer;
        DWORD offset = 0;
        DWORD type = 0;
        if( ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_OFFSET, &offset) &&
            ::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_TYPE, &type) )
        {
            ctx.current.base_pointer = (void*)((size_t)base_prev+offset);
            EachMembers(ctx, type, f);
            ctx.current.base_pointer = base_prev;
        }
    }
    else if(tag==SymTagUDT) {
        std::string prev = ctx.current.class_name;
        GetSymbolName(ctx, t);
        ctx.current.class_name = ctx.tmp_name;
        if(ctx.current.this_type!=ctx.current.class_name) {
            ctx.current.type = MT_BaseClass;
            ctx.current.value = nullptr;
            ctx.current.value_name = "";
            ctx.current.type_name = "";
            ctx.members.push_back(ctx.current);
        }

        DWORD num_members = 0;
        if(::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_GET_CHILDRENCOUNT, &num_members)) {
            TI_FINDCHILDREN_PARAMS *params = (TI_FINDCHILDREN_PARAMS*)malloc(sizeof(TI_FINDCHILDREN_PARAMS ) + (sizeof(ULONG)*num_members));
            params->Count = num_members;
            params->Start = 0;
            if(::SymGetTypeInfo(ctx.hprocess, ctx.modbase, t, TI_FINDCHILDREN, params )) {
                for(DWORD i=0; i<num_members; ++i) {
                    EachMembers(ctx, params->ChildId[i], f);
                }
            }
            free(params);
        }
        ctx.current.class_name = prev;
    }
}



bool EachMembersImpl(EMVContext &ctx, const MemberInfoCallback &f)
{
    ULONG tindex = 0;
    bool ok = false;
    {
        char buf[sizeof(SYMBOL_INFO)+MAX_SYM_NAME];
        PSYMBOL_INFO sinfo = (PSYMBOL_INFO)buf;
        sinfo->SizeOfStruct = sizeof(SYMBOL_INFO);
        sinfo->MaxNameLen = MAX_SYM_NAME;

        if(::SymGetTypeFromName(ctx.hprocess, ctx.modbase, ctx.current.this_type.c_str(), sinfo)) {
            ok = true;
            tindex = sinfo->TypeIndex;
        }
    }
    if(ok) {
        EachMembers(ctx, tindex, f);
        f(&ctx.members[0], ctx.members.size());
    }
    return ok;
}

istAPI bool EachMembersByTypeName(const char *classname, const MemberInfoCallback &f)
{
    EMVContext ctx;
    ctx.current.this_type = classname;
    return EachMembersImpl(ctx, f);
}

istAPI bool EachMembersByTypeName(const char *classname, void *_this, const MemberInfoCallback &f)
{
    EMVContext ctx;
    ctx.current.this_pointer = ctx.current.base_pointer = _this;
    ctx.current.this_type = classname;
    return EachMembersImpl(ctx, f);
}

istAPI bool EachMembersByPointer(void *_this, const MemberInfoCallback &f)
{
    EMVContext ctx;
    {
        char buf[sizeof(SYMBOL_INFO)+MAX_SYM_NAME];
        PSYMBOL_INFO sinfo = (PSYMBOL_INFO)buf;
        sinfo->SizeOfStruct = sizeof(SYMBOL_INFO);
        sinfo->MaxNameLen = MAX_SYM_NAME;

        // vftable のシンボル名が "class名::`vftable'" になっているので、そこから class 名を取得
        if(::SymFromAddr(ctx.hprocess, (DWORD64)((void***)_this)[0], nullptr, sinfo)) {
            char vftable[MAX_SYM_NAME];
            ::UnDecorateSymbolName(sinfo->Name, vftable, MAX_SYM_NAME, UNDNAME_NAME_ONLY);
            if(char *colon=strstr(vftable, "::`vftable'")) {
                *colon = '\0';
                ctx.current.this_pointer = ctx.current.base_pointer = _this;
                ctx.current.this_type = vftable;
            }
        }
    }
    if(!ctx.current.this_type.empty()) {
        return EachMembersImpl(ctx, f);
    }
    return false;
}

} // namespace ist
