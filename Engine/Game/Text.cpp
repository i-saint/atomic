#include "atmPCH.h"
#include "Text.h"

namespace atm {

    static const wchar_t* g_jp[txt_End];
    static const wchar_t* g_en[txt_End];
    static const wchar_t** g_text = g_jp;

    bool InitializeText()
    {
        // error message
        
        g_jp[txt_OpenGL_330NotSupported]    = L"OpenGL 3.3 がサポートされていません。\nドライバのアップデートで改善されない場合、申し訳ありませんがお使いの PC ではこのプログラムは動きません。";
        g_en[txt_OpenGL_330NotSupported]    = L"OpenGL 3.3 is not supported on your video card";

        // system message

        return true;
    }

    void FinalizeText()
    {

    }

    void SetLanguage( Language lid )
    {
        switch(lid) {
        case lang_JP: g_text = g_jp; break;
        case lang_EN: g_text = g_en; break;
        }
    }

    const wchar_t* GetText(TextID tid)
    {
        return g_text[tid];
    }

} // namespace atm
