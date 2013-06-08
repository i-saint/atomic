#include "stdafx.h"
#include "Text.h"

namespace atm {

    static const wchar_t* g_jp[TID_END];
    static const wchar_t* g_en[TID_END];
    static const wchar_t** g_text = g_jp;

    bool InitializeText()
    {
        // error message
        
        g_jp[TID_OPENGL330_IS_NOT_SUPPORTED]    = L"OpenGL 3.3 がサポートされていません。\nドライバのアップデートで改善されない場合、申し訳ありませんがお使いの PC ではこのプログラムは動きません。";
        g_en[TID_OPENGL330_IS_NOT_SUPPORTED]    = L"OpenGL 3.3 is not supported on your video card";
        g_jp[TID_ERROR_CUDA_NO_DEVICE]          = L"CUDA デバイスが見つかりませんでした。\n申し訳ありませんが、お使いの PC では恐らくこのプログラムは動きません。";
        g_en[TID_ERROR_CUDA_NO_DEVICE]          = L"CUDA device not found.\nthis machine can't run this program";
        g_jp[TID_ERROR_CUDA_INSUFFICIENT_DRIVER]= L"CUDA デバイスが見つかりませんでした。\nドライバのアップデートで改善されない場合、申し訳ありませんがお使いの PC ではこのプログラムは動きません。";
        g_en[TID_ERROR_CUDA_INSUFFICIENT_DRIVER]= L"CUDA device not found.\nupdating video card driver may solve this problem.";

        // system message

        return true;
    }

    void FinalizeText()
    {

    }

    void SetLanguage( LANGUAGE_ID lid )
    {
        switch(lid) {
        case LANG_JP: g_text = g_jp; break;
        case LANG_EN: g_text = g_en; break;
        }
    }

    const wchar_t* GetText(TEXT_ID tid)
    {
        return g_text[tid];
    }

} // namespace atm
