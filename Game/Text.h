#ifndef atm_Game_Text_h
#define atm_Game_Text_h

namespace atm {

    enum Language {
        lang_JP,
        lang_EN,
    };
    enum TextID {
        txt_OpenGL_330NotSupported,
        txt_End,
    };

    bool InitializeText();
    void FinalizeText();
    void SetLanguage(Language lid);
    const wchar_t* GetText(TextID tid);

} // namespace atm
#endif // atm_Game_Text_h
