#ifndef atomic_Game_Text_h
#define atomic_Game_Text_h

namespace atomic {

    enum LANGUAGE_ID {
        LANG_JP,
        LANG_EN,
    };
    enum TEXT_ID {
        TID_OPENGL330_IS_NOT_SUPPORTED,
        TID_ERROR_CUDA_NO_DEVICE,
        TID_ERROR_CUDA_INSUFFICIENT_DRIVER,
        TID_END,
    };

    bool InitializeText();
    void FinalizeText();
    void SetLanguage(LANGUAGE_ID lid);
    const wchar_t* GetText(TEXT_ID tid);


} // namespace atomic
#endif // atomic_Game_Text_h
