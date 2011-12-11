#ifndef __atomic_Game_Text__
#define __atomic_Game_Text__

namespace atomic {

    enum LANGUAGE_ID {
        LANG_JP,
        LANG_EN,
    };
    enum TEXT_ID {
        TID_ERROR_CUDA_NO_DEVICE,
        TID_ERROR_CUDA_INSUFFICIENT_DRIVER,
        TID_END,
    };

    bool InitializeText();
    void FinalizeText();
    void SetLanguage(LANGUAGE_ID lid);
    const wchar_t* GetText(TEXT_ID tid);


} // namespace atomic
#endif // __atomic_Game_Text__
