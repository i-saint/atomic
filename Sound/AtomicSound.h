#ifndef atm_Sound_AtomicSound_h
#define atm_Sound_AtomicSound_h

#include "SoundResourceID.h"

namespace atm {

class SoundThread;

class atmAPI AtomicSound
{
private:
    static AtomicSound  *s_instance;

    SoundThread         *m_sound_thread;

    AtomicSound();

public:
    ~AtomicSound();
    static bool initializeInstance();
    static void finalizeInstance();
    static AtomicSound* getInstance();

    void setListenerPosition(const vec3 &pos);

    void playSE(SE_CHANNEL channel, SE_RID se, const vec3 &pos, bool _override);
    void haltSE(SE_CHANNEL channel);
    bool isSEPlaying(SE_CHANNEL channel);

    // 
    void playBGM(BGM_CHANNEL channel, BGM_RID bgm);
    void fadeBGM(BGM_CHANNEL channel, uint32 ms);
    void haltBGM(BGM_CHANNEL channel);
    bool isBGMPlaying(BGM_CHANNEL channel);
};


#define atmGetSound()                            AtomicSound::getInstance()

#define atmSetListenerPosition(move)             atmGetSound()->setListenerPosition(move)
#define atmPlaySE(channel, se, move, _override)  atmGetSound()->playSE(channel, se, move, _override)
#define atmHaltSE(channel)                       atmGetSound()->haltSE(channel)
#define atmIsSEPlaying(channel)                  atmGetSound()->isSEPlaying(channel)

#define atmPlayBGM(channel, bgm)                 atmGetSound()->playBGM(channel, bgm)
#define atmFadeBGM(channel, fade_ms)                  atmGetSound()->fadeBGM(channel, fade_ms)
#define atmHaltBGM(channel)                      atmGetSound()->haltBGM(channel)
#define atmIsBGMPlaying(channel)                 atmGetSound()->isBGMPlaying(channel)

} //namespace atm
#endif // atm_Sound_AtomicSound_h
