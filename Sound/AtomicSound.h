#ifndef __atomic_Sound_AtomicSound__
#define __atomic_Sound_AtomicSound__

#include "SoundResourceID.h"

namespace atomic {

class SoundThread;

class AtomicSound
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

    void setListenerPosition(const vec4 &pos);

    void playSE(SE_CHANNEL channel, SE_RID se, const vec4 &pos, bool _override);
    void haltSE(SE_CHANNEL channel);
    bool isSEPlaying(SE_CHANNEL channel);

    // 
    void playBGM(BGM_CHANNEL channel, BGM_RID bgm);
    void fadeBGM(BGM_CHANNEL channel, uint32 ms);
    void haltBGM(BGM_CHANNEL channel);
    bool isBGMPlaying(BGM_CHANNEL channel);
};


#define atomicGetSound()                            AtomicSound::getInstance()

#define atomicSetListenerPosition(move)             atomicGetSound()->setListenerPosition(move)
#define atomicPlaySE(channel, se, move, _override)  atomicGetSound()->playSE(channel, se, move, _override)
#define atomicHaltSE(channel)                       atomicGetSound()->haltSE(channel)
#define atomicIsSEPlaying(channel)                  atomicGetSound()->isSEPlaying(channel)

#define atomicPlayBGM(channel, bgm)                 atomicGetSound()->playBGM(channel, bgm)
#define atomicFadeBGM(channel, fade_ms)                  atomicGetSound()->fadeBGM(channel, fade_ms)
#define atomicHaltBGM(channel)                      atomicGetSound()->haltBGM(channel)
#define atomicIsBGMPlaying(channel)                 atomicGetSound()->isBGMPlaying(channel)

} //namespace atomic
#endif // __atomic_Sound_AtomicSound__
