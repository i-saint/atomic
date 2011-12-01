#include "stdafx.h"
#include "ist/Base.h"
#include "ist/Sound.h"
#include "AtomicSound.h"

namespace sound = ist::sound;

namespace atomic {


    AtomicSoundThread::AtomicSoundThread()
        : m_stop_request(false)
    {
    }

    AtomicSoundThread::~AtomicSoundThread()
    {
        requestStop();
        if(m_thread) { m_thread->join(); }
    }

    void AtomicSoundThread::run()
    {
        //m_thread.reset(new boost::thread(boost::ref(*this)));
    }

    void AtomicSoundThread::operator()()
    {
        ist::SetThreadName("AtomicSoundThread");

        sound::StreamSourceSet source_set;
        {
            sound::StreamPtr sp = sound::CreateStreamFromOggFile("sound.ogg");
            if(!sp) { IST_PRINT("sound file couldn't loaded.\n"); return; }
            sound::StreamSourcePtr ssp(new sound::StreamSource(sp));
            source_set.append(ssp);
        }

        source_set.play();

        //  while(source_set.isPlaying()) {
        while(!m_stop_request) {
            ::Sleep(1);
            source_set.update();
            sound::StreamSourcePtr ssp = source_set.getSource(0);
            printf("%d / %d\r", ssp->tell(), ssp->size());
            if(source_set.eof()) {
                source_set.seek(0);
            }
        }
        IST_PRINT("sound thread end.\n");
    }

} // namespace atomic
