#include "stdafx.h"
#include "ist/Base.h"
#include "ist/Sound.h"
#include "types.h"
#include "AtomicSound.h"
#include "../Game/AtomicApplication.h"

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
        m_thread.reset(new boost::thread(boost::ref(*this)));
    }

    void AtomicSoundThread::operator()()
    {
        ist::SetThreadName("AtomicSoundThread");

        ist::sound::IntializeSound();
        //boost::thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(1000));

        std::string sound_data;
        const char filepath[] = "sound.ogg";
        if(FILE *f=fopen(filepath, "rb")) {
            fseek(f, 0, SEEK_END);
            size_t data_size = ftell(f);
            fseek(f, 0, SEEK_SET);
            sound_data.resize(data_size);
            fread(&sound_data[0], 1, data_size, f);
            fclose(f);
        }
        else {
            return;
        }

        sound::Listener listener;
        listener.setGain(atomicGetConfig()->sound_volume);

        sound::StreamSourceSet source_set;
        {
            sound::OggVorbisMemoryStream *ovms = new sound::OggVorbisMemoryStream();
            if(!ovms->openStream(&sound_data[0], sound_data.size())) {
                IST_PRINT("sound file couldn't loaded.\n"); return;
            }
            sound::StreamPtr sp(ovms);
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

        sound::FinalizeSound();
        IST_PRINT("sound thread end.\n");
    }

} // namespace atomic
