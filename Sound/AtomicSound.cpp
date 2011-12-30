#include "stdafx.h"
#include "ist/Base.h"
#include "ist/Sound.h"
#include "types.h"
#include "AtomicSound.h"
#include "../Game/AtomicApplication.h"

namespace sound = ist::sound;

namespace atomic {


struct SoundRequest
{
    enum REQ_TYPE {
        REQ_LISTENER_POS,
        REQ_SE_PLAY,
        REQ_SE_HALT,
        REQ_BGM_PLAY,
        REQ_BGM_FADE,
        REQ_BGM_HALT,
    };

    REQ_TYPE    type;
    union {
        struct {
            float32     pos[4];
        } listener;
        struct {
            SE_CHANNEL  ch;
            SE_RID      rid;
            bool        force;
            float32     pos[3];
        } se;
        struct {
            BGM_CHANNEL ch;
            BGM_RID     rid;
            uint32      ms;
        } bgm;
    };
};


class SoundThread
{
private:
    boost::scoped_ptr<boost::thread> m_thread;
    bool m_initialization_complete;
    bool m_stop_request;

    SpinLock                    m_se_lock;
    stl::vector<SoundRequest>   m_requests;
    stl::vector<SoundRequest>   m_reqests2;

    sound::Listener         *m_listener;
    sound::StreamSource     *m_bgm_source;
    sound::Source           *m_se_sources[SE_CHANNEL_END];

    sound::Stream           *m_stream;
    sound::Buffer           *m_se_data[SE_END];

public:
    SoundThread();
    ~SoundThread();

    void initialize();
    void finalize();

    void run();
    void requestStop() { m_stop_request=true; }

    bool isInitializationComplete() const { return m_initialization_complete; }

    void addRequest(const SoundRequest & v);

public:
    void operator()();
    void processSERequests();
};

SoundThread::SoundThread()
    : m_initialization_complete(false)
    , m_stop_request(false)
{
    m_listener = NULL;
    m_bgm_source = NULL;
    std::fill_n(m_se_sources, _countof(m_se_sources), (sound::Source*)NULL);

    m_requests.reserve(64);
    m_reqests2.reserve(64);
}

SoundThread::~SoundThread()
{
    requestStop();
    if(m_thread) { m_thread->join(); }
}

void SoundThread::initialize()
{
    m_listener = IST_NEW(sound::Listener)();
    m_bgm_source = IST_NEW(sound::StreamSource)();
    for(uint32 i=0; i<_countof(m_se_sources); ++i)  { m_se_sources[i]=IST_NEW(sound::Source)(); }
    for(uint32 i=0; i<_countof(m_se_data); ++i)     { m_se_data[i]=IST_NEW(sound::Buffer)(); }

    {
        sound::OggVorbisFileStream *ovfs = IST_NEW(sound::OggVorbisFileStream)();
        ovfs->openStream("Resources/bgm1.ogg");
        m_stream = ovfs;
    }
    for(uint32 i=0; i<_countof(m_se_sources); ++i) {
        m_se_sources[i]->setRefferenceDistance(4.0f);
        m_se_sources[i]->setMaxDistance(12.0f);
    }
    {
        CreateBufferFromWaveFile("Resources/explosion1.wav", m_se_data[SE_EXPLOSION1]);
        CreateBufferFromWaveFile("Resources/explosion2.wav", m_se_data[SE_EXPLOSION2]);
        CreateBufferFromWaveFile("Resources/explosion3.wav", m_se_data[SE_EXPLOSION3]);
        CreateBufferFromWaveFile("Resources/explosion4.wav", m_se_data[SE_EXPLOSION4]);
        CreateBufferFromWaveFile("Resources/explosion5.wav", m_se_data[SE_EXPLOSION5]);
        CreateBufferFromWaveFile("Resources/explosion6.wav", m_se_data[SE_EXPLOSION6]);
    }

    m_listener->setGain(atomicGetConfig()->sound_volume);
}

void SoundThread::finalize()
{
    IST_DELETE(m_stream);

    for(uint32 i=0; i<_countof(m_se_data); ++i)     { IST_SAFE_DELETE(m_se_data[i]); }
    for(uint32 i=0; i<_countof(m_se_sources); ++i)  { IST_SAFE_DELETE(m_se_sources[i]); }
    IST_SAFE_DELETE(m_bgm_source);
    IST_SAFE_DELETE(m_listener);
}


void SoundThread::addRequest(const SoundRequest & v)
{
    m_se_lock.lock();
    m_requests.push_back(v);
    m_se_lock.unlock();
}


void SoundThread::run()
{
    m_thread.reset(new boost::thread(boost::ref(*this)));
}

void SoundThread::operator()()
{
    ist::SetThreadName("AtomicSoundThread");

    ist::sound::IntializeSound();
    initialize();
    m_initialization_complete = true;

    m_bgm_source->setStream(m_stream);
    m_bgm_source->play();

    while(!m_stop_request) {
        ::Sleep(1);
        processSERequests();
        m_bgm_source->update();
        if(m_bgm_source->eof()) {
            m_bgm_source->seek(0);
        }
    }

    finalize();
    sound::FinalizeSound();
    IST_PRINT("sound thread end.\n");
}

void SoundThread::processSERequests()
{
    m_se_lock.lock();
    m_reqests2 = m_requests;
    m_requests.clear();
    m_se_lock.unlock();

    for(uint32 i=0; i<m_reqests2.size(); ++i) {
        SoundRequest &req = m_reqests2[i];
        switch(req.type) {
        case SoundRequest::REQ_LISTENER_POS:
            {
                m_listener->setPosition((vec3&)(req.listener.pos));
            }
            break;
        case SoundRequest::REQ_SE_PLAY:
            {
                sound::Source *src = m_se_sources[req.se.ch];
                if(src->isInitial() || req.se.force) {
                    src->clearQueue();
                    src->setPosition((vec3&)(req.se.pos));
                    src->queue(m_se_data[req.se.rid]);
                    src->play();
                }
            }
            break;
        case SoundRequest::REQ_SE_HALT:
            {
                sound::Source *src = m_se_sources[req.se.ch];
                src->stop();
            }
            break;
        }
    }
}




AtomicSound* AtomicSound::s_instance = NULL;

bool AtomicSound::initializeInstance()
{
    if(!s_instance) {
        s_instance = IST_NEW(AtomicSound)();
        return true;
    }
    return false;
}

void AtomicSound::finalizeInstance()
{
    if(s_instance) {
        IST_SAFE_DELETE(s_instance);
    }
}

AtomicSound* AtomicSound::getInstance() { return s_instance; }


AtomicSound::AtomicSound()
{
    m_sound_thread  = IST_NEW(SoundThread)();
    m_sound_thread->run();
}

AtomicSound::~AtomicSound()
{
    IST_SAFE_DELETE(m_sound_thread);
}

void AtomicSound::setListenerPosition(const vec4 &pos)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_LISTENER_POS;
    (vec4&)(req.listener.pos) = pos;
    m_sound_thread->addRequest(req);
}

void AtomicSound::playSE(SE_CHANNEL channel, SE_RID se, const vec4 &pos, bool _override)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_SE_PLAY;
    req.se.ch   = channel;
    req.se.rid  = se;
    req.se.force= _override;
    (vec4&)(req.se.pos) = pos;
    m_sound_thread->addRequest(req);
}

void AtomicSound::haltSE(SE_CHANNEL channel)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_SE_HALT;
    req.se.ch   = channel;
    m_sound_thread->addRequest(req);
}

bool AtomicSound::isSEPlaying(SE_CHANNEL channel)
{

}



void AtomicSound::playBGM(BGM_CHANNEL channel, BGM_RID bgm)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_BGM_PLAY;
    req.bgm.ch   = channel;
    req.bgm.rid  = bgm;
    m_sound_thread->addRequest(req);
}

void AtomicSound::fadeBGM(BGM_CHANNEL channel, uint32 ms)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_BGM_FADE;
    req.bgm.ch  = channel;
    req.bgm.ms  = ms;
    m_sound_thread->addRequest(req);
}

void AtomicSound::haltBGM(BGM_CHANNEL channel)
{
    SoundRequest req;
    req.type    = SoundRequest::REQ_BGM_HALT;
    req.bgm.ch  = channel;
    m_sound_thread->addRequest(req);
}

void AtomicSound::isBGMPlaying(BGM_CHANNEL channel)
{
}



} // namespace atomic
