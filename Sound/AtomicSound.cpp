#include "stdafx.h"
#include "ist/Base.h"
#include "ist/Sound.h"
#include "types.h"
#include "AtomicSound.h"
#include "../Game/AtomicApplication.h"

namespace isd = ist::isd;

namespace atm {


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
            float32     pos[3];
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
            uint32      fade_ms;
        } bgm;
    };
};


class SoundThread : public ist::Thread
{
public:
    SoundThread();
    ~SoundThread();

    void initialize();
    void finalize();

    void exec();
    void requestStop() { m_stop_request=true; }

    bool isInitializationComplete() const { return m_initialization_complete; }

    void addRequest(const SoundRequest & v);

public:
    void processRequests();

private:
    isd::Device *m_device;
    bool m_initialization_complete;
    bool m_stop_request;

    ist::Condition              m_cond_request;
    ist::Mutex                  m_mutex_request;
    ist::vector<SoundRequest>   m_requests;
    ist::vector<SoundRequest>   m_requests_temp;

    isd::Listener         *m_listener;
    isd::StreamSource     *m_bgm_source;
    isd::Source           *m_se_sources[SE_CHANNEL_END];

    isd::Stream           *m_stream;
    isd::Buffer           *m_se_data[SE_END];
};

SoundThread::SoundThread()
    : m_initialization_complete(false)
    , m_stop_request(false)
{
    m_listener = NULL;
    m_bgm_source = NULL;
    m_stream = NULL;
    stl::fill_n(m_se_sources, _countof(m_se_sources), (isd::Source*)NULL);
    stl::fill_n(m_se_data, _countof(m_se_data), (isd::Buffer*)NULL);

    m_requests.reserve(64);
    m_requests_temp.reserve(64);
}

SoundThread::~SoundThread()
{
    requestStop();
    m_cond_request.signalOne();
    join();
}

void SoundThread::initialize()
{
    m_device = isd::CreateDevice();

    m_listener = m_device->createListener();
    for(uint32 i=0; i<_countof(m_se_sources); ++i)  { m_se_sources[i] = m_device->createSource(); }
    for(uint32 i=0; i<_countof(m_se_data); ++i)     { m_se_data[i] = m_device->createBuffer(); }
    m_bgm_source = istNew(isd::StreamSource)(m_device);

    {
        isd::OggVorbisFileStream *ovfs = istNew(isd::OggVorbisFileStream)();
        ovfs->openStream("Resources/bgm1.ogg");
        m_stream = ovfs;
    }
    {
        m_bgm_source->setGain(atmGetConfig()->bgm_volume);
    }
    for(uint32 i=0; i<_countof(m_se_sources); ++i) {
        m_se_sources[i]->setRefferenceDistance(4.0f);
        m_se_sources[i]->setMaxDistance(12.0f);
        m_se_sources[i]->setGain(atmGetConfig()->se_volume);
    }
    {
        CreateBufferFromWaveFile("Resources/explosion1.wav", m_se_data[SE_EXPLOSION1]);
        CreateBufferFromWaveFile("Resources/explosion2.wav", m_se_data[SE_EXPLOSION2]);
        CreateBufferFromWaveFile("Resources/explosion3.wav", m_se_data[SE_EXPLOSION3]);
        CreateBufferFromWaveFile("Resources/explosion4.wav", m_se_data[SE_EXPLOSION4]);
        CreateBufferFromWaveFile("Resources/explosion5.wav", m_se_data[SE_EXPLOSION5]);
        CreateBufferFromWaveFile("Resources/explosion6.wav", m_se_data[SE_EXPLOSION6]);
    }
}

void SoundThread::finalize()
{
    istSafeRelease(m_stream);

    for(uint32 i=0; i<_countof(m_se_data); ++i)     { istSafeRelease(m_se_data[i]); }
    for(uint32 i=0; i<_countof(m_se_sources); ++i)  { istSafeRelease(m_se_sources[i]); }
    istSafeRelease(m_bgm_source);
    istSafeRelease(m_listener);

    istSafeRelease(m_device);
}


void SoundThread::addRequest(const SoundRequest & v)
{
    {
        ist::Mutex::ScopedLock lock(m_mutex_request);
        m_requests.push_back(v);
    }
    m_cond_request.signalOne();
}


void SoundThread::exec()
{
    ist::Thread::setNameToCurrentThread("AtomicSoundThread");

    initialize();
    m_initialization_complete = true;

    m_bgm_source->setStream(m_stream);
    m_bgm_source->play();

    while(!m_stop_request) {
        processRequests();
        m_bgm_source->update();
        if(m_bgm_source->eof()) {
            m_bgm_source->seek(0);
        }
        m_cond_request.timedWait(10);
    }

    finalize();
    istPrint("sound thread end.\n");
}

void SoundThread::processRequests()
{
    {
        // lock 範囲を最小にするため、ここではワーキング領域に移すだけ
        ist::Mutex::ScopedLock lock(m_mutex_request);
        m_requests_temp = m_requests;
        m_requests.clear();
    }

    for(size_t i=0; i<m_requests_temp.size(); ++i) {
        SoundRequest &req = m_requests_temp[i];
        switch(req.type) {
        case SoundRequest::REQ_LISTENER_POS:
            {
                m_listener->setPosition((vec3&)(req.listener.pos));
            }
            break;
        case SoundRequest::REQ_SE_PLAY:
            {
                isd::Source *src = m_se_sources[req.se.ch];
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
                isd::Source *src = m_se_sources[req.se.ch];
                src->stop();
            }
            break;
        }
    }
    m_requests_temp.clear();
}




AtomicSound* AtomicSound::s_instance = NULL;

bool AtomicSound::initializeInstance()
{
    if(!s_instance) {
        s_instance = istNew(AtomicSound)();
        return true;
    }
    return false;
}

void AtomicSound::finalizeInstance()
{
    if(s_instance) {
        istSafeDelete(s_instance);
    }
}

AtomicSound* AtomicSound::getInstance() { return s_instance; }


AtomicSound::AtomicSound()
    : m_sound_thread(NULL)
{
    if(atmGetConfig()->sound_enable) {
        m_sound_thread  = istNew(SoundThread)();
        m_sound_thread->run();
    }
}

AtomicSound::~AtomicSound()
{
    istSafeDelete(m_sound_thread);
}

void AtomicSound::setListenerPosition(const vec4 &pos)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type    = SoundRequest::REQ_LISTENER_POS;
    (vec3&)(req.listener.pos) = vec3(pos);
    m_sound_thread->addRequest(req);
}

void AtomicSound::playSE(SE_CHANNEL channel, SE_RID se, const vec4 &pos, bool _override)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type    = SoundRequest::REQ_SE_PLAY;
    req.se.ch   = channel;
    req.se.rid  = se;
    req.se.force= _override;
    (vec3&)(req.se.pos) = vec3(pos);
    m_sound_thread->addRequest(req);
}

void AtomicSound::haltSE(SE_CHANNEL channel)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type    = SoundRequest::REQ_SE_HALT;
    req.se.ch   = channel;
    m_sound_thread->addRequest(req);
}

bool AtomicSound::isSEPlaying(SE_CHANNEL channel)
{
    if(!m_sound_thread) { return false; }
    // todo
    istAssert(false);
    return false;
}



void AtomicSound::playBGM(BGM_CHANNEL channel, BGM_RID bgm)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type    = SoundRequest::REQ_BGM_PLAY;
    req.bgm.ch  = channel;
    req.bgm.rid = bgm;
    m_sound_thread->addRequest(req);
}

void AtomicSound::fadeBGM(BGM_CHANNEL channel, uint32 ms)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type        = SoundRequest::REQ_BGM_FADE;
    req.bgm.ch      = channel;
    req.bgm.fade_ms = ms;
    m_sound_thread->addRequest(req);
}

void AtomicSound::haltBGM(BGM_CHANNEL channel)
{
    if(!m_sound_thread) { return; }

    SoundRequest req;
    req.type    = SoundRequest::REQ_BGM_HALT;
    req.bgm.ch  = channel;
    m_sound_thread->addRequest(req);
}

bool AtomicSound::isBGMPlaying(BGM_CHANNEL channel)
{
    if(!m_sound_thread) { return false; }
    // todo
    istAssert(false);
    return false;
}



} // namespace atm
