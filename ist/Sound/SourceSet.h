#ifndef __ist_Sound_SourceSet__
#define __ist_Sound_SourceSet__

namespace ist {
namespace sound {


    class StreamSource : public Source
    {
    public:
        typedef std::vector<char> char_cont;

    private:
        static const int s_num_buffers = 4;
        StreamPtr m_stream;

    public:
        StreamSource(StreamPtr ptr) : m_stream(ptr)
        {
            fillBuffer();
        }

        void setStream(StreamPtr ptr)
        {
            m_stream = ptr;
            fillBuffer();
        }

        size_t size() { return m_stream->size(); }
        size_t tell() { return m_stream->tell(); }
        void seek(size_t v) { return m_stream->seek(v); }
        bool eof() { return m_stream->eof(); }

        void fillBuffer()
        {
            for(int i=0; i<s_num_buffers; ++i) {
                if(eof()) {
                    break;
                }
                BufferPtr buf(new Buffer());
                char_cont& tmp = m_stream->readMillisec(250);
                buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
                queue(buf);
            }
        }

        void update()
        {
            int processed = getProcessed();
            for(int i=0; i<processed; ++i) {
                if(eof()) {
                    return;
                }
                char_cont& tmp = m_stream->readMillisec(250);
                BufferPtr buf = unqueue();
                buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
                queue(buf);
            }
        }
    };
    typedef boost::shared_ptr<StreamSource> StreamSourcePtr;


    template<class SourceType>
    class TSourceSet
    {
    public:
        typedef boost::shared_ptr<SourceType> ptr_t;
        typedef std::vector<ptr_t> source_cont;
        typedef std::vector<ALuint> handle_cont;
    private:
        source_cont m_sources;
        handle_cont m_handles;

    public:
        TSourceSet()
        {}

        virtual ~TSourceSet()
        {}

        virtual void append(ptr_t v)
        {
            m_sources.push_back(v);
            gatherHandle();
        }

        virtual void gatherHandle()
        {
            m_handles.clear();
            for(size_t i=0; i<m_sources.size(); ++i) {
                m_handles.push_back(m_sources[i]->getHandle());
            }
        }

        virtual size_t getSourceSize() const { return m_sources.size(); }
        virtual ptr_t getSource(size_t i) { return m_sources[i]; }


        virtual void setGain(float v)
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                getSource(i)->setGain(v);
            }
        }


        virtual bool isPlaying()
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                if(getSource(i)->isPlaying()) {
                    return true;
                }
            }
            return false;
        }

        virtual void play()
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                getSource(i)->play();
            }
        }

        virtual void update()
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                getSource(i)->update();
            }
        }
    };
    typedef TSourceSet<Source> SourceSet;

    class StreamSourceSet : public TSourceSet<StreamSource>
    {
    public:

        void seek(size_t size)
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                getSource(i)->seek(size);
            }
        }

        bool eof()
        {
            for(size_t i=0; i<getSourceSize(); ++i) {
                if(!getSource(i)->eof()) {
                    return false;
                }
            }
            return true;
        }
    };


} // namespace sound
} // namespace ist

#endif // __ist_Sound_SourceSet__
