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
        Stream *m_stream; // shared
        Buffer *m_internal_buffers[s_num_buffers];  // internal
        int m_buffer_index;

    public:
        StreamSource() : m_stream(NULL), m_buffer_index(0)
        {
            for(size_t i=0; i<_countof(m_internal_buffers); ++i) {
                m_internal_buffers[i] = new Buffer();
            }
        }

        virtual ~StreamSource()
        {
            for(size_t i=0; i<_countof(m_internal_buffers); ++i) {
                delete m_internal_buffers[i];
            }
        }

        void setStream(Stream *ptr) { m_stream=ptr; fillBuffer(); }

        Stream* getStream() { return m_stream; }

        size_t size() { return m_stream->size(); }
        size_t tell() { return m_stream->tell(); }
        void seek(size_t v) { return m_stream->seek(v); }
        bool eof() { return m_stream->eof(); }

        void fillBuffer()
        {
            for(int i=0; i<_countof(m_internal_buffers); ++i) {
                if(eof()) { break; }

                Buffer *buf = m_internal_buffers[i];
                char_cont& tmp = m_stream->readMillisec(250);
                buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
                queue(buf);
            }
        }

        void update()
        {
            int processed = getProcessed();
            for(int i=0; i<processed; ++i) {
                if(eof()) { return; }

                char_cont& tmp = m_stream->readMillisec(250);
                unqueue();
                Buffer *buf = m_internal_buffers[m_buffer_index];
                buf->copy(&tmp[0], tmp.size(), m_stream->getALFormat(), m_stream->getSampleRate());
                queue(buf);

                m_buffer_index = (m_buffer_index+1) % s_num_buffers;
            }
        }
    };


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
