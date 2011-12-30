#ifndef __ist_BinaryStream__
#define __ist_BinaryStream__
namespace ist {

class bistream_base
{
public:
    virtual ~bistream_base() {}
    virtual uint64 read(void* p, uint64 s)=0;
    virtual uint64 getReadPos() const=0;
    virtual void setReadPos(uint64 pos)=0;
};

class bostream_base
{
public:
    virtual ~bostream_base() {}
    virtual uint64 write(const void* p, uint64 s)=0;
    virtual uint64 getWritePos() const=0;
    virtual void setWritePos(uint64 pos)=0;
};


class bistream : public bistream_base
{
public:
    bistream(std::istream& s) : m_is(*s.rdbuf()) {}
    bistream(std::streambuf& s) : m_is(s) {}

    uint64 read(void* p, uint64 s)  { return m_is.sgetn(reinterpret_cast<char*>(p), s); }
    uint64 getReadPos() const       { return m_is.pubseekoff(0, std::ios::cur, std::ios::in); }
    void setReadPos(uint64 pos)     { m_is.pubseekoff(pos, std::ios::beg, std::ios::in); }

private:
    std::streambuf& m_is;
};

class bostream : public bostream_base
{
public:
    bostream(std::ostream& s) : m_os(*s.rdbuf()) {}
    bostream(std::streambuf& s) : m_os(s) {}

    uint64 write(const void* p, uint64 s)   { return m_os.sputn(reinterpret_cast<const char*>(p), s); }
    uint64 getWritePos() const              { return m_os.pubseekoff(0, std::ios::cur, std::ios::out); }
    void setWritePos(uint64 pos)            { m_os.pubseekoff(pos, std::ios::beg, std::ios::out); }

private:
    std::streambuf& m_os;
};

class biostream : public bistream, public bostream
{
public:
    biostream(std::iostream& s) : bistream(s), bostream(s) {}
    biostream(std::streambuf& s) : bistream(s), bostream(s) {}
};


} // namespace ist


inline ist::bostream_base& operator<<(ist::bostream_base &s, char &v)                 { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, unsigned char &v)        { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, wchar_t &v)              { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, short &v)                { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, unsigned short &v)       { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, int &v)                  { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, unsigned int &v)         { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, long long &v)            { s.write(&v, sizeof(v)); return s; }
inline ist::bostream_base& operator<<(ist::bostream_base &s, unsigned long long &v)   { s.write(&v, sizeof(v)); return s; }

inline ist::bistream_base& operator>>(ist::bistream_base &s, char &v)                 { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, unsigned char &v)        { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, wchar_t &v)              { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, short &v)                { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, unsigned short &v)       { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, int &v)                  { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, unsigned int &v)         { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, long long &v)            { s.read(&v, sizeof(v)); return s; }
inline ist::bistream_base& operator>>(ist::bistream_base &s, unsigned long long &v)   { s.read(&v, sizeof(v)); return s; }


#ifdef __ist_with_zlib__
#include "zlib/zlib.h"
#pragma comment(lib, "zdll.lib")

namespace ist {

class gzbiostream : public bistream_base, public bostream_base
{
private:
    gzFile m_gz;

public:
    gzbiostream() : m_gz(NULL)
    {
    }

    gzbiostream(const char *path, const char *mode) : m_gz(NULL)
    {
        open(path, mode);
    }

    ~gzbiostream()
    {
        close();
    }

    bool open(const char *path, const char *mode)
    {
        close();
        m_gz = gzopen(path, mode);
        return isOpened();
    }

    void close()
    {
        if(m_gz) {
            gzclose(m_gz);
            m_gz = NULL;
        }
    }

    bool isOpened() const   { return m_gz!=NULL; }
    bool isEOF() const      { return gzeof(m_gz)==1; }

    uint64 write(const void* p, uint64 s)   { return gzwrite(m_gz, p, (uint32)s); }
    uint64 getWritePos() const              { return gztell(m_gz); }
    void setWritePos(uint64 p)              { gzseek(m_gz, SEEK_SET, (uint32)p); }

    uint64 read(void* p, uint64 s)  { return gzread(m_gz, p, (uint32)s); }
    uint64 getReadPos() const       { return gztell(m_gz); }
    void setReadPos(uint64 p)       { gzseek(m_gz, SEEK_SET, (uint32)p); }
};

} // namespace ist
#endif // IST_ENABLE_ZLIB

#endif // __ist_BinaryStream__
