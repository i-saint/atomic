#ifndef ist_Base_BinaryStream_h
#define ist_Base_BinaryStream_h

#include "ist/Config.h"
#include "ist/Base/Types.h"
#include "ist/stdex/ist_raw_vector.h"

namespace ist {

class istAPI IBinaryStream
{
istNonCopyable(IBinaryStream);
public:
    enum SeekDir {
        Seek_Begin,
        Seek_End,
        Seek_Current,
    };
    IBinaryStream() {}
    virtual ~IBinaryStream()=0;

    virtual uint64 read(void* p, uint64 s)=0;
    virtual uint64 getReadPos() const=0;
    virtual void setReadPos(uint64 pos, SeekDir dir=Seek_Begin)=0;

    virtual uint64 write(const void* p, uint64 s)=0;
    virtual uint64 getWritePos() const=0;
    virtual void setWritePos(uint64 pos, SeekDir dir=Seek_Begin)=0;
};


class FileStream : public IBinaryStream
{
public:
    FileStream();
    FileStream(const char *path, const char *mode);
    ~FileStream();

    bool open(const char *path, const char *mode);
    bool isOpened() const;
    bool isEOF() const;

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 getReadPos() const;
    virtual void setReadPos(uint64 pos, SeekDir dir=Seek_Begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 getWritePos() const;
    virtual void setWritePos(uint64 pos, SeekDir dir=Seek_Begin);

private:
    FILE *m_file;

private:
    // non copyable
    FileStream(const FileStream&);
    FileStream& operator=(const FileStream&);
};


class MemoryStream : public IBinaryStream
{
public:
    MemoryStream();
    ~MemoryStream();

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 getReadPos() const;
    virtual void setReadPos(uint64 pos, SeekDir dir=Seek_Begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 getWritePos() const;
    virtual void setWritePos(uint64 pos, SeekDir dir=Seek_Begin);

private:
    ist::raw_vector<char> m_buffer;
    size_t m_readpos;
    size_t m_writepos;
};

class IntrusiveMemoryStream : public IBinaryStream
{
public:
    IntrusiveMemoryStream();
    IntrusiveMemoryStream(void *mem, size_t size);
    ~IntrusiveMemoryStream();

    void initialize(void *mem, size_t size);
    char* data();
    const char* data() const;

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 getReadPos() const;
    virtual void setReadPos(uint64 pos, SeekDir dir=Seek_Begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 getWritePos() const;
    virtual void setWritePos(uint64 pos, SeekDir dir=Seek_Begin);

private:
    char *m_memory;
    size_t m_size;
    size_t m_readpos;
    size_t m_writepos;

private:
    // non copyable
    IntrusiveMemoryStream(const IntrusiveMemoryStream&);
    IntrusiveMemoryStream& operator=(const IntrusiveMemoryStream&);
};



class STDStream : public IBinaryStream
{
public:
    explicit STDStream(std::iostream& s);
    explicit STDStream(std::streambuf& s);

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 getReadPos() const;
    virtual void setReadPos(uint64 pos, SeekDir dir=Seek_Begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 getWritePos() const;
    virtual void setWritePos(uint64 pos, SeekDir dir=Seek_Begin);

private:
    std::streambuf &m_io;
};


} // namespace ist


inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const char &v)                 { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const unsigned char &v)        { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const wchar_t &v)              { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const short &v)                { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const unsigned short &v)       { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const int &v)                  { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const unsigned int &v)         { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const long long &v)            { s.write(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const unsigned long long &v)   { s.write(&v, sizeof(v)); return s; }

template<class T, size_t N>
inline ist::IBinaryStream& operator<<(ist::IBinaryStream &s, const T (&v)[N])
{
    for(size_t i=0; i<N; ++i) { s.write(&v[i], sizeof(T)); }
    return s;
}

inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, char &v)                 { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, unsigned char &v)        { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, wchar_t &v)              { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, short &v)                { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, unsigned short &v)       { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, int &v)                  { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, unsigned int &v)         { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, long long &v)            { s.read(&v, sizeof(v)); return s; }
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, unsigned long long &v)   { s.read(&v, sizeof(v)); return s; }

template<class T, size_t N>
inline ist::IBinaryStream& operator>>(ist::IBinaryStream &s, T (&v)[N])
{
    for(size_t i=0; i<N; ++i) { s.read(&v[i], sizeof(T)); }
    return s;
}



namespace ist {

#ifdef ist_with_zlib
class istAPI GZFileStream : public IBinaryStream
{
public:
    GZFileStream();
    GZFileStream(const char *path, const char *mode);
    ~GZFileStream();

    bool open(const char *path, const char *mode);
    void close();

    bool isOpened() const;
    bool isEOF() const;

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 getReadPos() const;
    virtual void setReadPos(uint64 p, SeekDir dir=Seek_Begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 getWritePos() const;
    virtual void setWritePos(uint64 p, SeekDir dir=Seek_Begin);

private:
    gzFile m_gz;
private:
    // non copyable
    GZFileStream(const GZFileStream&);
    GZFileStream& operator=(const GZFileStream&);
};
#endif // ist_with_zlib

} // namespace ist

#endif // ist_Base_BinaryStream_h
