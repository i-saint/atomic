#ifndef __ist_BinaryStream__
#define __ist_BinaryStream__
namespace ist {

class istInterModule bistream
{
public:
    enum seekg_dir {
        seekg_begin,
        seekg_end,
        seekg_cur,
    };
    virtual ~bistream()=0;
    virtual uint64 read(void* p, uint64 s)=0;
    virtual uint64 tellg() const=0;
    virtual void seekg(uint64 pos, seekg_dir dir=seekg_begin)=0;
};

class istInterModule bostream
{
public:
    enum seekp_dir {
        seekp_begin,
        seekp_end,
        seekp_cur,
    };
    virtual ~bostream()=0;
    virtual uint64 write(const void* p, uint64 s)=0;
    virtual uint64 tellp() const=0;
    virtual void seekp(uint64 pos, seekp_dir dir=seekp_begin)=0;
};


class istInterModule bfilestream : public bistream, public bostream
{
public:
    bfilestream();
    bfilestream(const char *path, const char *mode);
    ~bfilestream();

    bool open(const char *path, const char *mode);
    bool isOpened() const;

    virtual uint64 read(void* p, uint64 s);
    virtual uint64 tellg() const;
    virtual void seekg(uint64 pos, seekg_dir dir=seekg_begin);

    virtual uint64 write(const void* p, uint64 s);
    virtual uint64 tellp() const;
    virtual void seekp(uint64 pos, seekp_dir dir=seekp_begin);

private:
    FILE *m_file;

private:
    // non copyable
    bfilestream(const bfilestream&);
    bfilestream& operator=(const bfilestream&);
};

class istInterModule bstdistream : public bistream
{
public:
    explicit bstdistream(std::istream& s);
    explicit bstdistream(std::streambuf& s);

    uint64 read(void* p, uint64 s);
    uint64 tellg() const;
    void seekg(uint64 pos, seekg_dir dir=seekg_begin);

private:
    std::streambuf& m_is;
};

class istInterModule bstdostream : public bostream
{
public:
    explicit bstdostream(std::ostream& s);
    explicit bstdostream(std::streambuf& s);

    uint64 write(const void* p, uint64 s);
    uint64 tellp() const;
    void seekp(uint64 pos, seekp_dir dir=seekp_begin);

private:
    std::streambuf& m_os;
};

class istInterModule bstdiostream : public bstdistream, public bstdostream
{
public:
    explicit bstdiostream(std::iostream& s);
    explicit bstdiostream(std::streambuf& s);
};


} // namespace ist


inline ist::bostream& operator<<(ist::bostream &s, const char &v)                 { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const unsigned char &v)        { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const wchar_t &v)              { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const short &v)                { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const unsigned short &v)       { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const int &v)                  { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const unsigned int &v)         { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const long long &v)            { s.write(&v, sizeof(v)); return s; }
inline ist::bostream& operator<<(ist::bostream &s, const unsigned long long &v)   { s.write(&v, sizeof(v)); return s; }

template<class T, size_t N>
inline ist::bostream& operator<<(ist::bostream &s, const T (&v)[N])
{
    for(size_t i=0; i<N; ++i) { s.write(&v[i], sizeof(T)); }
    return s;
}

inline ist::bistream& operator>>(ist::bistream &s, char &v)                 { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, unsigned char &v)        { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, wchar_t &v)              { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, short &v)                { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, unsigned short &v)       { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, int &v)                  { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, unsigned int &v)         { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, long long &v)            { s.read(&v, sizeof(v)); return s; }
inline ist::bistream& operator>>(ist::bistream &s, unsigned long long &v)   { s.read(&v, sizeof(v)); return s; }

template<class T, size_t N>
inline ist::bistream& operator>>(ist::bistream &s, T (&v)[N])
{
    for(size_t i=0; i<N; ++i) { s.read(&v[i], sizeof(T)); }
    return s;
}



namespace ist {

#ifdef __ist_with_zlib__
class istInterModule gzbiostream : public bistream, public bostream
{
public:
    gzbiostream();
    gzbiostream(const char *path, const char *mode);
    ~gzbiostream();

    bool open(const char *path, const char *mode);
    void close();

    bool isOpened() const;
    bool isEOF() const;

    uint64 read(void* p, uint64 s);
    uint64 tellg() const;
    void seekg(uint64 p, seekg_dir dir=seekg_begin);

    uint64 write(const void* p, uint64 s);
    uint64 tellp() const;
    void seekp(uint64 p, seekp_dir dir=seekp_begin);

private:
    gzFile m_gz;
private:
    // non copyable
    gzbiostream(const gzbiostream&);
    gzbiostream& operator=(const gzbiostream&);
};
#endif // __ist_with_zlib__

} // namespace ist

#endif // __ist_BinaryStream__
