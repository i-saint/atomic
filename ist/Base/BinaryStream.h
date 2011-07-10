#ifndef __ist_BinaryStream__
#define __ist_BinaryStream__
namespace ist {

class bistream
{
public:
    bistream(std::istream& s) : m_is(*s.rdbuf()) {}
    bistream(std::streambuf& s) : m_is(s) {}

    size_t read(void* p, size_t s)  { return m_is.sgetn(reinterpret_cast<char*>(p), s); }
    size_t getReadPos() const       { return m_is.pubseekoff(0, std::ios::cur, std::ios::in); }
    void setReadPos(size_t pos)     { m_is.pubseekoff(pos, std::ios::beg, std::ios::in); }

private:
    std::streambuf& m_is;
};

class bostream
{
public:
    bostream(std::ostream& s) : m_os(*s.rdbuf()) {}
    bostream(std::streambuf& s) : m_os(s) {}

    size_t write(const void* p, size_t s)   { return m_os.sputn(reinterpret_cast<const char*>(p), s); }
    size_t getWritePos() const              { return m_os.pubseekoff(0, std::ios::cur, std::ios::out); }
    void setWritePos(size_t pos)            { m_os.pubseekoff(pos, std::ios::beg, std::ios::out); }

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


inline ist::bostream& operator<<(ist::bostream &s, char &v)                 { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, unsigned char &v)        { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, wchar_t &v)              { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, short &v)                { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, unsigned short &v)       { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, int &v)                  { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, unsigned int &v)         { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, long long &v)            { s.write(&v, sizeof(v)); }
inline ist::bostream& operator<<(ist::bostream &s, unsigned long long &v)   { s.write(&v, sizeof(v)); }

inline ist::bistream& operator>>(ist::bistream &s, char &v)                 { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, unsigned char &v)        { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, wchar_t &v)              { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, short &v)                { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, unsigned short &v)       { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, int &v)                  { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, unsigned int &v)         { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, long long &v)            { s.read(&v, sizeof(v)); }
inline ist::bistream& operator>>(ist::bistream &s, unsigned long long &v)   { s.read(&v, sizeof(v)); }

#endif // __ist_BinaryStream__
