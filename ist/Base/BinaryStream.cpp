#include "istPCH.h"
#include "ist/Base/BinaryStream.h"

namespace ist {

inline int32 GetCRTSeekDir(bistream::seekg_dir dir) {
    switch(dir) {
    case bistream::seekg_begin: return SEEK_SET;
    case bistream::seekg_cur: return SEEK_CUR;
    case bistream::seekg_end: return SEEK_END;
    }
    return SEEK_SET;
}
inline int32 GetCRTSeekDir(bostream::seekp_dir dir) {
    return GetCRTSeekDir((bistream::seekg_dir)dir);
}

inline std::ios::seekdir GetSTDSeekDir(bistream::seekg_dir dir) {
    switch(dir) {
    case bistream::seekg_begin: return std::ios::beg;
    case bistream::seekg_cur: return std::ios::cur;
    case bistream::seekg_end: return std::ios::end;
    }
    return SEEK_SET;
}
inline std::ios::seekdir GetSTDSeekDir(bostream::seekp_dir dir) {
    return GetSTDSeekDir((bistream::seekg_dir)dir);
}



bistream::~bistream() {}

bostream::~bostream() {}


bfilestream::bfilestream() : m_file(NULL)
{}

bfilestream::bfilestream(const char *path, const char *mode)
{
    open(path, mode);
}

bfilestream::~bfilestream()
{
    if(m_file!=NULL) {
        fclose(m_file);
    }
}

bool bfilestream::open(const char *path, const char *mode)
{
    m_file = fopen(path, mode);
    return isOpened();
}

bool bfilestream::isOpened() const
{
    return m_file!=NULL;
}

uint64 bfilestream::read(void* p, uint64 s)         { return fread(p, 1, (size_t)s, m_file); }
uint64 bfilestream::tellg() const                   { return ftell(m_file); }
void bfilestream::seekg(uint64 pos, seekg_dir dir)  { fseek(m_file, (size_t)pos, GetCRTSeekDir(dir)); }

uint64 bfilestream::write(const void* p, uint64 s)  { return fwrite(p, 1, (size_t)s, m_file); }
uint64 bfilestream::tellp() const                   { return ftell(m_file); }
void bfilestream::seekp(uint64 pos, seekp_dir dir)  { fseek(m_file, (size_t)pos, GetCRTSeekDir(dir)); }



bstdistream::bstdistream(std::istream& s) : m_is(*s.rdbuf()) {}
bstdistream::bstdistream(std::streambuf& s) : m_is(s) {}
uint64 bstdistream::read(void* p, uint64 s)         { return m_is.sgetn(reinterpret_cast<char*>(p), s); }
uint64 bstdistream::tellg() const                   { return m_is.pubseekoff(0, std::ios::cur, std::ios::in); }
void bstdistream::seekg(uint64 pos, seekg_dir dir)  { m_is.pubseekoff(pos, GetSTDSeekDir(dir), std::ios::in); }


bstdostream::bstdostream(std::ostream& s) : m_os(*s.rdbuf()) {}
bstdostream::bstdostream(std::streambuf& s) : m_os(s) {}
uint64 bstdostream::write(const void* p, uint64 s)  { return m_os.sputn(reinterpret_cast<const char*>(p), s); }
uint64 bstdostream::tellp() const                   { return m_os.pubseekoff(0, std::ios::cur, std::ios::out); }
void bstdostream::seekp(uint64 pos, seekp_dir dir)  { m_os.pubseekoff(pos, GetSTDSeekDir(dir), std::ios::out); }



bstdiostream::bstdiostream(std::iostream& s) : bstdistream(s), bstdostream(s) {}
bstdiostream::bstdiostream(std::streambuf& s) : bstdistream(s), bstdostream(s) {}

#ifdef __ist_with_zlib__
gzbiostream::gzbiostream() : m_gz(NULL)
{
}

gzbiostream::gzbiostream(const char *path, const char *mode) : m_gz(NULL)
{
    open(path, mode);
}

gzbiostream::~gzbiostream()
{
    close();
}

bool gzbiostream::open(const char *path, const char *mode)
{
    close();
    m_gz = gzopen(path, mode);
    return isOpened();
}

void gzbiostream::close()
{
    if(m_gz) {
        gzclose(m_gz);
        m_gz = NULL;
    }
}

bool gzbiostream::isOpened() const   { return m_gz!=NULL; }
bool gzbiostream::isEOF() const      { return gzeof(m_gz)==1; }

uint64 gzbiostream::write(const void* p, uint64 s)  { return gzwrite(m_gz, p, (uint32)s); }
uint64 gzbiostream::tellp() const                   { return gztell(m_gz); }
void gzbiostream::seekp(uint64 p, seekp_dir dir)    { gzseek(m_gz, (uint32)p, GetCRTSeekDir(dir)); }

uint64 gzbiostream::read(void* p, uint64 s)     { return gzread(m_gz, p, (uint32)s); }
uint64 gzbiostream::tellg() const               { return gztell(m_gz); }
void gzbiostream::seekg(uint64 p, seekg_dir dir){ gzseek(m_gz, (uint32)p, GetCRTSeekDir(dir)); }
#endif // __ist_with_zlib__

} // namespace ist
