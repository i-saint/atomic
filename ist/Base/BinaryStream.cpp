#include "istPCH.h"
#include "ist/Base/BinaryStream.h"
#include "ist/Math/Misc.h"

namespace ist {

inline int32 GetCRTSeekDir(IBinaryStream::SeekDir dir) {
    switch(dir) {
    case IBinaryStream::Seek_Begin: return SEEK_SET;
    case IBinaryStream::Seek_Current: return SEEK_CUR;
    case IBinaryStream::Seek_End: return SEEK_END;
    }
    return SEEK_SET;
}

inline std::ios::seekdir GetSTDSeekDir(IBinaryStream::SeekDir dir) {
    switch(dir) {
    case IBinaryStream::Seek_Begin: return std::ios::beg;
    case IBinaryStream::Seek_Current: return std::ios::cur;
    case IBinaryStream::Seek_End: return std::ios::end;
    }
    return SEEK_SET;
}



IBinaryStream::~IBinaryStream() {}



FileStream::FileStream() : m_file(NULL)
{}

FileStream::FileStream(const char *path, const char *mode)
{
    open(path, mode);
}

FileStream::~FileStream()
{
    if(m_file!=NULL) {
        fclose(m_file);
    }
}

bool FileStream::open(const char *path, const char *mode)
{
    m_file = fopen(path, mode);
    return isOpened();
}

bool FileStream::isOpened() const   { return m_file!=NULL; }
bool FileStream::isEOF() const      { return feof(m_file)!=0; }

uint64 FileStream::read(void* p, uint64 s)              { return fread(p, 1, (size_t)s, m_file); }
uint64 FileStream::getReadPos() const                   { return ftell(m_file); }
void FileStream::setReadPos(uint64 pos, SeekDir dir)    { fseek(m_file, (size_t)pos, GetCRTSeekDir(dir)); }

uint64 FileStream::write(const void* p, uint64 s)       { return fwrite(p, 1, (size_t)s, m_file); }
uint64 FileStream::getWritePos() const                  { return ftell(m_file); }
void FileStream::setWritePos(uint64 pos, SeekDir dir)   { fseek(m_file, (size_t)pos, GetCRTSeekDir(dir)); }



MemoryStream::MemoryStream() : m_readpos(0), m_writepos(0) {}
MemoryStream::~MemoryStream() {}

uint64 MemoryStream::read(void* p, uint64 s)
{
    size_t actual_size = stl::min<size_t>(m_buffer.size()-m_readpos, (size_t)s);
    memcpy(p, &m_buffer[0]+m_readpos, (size_t)actual_size);
    m_readpos += actual_size;
    return actual_size;
}

uint64 MemoryStream::getReadPos() const { return m_readpos; }
void MemoryStream::setReadPos(uint64 pos, SeekDir dir)
{
    if(dir==Seek_Begin) {
        m_readpos = stl::min<size_t>(m_buffer.size(), (size_t)pos);
    }
    else if(dir==Seek_End) {
        m_readpos = stl::max<size_t>(m_buffer.size()-(size_t)pos, 0);
    }
    else if(dir==Seek_Current) {
        m_readpos = clamp<size_t>(m_readpos+(size_t)pos, 0, m_buffer.size());
    }
}

uint64 MemoryStream::write(const void* p, uint64 s)
{
    size_t after = m_writepos+(size_t)s;
    m_buffer.resize(after);
    memcpy(&m_buffer[0]+m_writepos, p, (size_t)s);
    return s;
}
uint64 MemoryStream::getWritePos() const { return m_writepos; }
void MemoryStream::setWritePos(uint64 pos, SeekDir dir)
{
    if(dir==Seek_Begin) {
        m_writepos = stl::min<size_t>(m_buffer.size(), (size_t)pos);
    }
    else if(dir==Seek_End) {
        m_writepos = stl::max<size_t>(m_buffer.size()-(size_t)pos, 0);
    }
    else if(dir==Seek_Current) {
        m_writepos = clamp<size_t>(m_writepos+(size_t)pos, 0, m_buffer.size());
    }
}


IntrusiveMemoryStream::IntrusiveMemoryStream() { initialize(NULL, 0); }
IntrusiveMemoryStream::IntrusiveMemoryStream(void *mem, size_t size) { initialize(mem, size); }
IntrusiveMemoryStream::~IntrusiveMemoryStream() {}

void IntrusiveMemoryStream::initialize(void *mem, size_t size)
{
    m_memory = (char*)mem;
    m_size = size;
    m_readpos = 0;
    m_writepos = 0;
}
char* IntrusiveMemoryStream::data() { return m_memory; }
const char* IntrusiveMemoryStream::data() const { return m_memory; }

uint64 IntrusiveMemoryStream::read(void* p, uint64 s)
{
    size_t actual_size = stl::min<size_t>(m_size-m_readpos, (size_t)s);
    memcpy(p, m_memory+m_readpos, (size_t)actual_size);
    m_readpos += actual_size;
    return actual_size;
}
uint64 IntrusiveMemoryStream::getReadPos() const { return m_readpos; }
void IntrusiveMemoryStream::setReadPos(uint64 pos, SeekDir dir)
{
    if(dir==Seek_Begin) {
        m_readpos = stl::min<size_t>(m_size, (size_t)pos);
    }
    else if(dir==Seek_End) {
        m_readpos = stl::max<size_t>(m_size-(size_t)pos, 0);
    }
    else if(dir==Seek_Current) {
        m_readpos = clamp<size_t>(m_readpos+(size_t)pos, 0, m_size);
    }
}

uint64 IntrusiveMemoryStream::write(const void* p, uint64 s)
{
    size_t actual_size = stl::min<size_t>(m_size-m_writepos, (size_t)s);
    memcpy(m_memory+m_writepos, p, actual_size);
    return actual_size;
}
uint64 IntrusiveMemoryStream::getWritePos() const { return m_writepos; }
void IntrusiveMemoryStream::setWritePos(uint64 pos, SeekDir dir)
{
    if(dir==Seek_Begin) {
        m_writepos = stl::min<size_t>(m_size, (size_t)pos);
    }
    else if(dir==Seek_End) {
        m_writepos = stl::max<size_t>(m_size-(size_t)pos, 0);
    }
    else if(dir==Seek_Current) {
        m_writepos = clamp<size_t>(m_writepos+(size_t)pos, 0, m_size);
    }
}



STDStream::STDStream(std::iostream &s) : m_io(*s.rdbuf())   {}
STDStream::STDStream(std::streambuf &s) : m_io(s)           {}

uint64 STDStream::read(void* p, uint64 s)           { return m_io.sgetn(reinterpret_cast<char*>(p), s); }
uint64 STDStream::getReadPos() const                { return m_io.pubseekoff(0, std::ios::cur, std::ios::in); }
void STDStream::setReadPos(uint64 pos, SeekDir dir) { m_io.pubseekoff(pos, GetSTDSeekDir(dir), std::ios::in); }

uint64 STDStream::write(const void* p, uint64 s)    { return m_io.sputn(reinterpret_cast<const char*>(p), s); }
uint64 STDStream::getWritePos() const               { return m_io.pubseekoff(0, std::ios::cur, std::ios::out); }
void STDStream::setWritePos(uint64 pos, SeekDir dir){ m_io.pubseekoff(pos, GetSTDSeekDir(dir), std::ios::out); }


#ifdef __ist_with_zlib__
GZFileStream::GZFileStream() : m_gz(NULL)
{
}

GZFileStream::GZFileStream(const char *path, const char *mode) : m_gz(NULL)
{
    open(path, mode);
}

GZFileStream::~GZFileStream()
{
    close();
}

bool GZFileStream::open(const char *path, const char *mode)
{
    close();
    m_gz = gzopen(path, mode);
    return isOpened();
}

void GZFileStream::close()
{
    if(m_gz) {
        gzclose(m_gz);
        m_gz = NULL;
    }
}

bool GZFileStream::isOpened() const   { return m_gz!=NULL; }
bool GZFileStream::isEOF() const      { return gzeof(m_gz)==1; }

uint64 GZFileStream::write(const void* p, uint64 s) { return gzwrite(m_gz, p, (uint32)s); }
uint64 GZFileStream::getWritePos() const                  { return gztell(m_gz); }
void GZFileStream::setWritePos(uint64 p, SeekDir dir)    { gzseek(m_gz, (uint32)p, GetCRTSeekDir(dir)); }

uint64 GZFileStream::read(void* p, uint64 s)        { return gzread(m_gz, p, (uint32)s); }
uint64 GZFileStream::getReadPos() const                  { return gztell(m_gz); }
void GZFileStream::setReadPos(uint64 p, SeekDir dir)    { gzseek(m_gz, (uint32)p, GetCRTSeekDir(dir)); }
#endif // __ist_with_zlib__

} // namespace ist
