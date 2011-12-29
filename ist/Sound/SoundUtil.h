#ifndef __ist_Sound_Util__
#define __ist_Sound_Util__

namespace ist {
namespace sound {

bool CreateBufferFromWaveFile(const char *filepath, Buffer *buf);
bool CreateBufferFromOggFile(const char *filepath, Buffer *buf);

Stream* CreateStreamFromWaveFile(const char* filepath);
Stream* CreateStreamFromOggFile(const char* filepath);

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Util__
