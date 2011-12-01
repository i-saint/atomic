#ifndef __ist_Sound_Util__
#define __ist_Sound_Util__

namespace ist {
namespace sound {

BufferPtr CreateBufferFromWaveFile(const char* filepath);
BufferPtr CreateBufferFromOggFile(const char* filepath);

StreamPtr CreateStreamFromWaveFile(const char* filepath);
StreamPtr CreateStreamFromOggFile(const char* filepath);

} // namespace sound
} // namespace ist

#endif // __ist_Sound_Util__
