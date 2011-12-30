#ifndef __ist_Sound__
#define __ist_Sound__

#include <AL/al.h>
#include <AL/alc.h>
#include <vector>
#include <deque>
#include <boost/shared_ptr.hpp>

#include "Base.h"
#include "Sound/Buffer.h"
#include "Sound/Stream.h"
#include "Sound/Source.h"
#include "Sound/SourceSet.h"
#include "Sound/Listener.h"
#include "Sound/SoundUtil.h"
#include "Sound/System.h"
#ifdef __ist_with_oggvorbis__
    #include <vorbis/vorbisfile.h>
    #include "Sound/OggVorbis.h"
    #pragma comment(lib, "libogg.lib")
    #pragma comment(lib, "libvorbis.lib")
    #pragma comment(lib, "libvorbisfile.lib")
    //#pragma comment(lib, "libogg_static.lib")
    //#pragma comment(lib, "libvorbis_static.lib")
    //#pragma comment(lib, "libvorbisfile_static.lib")
#endif
#pragma comment(lib, "OpenAL32.lib")
//#pragma comment(lib, "EFX-Util.lib")

#endif // __ist_Sound__
