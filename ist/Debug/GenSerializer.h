#ifndef ist_Debug_GenSerializer_h
#define ist_Debug_GenSerializer_h
#include <functional>
#include <string>
#include <vector>
#include "EachMembers.h"

namespace ist {

istAPI std::string GSCallback_boost_serialization(const char*, const MemberInfo*, size_t);

class istAPI _GenSerializerProcess
{
public:
    typedef std::vector<_GenSerializerProcess*> instance_cont;
    typedef std::function<std::string (const char*, const MemberInfo*, size_t)> callback_t;

    _GenSerializerProcess(const char *path, const callback_t &cb);
    void process();
    static void exec();
private:
    static instance_cont& getInstances();
    const char *m_path;
    callback_t m_callback;
};

} // namespace ist


#ifndef ist_disable_GenSerializer
#   define istGenSerializerBegin(ClassName)   
#   define istGenSerializerEnd()              
#   define istGenSerializerProcess(...)       static ist::_GenSerializerProcess g_gsp##__LINE__(__FILE__, __VA_ARGS__)
#   define istGenSerializerExec()             ist::_GenSerializerProcess::exec()
#else // ist_disable_GenSerializer
#   define istGenSerializerBegin(ClassName)   
#   define istGenSerializerEnd()              
#   define istGenSerializerProcess(...)       
#   define istGenSerializerExec()             
#endif // ist_disable_GenSerializer

#endif // ist_Debug_GenSerializer_h
