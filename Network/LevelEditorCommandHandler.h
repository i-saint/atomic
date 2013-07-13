#ifndef atm_Network_LevelEditorCommandHandler_h
#define atm_Network_LevelEditorCommandHandler_h
#include "externals.h"
#include "LevelEditorCommand.h"

namespace atm {

#ifdef atm_enable_LevelEditorServer

using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
enum EntityClassID;

inline void GetDecodedRequestBody(HTTPServerRequest &request, std::string &out)
{
    if(!request.hasContentLength() || request.getContentLength()>1024*64) {
        return;
    }
    size_t size = (size_t)request.getContentLength();
    std::istream& stream = request.stream();
    std::string encoded_data;
    encoded_data.resize(size);
    stream.read(&encoded_data[0], size);
    Poco::URI::decode(encoded_data, out);
}

template<class Func>
inline void EachFormData(const std::string &form_fata, const Func &f)
{
    size_t pos = 0;
    size_t sep = 0;
    for(;;) {
        sep = form_fata.find_first_of('&', pos);
        if(sep==std::string::npos) {
            sep = form_fata.size();
            break;
        }
        f(&form_fata[pos], sep-pos);
        pos = sep+1;
    }
    f(&form_fata[pos], sep-pos);
}

inline bool ParseArg(variant &out, const std::string &str)
{
    ivec4 iv;
    uvec4 uv;
    vec4 fv;
    if(sscanf(str.c_str(), "int32(%d)", &iv.x)==1) {
        out=iv; return true;
    }
    else if(sscanf(str.c_str(), "uint32(%u)", &uv.x)==1) {
        out=uv; return true;
    }
    else if(sscanf(str.c_str(), "float32(%f)", &fv.x)==1) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "vec2(%f,%f)", &fv.x, &fv.y)==2) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "vec3(%f,%f,%f)", &fv.x, &fv.y, &fv.z)==3) {
        out=fv; return true;
    }
    else if(sscanf(str.c_str(), "instruction(%f,%f,%f,%u)", &fv.x, &fv.y, &fv.z, &uv.x)==4) {
        out=ist::MakeValueList(vec3(fv),uv.x); return true;
    }
    return false;
}


class NucleiCommandHandler : public Poco::Net::HTTPRequestHandler
{
public:
    enum ReturnCode {
        RC_Ok = 0,
        RC_InvalidCommand = -1,
        RC_InvalicParam = -2,
    };
    typedef std::function<void (NucleiCommandHandler*, HTTPServerRequest&, HTTPServerResponse&)> Handler;
    typedef stl::map<stl::string, Handler> HandlerTable;
    static HandlerTable&    getHandlerTable();
    static Handler*         findHandler(const stl::string &path);
    static FunctionID       getValidFID(const stl::string &name);
    static bool             isDeployable(EntityClassID ecid);

public:
    NucleiCommandHandler();
    void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response);
    void respondCode(HTTPServerResponse &response, int32 code);

    void handleCreate(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleDelete(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleCall(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleState(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleEntity(HTTPServerRequest &request, HTTPServerResponse &response);
    void handleConst(HTTPServerRequest &request, HTTPServerResponse &response);
};

#endif // atm_enable_LevelEditorServer

} // namespace atm
#endif // atm_Network_LevelEditorCommandHandler_h
