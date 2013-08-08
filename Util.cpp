#include "stdafx.h"
#include <ctime>
#include "types.h"
#include "Game/AtomicApplication.h"
#include "Game/AtomicGame.h"
#include "Game/World.h"
#include "Game/CollisionModule.h"
#include "Game/EntityModule.h"
#include "Game/EntityClass.h"
#include "Game/EntityQuery.h"
#include "Graphics/ResourceManager.h"
#include "Graphics/Renderer.h"
#include "Graphics/Shader.h"
#include "Util.h"

namespace atm {

atmAPI void FillScreen( const vec4 &color )
{
    i3d::DeviceContext *dc = atmGetGLDeviceContext();
    AtomicShader *sh_fill   = atmGetShader(SH_FILL);
    VertexArray *va_quad    = atmGetVertexArray(VA_SCREEN_QUAD);
    Buffer *ubo_params      = atmGetUniformBuffer(UBO_FILL_PARAMS);
    static uint32 location  = sh_fill->getUniformBlockIndex("fill_params");

    FillParams params;
    params.Color = color;
    MapAndWrite(dc, ubo_params, &params, sizeof(params));

    sh_fill->bind();
    sh_fill->setUniformBlock(location, GLSL_FILL_BINDING, ubo_params);
    dc->setVertexArray(va_quad);
    dc->draw(i3d::I3D_QUADS, 0, 4);
    sh_fill->unbind();
}


atmAPI vec2 GenRandomVector2()
{
    vec2 axis( atmGenRandFloat(), atmGenRandFloat() );
    axis -= vec2(0.5f, 0.5f);
    axis *= 2.0f;
    return axis;
}

atmAPI vec3 GenRandomVector3()
{
    vec3 axis( atmGenRandFloat(), atmGenRandFloat(), atmGenRandFloat() );
    axis -= 0.5f;
    axis *= 2.0f;
    return axis;
}

atmAPI vec2 GenRandomUnitVector2()
{
    return glm::normalize(GenRandomVector2());
}

atmAPI vec3 GenRandomUnitVector3()
{
    return glm::normalize(GenRandomVector3());
}

atmAPI void CreateDateString(char *buf, uint32 len)
{
    time_t t = ::time(0);
    tm *l = ::localtime(&t);
    istSNPrintf(buf, len, "%d%02d%02d_%02d%02d%02d",
        l->tm_year+1900, l->tm_mon+1, l->tm_mday, l->tm_hour, l->tm_min, l->tm_sec);
}

atmAPI bool mkdir( const char *path )
{
    Poco::File dir(path);
    return dir.createDirectory();
}

/*
    example:
    {
        std::string res;
        HTTPGet("http://google.com/", [&](std::istream &is){
            char buf[1024];
            while(!is.eof()) {
                is.read(buf, _countof(buf));
                res += std::string(buf, (size_t)is.gcount());
            }
        });
        ::OutputDebugStringA(res.c_str());
    }
*/
atmAPI void HTTPGet(const char *url, const std::function<void (std::istream &res)> &on_complete, const std::function<void (int32)> &on_fail)
{
    try {
        Poco::URI uri(url);
        Poco::Net::HTTPClientSession session(uri.getHost(), uri.getPort());
        session.setTimeout(Poco::Timespan(1,0)); // 1 sec

        std::string path(uri.getPathAndQuery());
        if(path.empty()) {
            path = "/";
        }

        Poco::Net::HTTPRequest req(Poco::Net::HTTPRequest::HTTP_GET, path, Poco::Net::HTTPMessage::HTTP_1_1);
        session.sendRequest(req);

        Poco::Net::HTTPResponse res;
        if(res.getStatus()==Poco::Net::HTTPResponse::HTTP_OK) {
            std::istream &is = session.receiveResponse(res);
            on_complete(is);
        }
        else {
            if(on_fail) {
                on_fail(res.getStatus());
            }
        }
    }
    catch(...) {
        if(on_fail) {
            on_fail(0);
        }
    }
}

class HTTPGetAsyncImpl : public ist::Thread
{
private:
    std::string m_url;
    std::function<void (std::istream &res)> m_on_complete;
    std::function<void (int32)> m_on_fail;

public:
    HTTPGetAsyncImpl(const char *url, const std::function<void (std::istream &res)> &on_complete, const std::function<void (int32)> &on_fail)
        : m_url(url)
        , m_on_complete(on_complete)
        , m_on_fail(on_fail)
    {
        setName("HTTPGetAsync");
    }

    void exec() override
    {
        HTTPGet(m_url.c_str(), m_on_complete, m_on_fail);
        istDelete(this);
    }
};

atmAPI void HTTPGetAsync( const char *url, const std::function<void (std::istream &res)> &on_complete, const std::function<void (int32)> &on_fail )
{
    auto *impl = istNew(HTTPGetAsyncImpl)(url, on_complete, on_fail);
    impl->run();
}


} // namespace atm
