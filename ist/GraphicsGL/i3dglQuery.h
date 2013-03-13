#ifndef i3dgl_Query_h
#define i3dgl_Query_h
#include "i3dglDeviceResource.h"
namespace ist {
namespace i3dgl {

class Query_Base
{
public:
    Query_Base();
    ~Query_Base();
protected:
    GLuint getHandle() const;
private:
    GLuint  m_query;
};

class Query_TimeElapsed : public Query_Base
{
public:
    void begin();
    float32 end();
};
    
} // namespace i3dgl
} // namespace ist
#endif // i3dgl_Query_h
