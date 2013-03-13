#include "istPCH.h"
#include "i3dglQuery.h"

namespace ist {
namespace i3dgl {

Query_Base::Query_Base()
    : m_query(0)
{
    glGenQueries(1, &m_query);
}

Query_Base::~Query_Base()
{
    glDeleteQueries(1, &m_query);
}

GLuint Query_Base::getHandle() const { return m_query;  }

void Query_TimeElapsed::begin()
{
    glBeginQuery(GL_TIME_ELAPSED, getHandle());
}

float32 Query_TimeElapsed::end()
{
    glEndQuery(GL_TIME_ELAPSED);

    GLint available = 0;
    while (!available) {
        glGetQueryObjectiv(getHandle(), GL_QUERY_RESULT_AVAILABLE, &available);
    }
    GLuint result;
    glGetQueryObjectuiv(getHandle(), GL_QUERY_RESULT, &result);
    return static_cast<float32>(result)/1000000.0f;

}

} // namespace i3dgl
} // namespace ist
