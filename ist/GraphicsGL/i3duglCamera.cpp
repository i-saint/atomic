#include "istPCH.h"
#ifdef ist_with_OpenGL
#include "ist/Base.h"
#include "ist/GraphicsGL/i3duglCamera.h"
#include "ist/GraphicsGL/i3dglTypes.h"

namespace ist {
namespace i3dgl {




void Camera::updateMatrix()
{
    m_v_matrix = glm::lookAt(vec3(m_position), vec3(m_target), vec3(m_up));
}

vec4 Camera::getDirection() const
{
     vec4 tmp = m_target-m_position;
     tmp.w = 0.0f;
     return glm::normalize(tmp);
}


void OrthographicCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::ortho( m_left, m_right, m_bottom, m_top, m_znear, m_zfar);
    m_vp_matrix = getViewMatrix()*getProjectionMatrix();
}



void PerspectiveCamera::updateMatrix()
{
    super::updateMatrix();
    m_p_matrix = glm::perspective(m_fovy, m_aspect, m_znear, m_zfar);
    m_vp_matrix = getProjectionMatrix()*getViewMatrix();
}


} // namespace i3d
} // namespace ist
#endif // ist_with_OpenGL
