#include "stdafx.h"
#include "Camera.h"

namespace ist {
namespace graphics {


bool Viewport::bind() const
{
    glViewport(getX(), getY(), getWidth(), getHeight());
    return true;
}


bool Camera::bind() const
{
    const float *pos = (const float*)&getPosition();
    const float *tar = (const float*)&getTarget();
    const float *dir = (const float*)&getDirection();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(
        pos[0], pos[1], pos[2],
        tar[0], tar[1], tar[2],
        dir[0], dir[1], dir[2]);
    return true;
}

bool OrthographicCamera::bind() const
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(getLeft(), getRight(), getBottom(), getTop(), getZNear(), getZFar());
    return super::bind();
}


bool PerspectiveCamera::bind() const
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(getFovy(), getAspect(), getZNear(), getZFar());
    return super::bind();
}


} // namespace graphics
} // namespace ist
