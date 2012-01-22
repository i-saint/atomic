#include "stdafx.h"
#include "../Base.h"
#include "Constants.h"

namespace ist {
    const float32 PI = 3.1415926535897932384626433832795f;
    const float32 DEG_TO_RAD = PI / 180.0f;
    const float32 RAD_TO_DEG = 1.0f / DEG_TO_RAD;

    float32 DegToRad( float32 v )
    {
        return v * DEG_TO_RAD;
    }

    float32 RadToDeg( float32 v )
    {
        return v * RAD_TO_DEG;
    }

} // namespace ist
