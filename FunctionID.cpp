#include "stdafx.h"
#include "FunctionID.h"
#undef atomic_FunctionID_h

#define istStringnizeEnum
#include "ist/Base/EnumString.h"
#include "FunctionID.h"
#undef istStringnizeEnum

using namespace ist;
EnumStr test[] = {
    {0, "hoge"},
    {0, "hage"},
    {0, "hige"},
};
