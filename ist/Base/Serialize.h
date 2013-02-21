#ifndef ist_Base_Serialize_h
#define ist_Base_Serialize_h
#include "ist/Config.h"

// class の内側から serialize 可能にする場合：
// struct vec4
// {
//      float x,y,z,w;
//      istSerializeBlock(
//          istSerialize(x)
//          istSerialize(y)
//          istSerialize(z)
//          istSerialize(w)
//      )
// };
//
// class の外側から serialize 可能にする場合：
// istSerializeBlockEx(
//    vec4& v,
//    istSerialize(v.x)
//    istSerialize(v.y)
//    istSerialize(v.z)
//    istSerialize(v.w)
// )


#define istSerializeBlock(...)\
private:\
    friend class boost::serialization::access;\
    template<class A>\
    void serialize(A &ar, const uint32 version)\
    {\
        __VA_ARGS__\
    }

#define istSerializeSaveBlock(...)\
private:\
    friend class boost::serialization::access;\
    BOOST_SERIALIZATION_SPLIT_MEMBER();\
    template<class A>\
    void save(A &ar, const uint32 version) const\
    {\
        __VA_ARGS__\
    }

#define istSerializeLoadBlock(...)\
    template<class A>\
    void load(A& ar, const uint32 version)\
    {\
        __VA_ARGS__\
    }

#define istSerializeExportClass(ClassName)\
    BOOST_CLASS_EXPORT(ClassName)

#define istSerializePrimitive(ClassName)\
    BOOST_CLASS_IMPLEMENTATION(ClassName, boost::serialization::primitive_type)

#define istSerializeBlockEx(ValueType, ...)\
namespace boost {\
namespace serialization {\
    template <class Archive>\
    void serialize(Archive& ar, ValueType, const uint32 version)\
    {\
        __VA_ARGS__\
    }\
}}

#define istSerialize(Value)\
    ar & Value;

#define istSerializeBase(BaseClass)\
    ar & boost::serialization::base_object<BaseClass>(*this);



#endif // ist_Base_Serialize_h
