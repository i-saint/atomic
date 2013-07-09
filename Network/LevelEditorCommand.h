#ifndef atm_Network_LevelEditorCommand_h
#define atm_Network_LevelEditorCommand_h

namespace atm {


struct EntitiesQueryContext
{
#ifdef atm_enable_WebGL
    ist::raw_vector<uint32> id;
    ist::raw_vector<mat4>   trans;
    ist::raw_vector<vec3>   size;
    ist::raw_vector<vec4>   color;
    ist::raw_vector<vec2>   bullets;
    ist::raw_vector<vec3>   lasers;
    ist::raw_vector<vec2>   fluids;

    void clear()
    {
        id.clear();
        trans.clear();
        size.clear();
        color.clear();
        bullets.clear();
        lasers.clear();
        fluids.clear();
    }

    size_t sizeByte() const
    {
        return
            sizeof(uint32)* id.size()       +
            sizeof(mat4)  * trans.size()    +
            sizeof(vec3)  * size.size()     +
            sizeof(vec4)  * color.size()    +
            sizeof(vec2)  * bullets.size()  +
            sizeof(vec3)  * lasers.size()   +
            sizeof(vec2)  * fluids.size();
    }

#else  // atm_enable_WebGL
    ist::raw_vector<uint32> id;
    ist::raw_vector<uint32> type;
    ist::raw_vector<vec2>   size;
    ist::raw_vector<vec2>   pos;

    void clear()
    {
        id.clear();
        type.clear();
        size.clear();
        pos.clear();
    }

    size_t sizeByte() const
    {
        return
            id.size()*sizeof(uint32) +
            type.size()*sizeof(uint32) +
            size.size()*sizeof(vec2) +
            pos.size()*sizeof(vec2);
    }
#endif // atm_enable_WebGL
};

// Level Editor Commans
enum LEC_Type
{
    LEC_Unknown,
    LEC_Create,
    LEC_Delete,
    LEC_Call,
};

// Level Editor Query
enum LEQ_Type
{
    LEQ_Unknown,
    LEQ_Entities,
    LEQ_Entity,
    LEQ_Players,
};


union istAlign(16) LevelEditorCommand
{
    struct {
        LEC_Type type;
        uint32 frame;
    };
    uint32 dummy[8];

    LevelEditorCommand() : type(LEC_Unknown), frame(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
    bool operator<(const LevelEditorCommand &v) const { return frame<v.frame; }
};
atmGlobalNamespace(
    istSerializeRaw(atm::LevelEditorCommand)
)

#define LEC_Ensure(T) istStaticAssert(sizeof(T)==sizeof(LevelEditorCommand))


struct istAlign(16) LevelEditorCommand_Create
{
    LEC_Type type;
    uint32 frame;
    uint32 classid;
    uint32 dummy[5];

    LevelEditorCommand_Create() : type(LEC_Create), frame(0), classid(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
};
LEC_Ensure(LevelEditorCommand_Create);


struct istAlign(16) LevelEditorCommand_Delete
{
    LEC_Type type;
    uint32 frame;
    uint32 entity_id;
    uint32 dummy[5];

    LevelEditorCommand_Delete() : type(LEC_Delete), frame(0), entity_id(0)
    {
        std::fill_n(dummy, _countof(dummy), 0);
    }
};
LEC_Ensure(LevelEditorCommand_Delete);


struct istAlign(16) LevelEditorCommand_Call
{
    LEC_Type type;
    uint32 frame;
    uint32 entity;
    uint32 function;
    variant arg;

    LevelEditorCommand_Call() : type(LEC_Call) {}
};
LEC_Ensure(LevelEditorCommand_Call);

#undef LEC_Ensure


struct LevelEditorQuery
{
    LEQ_Type type;
    uint32 optional;

    std::string response;
    bool completed;

    LevelEditorQuery()
        : optional(0)
        , completed(false)
    {}
};


} // namespace atm
#endif // atm_Network_LevelEditorCommand_h
