#ifndef atm_Game_Entity_Routine_h
#define atm_Game_Entity_Routine_h
namespace atm {

istSEnumBlock(RoutineClassID,
    istSEnum(RCID_Null),

    istSEnum(RCID_Routine_SingleShoot),
    istSEnum(RCID_Routine_CircularShoot),
    istSEnum(RCID_Routine_HomingPlayer),
    istSEnum(RCID_Routine_Pinball),

    istSEnum(RCID_End)
)

class IRoutine;
typedef IRoutine* (*RoutineCreator)();
typedef RoutineCreator (RoutineCreatorTable)[RCID_End];

RoutineCreatorTable& GetRoutineCreatorTable();
IRoutine* CreateRoutine(RoutineClassID rcid);

template<class RoutineType> IRoutine* CreateRoutine();
template<class RoutineType> class AddRoutineTable;

#define atmImplementRoutine(Class) \
    template<> IRoutine* CreateRoutine<Class>() { return istNew(Class)(); } \
    template<> struct AddRoutineTable<Class> {\
        AddRoutineTable() { GetRoutineCreatorTable()[RCID_##Class] = &CreateRoutine<Class>; }\
    };\
    AddRoutineTable<Class> g_add_routine_creator_##Class;


class IEntity;

class IRoutine
{
protected:
    IEntity *m_obj;

    istSerializeBlock(
        istSerialize(m_obj)
    )

public:
    IRoutine()  : m_obj(NULL) {}
    virtual ~IRoutine() {}
    IEntity* getEntity() { return m_obj; }
    void setEntity(IEntity *v) { m_obj=v; }

    virtual void finalize()             {}
    virtual void update(float32 dt)     {}
    virtual void asyncupdate(float32 dt){}
    virtual void draw() {}

    virtual bool call(FunctionID fid, const void *args, void *ret) { return false; }
};




} // namespace atm
#endif // atm_Game_Entity_Routine_h
