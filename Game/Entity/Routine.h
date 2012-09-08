#ifndef __atomic_Game_Character_Routine__
#define __atomic_Game_Character_Routine__
namespace atomic {

enum ROUTINE_CLASSID
{
    ROUTINE_NULL,
    ROUTINE_SHOOT,
    ROUTINE_HOMING_PLAYER,
    ROUTINE_PINBALL,

    ROUTINE_END,
};

class IRoutine;
IRoutine* CreateRoutine(ROUTINE_CLASSID rcid);




class IEntity;

class IRoutine
{
protected:
    IEntity *m_obj;

public:
    IRoutine()  : m_obj(NULL) {}
    virtual ~IRoutine() {}
    IEntity* getEntity() { return m_obj; }
    void setEntity(IEntity *v) { m_obj=v; }

    virtual void update(float32 dt)     {}
    virtual void asyncupdate(float32 dt){}
    virtual void draw() {}

    virtual bool call(FunctionID call_id, const variant &v) { return false; }
    virtual bool query(FunctionID query_id, variant &v) const { return false; }
};


typedef IRoutine* (*RoutineCreator)();
extern RoutineCreator g_routine_creators[ROUTINE_END];
template<class RoutineType> IRoutine* CreateRoutine();
template<class RoutineType> class AddRoutineTable;

#define atomicImplementRoutine(RoutineClass, RoutineClassID) \
    template<> IRoutine* CreateRoutine<RoutineClass>() { return istNew(RoutineClass)(); } \
    template<> struct AddRoutineTable<RoutineClass> {\
        AddRoutineTable() { g_routine_creators[RoutineClassID] = &CreateRoutine<RoutineClass>; }\
    };\
    AddRoutineTable<RoutineClass> g_add_##RoutineClass;


} // namespace atomic
#endif // __atomic_Game_Character_Routine__
