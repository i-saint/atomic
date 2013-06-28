#ifndef atm_Game_Entity_h
#define atm_Game_Entity_h

#include "EntityClass.h"
#include "Util.h"


namespace atm {


class IEntity
{
friend class EntityModule;
private:
    EntityHandle m_ehandle;

    bool isValidHandle(EntityHandle h);
    void setHandle(uint32 h) { m_ehandle=h; }

    istSerializeBlock(
        istSerialize(m_ehandle)
        )

public:
    IEntity();
    virtual ~IEntity() {}
    virtual void release() { istDelete(this); }

    uint32 getHandle() const { return m_ehandle; }

    // 初期化処理
    virtual void initialize() {}

    // 終了処理。
    // EntitySet から開放されたタイミングで呼ばれる。
    // 参照カウンタ方式を取る場合などは、EntitySet から解放され (finalize() が呼ばれ) ても delete はされないケースがありうるため、
    // デストラクタと使い分ける必要が出てくる。
    virtual void finalize() {}

    // 同期更新
    virtual void update(float32 dt) {}

    // 非同期更新。
    // Entity 間の更新は並列に行われるが、その間、衝突判定や描画などの他のモジュールの更新は行われない。
    // (それらは Entity の更新が全て終わってから行われる)
    // 位置などの更新を 1 フレーム遅らせて他モジュールとも並列に更新したかったが、それだとどうしても衝突の押し返しが不自然になるため、こうなった。
    virtual void asyncupdate(float32 dt) {}

    // 描画データを Renderer に渡す。
    // (渡すだけ。この中で i3d::Device などの描画 API を直接呼んではならない)
    virtual void draw() {}


    // fid に対応するメソッドを引数 args で呼ぶ。
    // Routine や外部スクリプトとの連動用。
    virtual bool call(FunctionID fid, const void *args, void *ret=NULL) { return false; }
};



class EntityModule : public IAtomicGameModule
{
friend class IEntity;
typedef IAtomicGameModule super;
public:
    typedef ist::vector<EntityHandle> Handles;
    typedef ist::vector<IEntity*> Entities;

public:
    EntityModule();
    ~EntityModule();

    void initialize();

    void frameBegin();
    void update(float32 dt);

    // なにもしない。
    // 不本意ながら Entity の非同期更新は update() 内で行う。 (IEntity を参照)
    void asyncupdate(float32 dt);

    void draw();
    void frameEnd();

    IEntity* getEntity(EntityHandle h);
    IEntity* createEntity(EntityClassID cid);
    void deleteEntity(EntityHandle h);

    void handleEntitiesQuery(EntitiesQueryContext &ctx);

    template<class Cond, class Func>
    void enumlateEntity(const Cond &c, const Func &f) {
        each(m_all, [&](EntityHandle h){
            if(c(h)) {
                if(IEntity *e=getEntity(h)) { f(e); }
            }
        });
    }

    template<class Func>
    void enumlateEntity(const Func &f) {
        each(m_all, [&](EntityHandle h){
            if(IEntity *e=getEntity(h)) { f(e); }
        });
    }

private:
    void generateHandle(EntityClassID classid);
    EntityHandle getGeneratedHandle();

    Entities    m_entities;
    Handles     m_all;
    Handles     m_vacants;
    Handles     m_dead;      // 死亡を検出できるようにするため、死んだ Entity の handle は 1 frame は無効なままにする必要がある。
    Handles     m_dead_prev; // 

    // 以下 serialize 不要
    EntityHandle m_tmp_handle;

    void resizeTasks(uint32 n);

    istSerializeBlock(
        istSerializeBase(super)
        istSerialize(m_entities)
        istSerialize(m_all)
        istSerialize(m_vacants)
        istSerialize(m_dead)
        istSerialize(m_dead_prev)
    )
};

} // namespace atm
#endif // atm_Game_Entity_h
