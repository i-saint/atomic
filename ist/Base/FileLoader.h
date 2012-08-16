#ifndef __ist_TaskScheduler__
#define __ist_TaskScheduler__


namespace ist {


class LoaderTaskBase
{
public:
    enum ERROR_CODE
    {
        ERR_OK,
        ERR_CANCELED,
        ERR_FILE_NOT_EXIST,
        ERR_CONNECTION_FAILED,
        ERR_UNKNOWN,
    };

    virtual ~LoaderTaskBase();
    virtual void requestCancel();
    virtual bool isFinishded();
    virtual ERROR_CODE getError();

    virtual size_t size() const;
    virtual char* ptr();
};


class LocalFileLoadTask : public LoaderTaskBase
{
public:
    void setPath();
};

class HTTPFileLoadTask : public LoaderTaskBase
{
};


class LoadScheduler
{
private:
public:
    static void initializeSingleton();
    static void finalizeSingleton();
    static LoadScheduler* getInstance();
};

} // namespace ist

#endif // __ist_TaskScheduler__
