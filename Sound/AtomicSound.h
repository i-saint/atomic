namespace atomic {

    class AtomicSoundThread
    {
    private:
        boost::scoped_ptr<boost::thread> m_thread;
        bool m_stop_request;

    public:
        AtomicSoundThread();
        ~AtomicSoundThread();

        void run();
        void requestStop() { m_stop_request=true; }

    public:
        void operator()();
    };


} //namespace atomic
