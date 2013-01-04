#ifndef ist_Base_NonCopyable_h
#define ist_Base_NonCopyable_h

#define istNonCopyable(ClassName)\
    private:\
        ClassName(const ClassName &other);\
        ClassName& operator=(const ClassName &other);

#endif // ist_Base_NonCopyable_h
