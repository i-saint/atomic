cl dll.cpp /Zi /LD /EHsc /MD /DBuildDLL /I../../external /link /libpath:../../external/lib/win32
cl main.cpp dll.lib /Zi /EHsc /MD /I../../external /link /libpath:../../external/lib/win32
