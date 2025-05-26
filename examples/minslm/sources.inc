TARGETTYPE = PROGRAM

UMTYPE = console
UMENTRY = main

USE_MSVCRT=1
USE_UNICRT=1

USE_STL=1
STL_VER=STL_VER_CURRENT

USE_NATIVE_EH=1

NO_WCHAR_T=1

GUARD=0
INIT_ALL_OPTIN=0

USER_C_FLAGS = \
    $(USER_C_FLAGS) \
    /Zc:forScope \
    /arch:AVX2 \
    /arch:AVX512 \
    /fp:fast \
    /d2jumptablerdata \

!if !defined(MSC_OPTIMIZATION)
MSC_OPTIMIZATION=/O2 /Ob1
!endif

MUI=0
MUI_COMMENT=No localizable resources
