/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    pch.h

Abstract:

    This module defines the precompiled header for the AIMX application.
    It includes common headers and suppresses specific build warnings coming
    from hnswlib library.

Author:

    Rupo Zhang (rizhang) 03/22/2025

--*/

#pragma once

// NOMINMAX to prevent Windows.h from defining min/max macros, as they are conflicting
// with hnswlib's internal reference of STL.
#ifndef NOMINMAX
#define NOMINMAX
#endif

// Suppress specific warnings from hnswlib library
#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4127) // cond expression is constant
#pragma warning (disable:4242) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4244) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

// Common system headers
#include <windows.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <queue>
#include <mutex>
#include <thread>

// HNSWLIB headers
#include "hnswlib/hnswlib.h"

// JSON library
#include "nlohmann/json.hpp"

// AIMX project headers
#include "Debug.h"
#include "StringUtils.h"
