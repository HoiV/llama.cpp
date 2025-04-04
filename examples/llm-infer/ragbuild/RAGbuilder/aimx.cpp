/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    aimx.cpp

Abstract:

    This module implements the main entry point for the AIMX application.
    Handles initialization, WebView2 hosting, and UI setup.

Author:

    Rupo Zhang (rizhang) 03/21/2025

--*/

#include "pch.h"
#include <stdlib.h>
#include <sstream>
#include <wrl.h>
#include <wrl/client.h>
#include <wil/com.h>
#include <urlmon.h>
#include <dwmapi.h>

#include <WebView2.h>
#include <WebView2EnvironmentOptions.h>

#include "LlmService.h"
#include "WebViewMessageHandler.h"
#include "resource.h"

using namespace Microsoft::WRL;

// Global variables
HINSTANCE hInst;
HWND hWnd;

// The location of the WebView2 UDF
const static WCHAR szUserDataFolder[] = L"%ProgramData%\\WebView2Data";

// Pointer to WebViewController
static wil::com_ptr<ICoreWebView2Controller> webviewController;

// Pointer to WebView window
static wil::com_ptr<ICoreWebView2> webviewWindow;

// LLM service and message handler
static LlmService* llmService = nullptr;
static WebViewMessageHandler* messageHandler = nullptr;

// Title bar height for WebView positioning
int titleBarHeight = 0; // Set to 0 to use standard window title bar

// Forward declaration
LRESULT CALLBACK MinimalWndProc(HWND, UINT, WPARAM, LPARAM);

// Set window to dark mode
void SetWindowDarkMode(HWND hwnd, bool darkMode)
{
    BOOL value = darkMode ? TRUE : FALSE;
    DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
    
    // Also set the caption color to a dark gray
    if (darkMode)
    {
        COLORREF captionColor = RGB(32, 32, 32);
        DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, &captionColor, sizeof(captionColor));
    }
}

/*++

Routine Description:

    Initializes the RAG functionality for the application.

Arguments:

    None.

Return Value:

    bool indicating success or failure of the initialization.

--*/
bool InitializeRAG()
{
    // Try to initialize the RAG database (non-critical, can continue if fails)
    try
    {
        if (llmService)
        {
            bool ragInitialized = llmService->InitializeRagDatabase();
            if (ragInitialized)
            {
                LOGINFO("RAG database initialized successfully");
            }
            else
            {
                LOGINFO("RAG database initialization failed, continuing without RAG functionality");
            }
            return ragInitialized;
        }
    }
    catch (const std::exception& e)
    {
        LOGERROR("RAG initialization exception: " + std::string(e.what()));
    }
    
    return false;
}

/*++

Routine Description:

    Converts a relative path to a local URI that can be used with WebView2.

Arguments:

    relativePath - The relative path to convert to a URI.

Return Value:

    std::wstring containing the formatted local URI.

--*/
std::wstring
GetLocalUri(
    _In_ std::wstring relativePath
    )
{
    WCHAR rawPath[MAX_PATH];
    ::GetModuleFileNameW(hInst, rawPath, MAX_PATH);
    std::wstring path(rawPath);
    std::size_t index = path.find_last_of(L"\\") + 1;
    path.replace(index, path.length(), relativePath);

    wil::com_ptr<IUri> uri;
    ::CreateUri(path.c_str(), Uri_CREATE_ALLOW_IMPLICIT_FILE_SCHEME, 0, &uri);

    wil::unique_bstr uriBstr;
    uri->GetAbsoluteUri(&uriBstr);
    return std::wstring(uriBstr.get());
}

/*++

Routine Description:

    Main entry point for the AIMX application. Initializes the application,
    creates the main window, sets up WebView2, and runs the message loop.

Arguments:

    hInstance - Handle to the current instance of the application.
    hPrevInstance - Handle to the previous instance of the application (always NULL in Win32).
    lpCmdLine - Command line arguments.
    nCmdShow - Controls how the window is shown.

Return Value:

    int indicating the exit code of the application.

--*/
int CALLBACK
wWinMain(
    _In_ HINSTANCE hInstance,
    _In_ HINSTANCE hPrevInstance,
    _In_ LPWSTR     lpCmdLine,
    _In_ int       nCmdShow
    )
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    
    // Initialize debug logging   
    LOGINFO("Application starting - log file: " + Debug::GetInstance().GetCurrentLogFileName());
    
    hInst = hInstance;
    
    // Enable DPI awareness for proper scaling
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    
    // Create a window class for hosting WebView2
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.lpfnWndProc = MinimalWndProc;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON)); 
    wc.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON)); 
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = L"WebViewHost";
    wc.hbrBackground = CreateSolidBrush(RGB(32, 32, 32));
    
    if (!RegisterClassExW(&wc))
    {
        LOGERROR("Failed to register window class. Error: " + std::to_string(GetLastError()));
        return 1;
    }
    
    // Get the screen dimensions
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // Create a window with standard decorations but initially not maximized
    int windowWidth = (int)(screenWidth * 0.8);
    int windowHeight = (int)(screenHeight * 0.8);
    int windowX = (screenWidth - windowWidth) / 2;
    int windowY = (screenHeight - windowHeight) / 2;
    
    // Set the window title with a proper name
    const wchar_t* windowTitle = L"AIMX - AI Assistant";
    
    hWnd = CreateWindowW(
        L"WebViewHost",   // Window class name
        windowTitle,      // Window title
        WS_OVERLAPPEDWINDOW, // Window style (includes caption)
        windowX, windowY, windowWidth, windowHeight,
        NULL, NULL, hInstance, NULL
    );
    
    if (!hWnd)
    {
        LOGERROR("Failed to create main window. Error: " + std::to_string(GetLastError()));
        return 1;
    }
    
    // Set the window icon explicitly
    SendMessage(hWnd, WM_SETICON, ICON_BIG, (LPARAM)LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON)));
    SendMessage(hWnd, WM_SETICON, ICON_SMALL, (LPARAM)LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON)));
    
    // Set dark mode for the window
    SetWindowDarkMode(hWnd, true);
    
    // Show the window
    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);
    
    // Initialize LLM Service
    LOGINFO("Initializing LLM Service...");
    llmService = new LlmService();
    
    // Initialize RAG
    InitializeRAG();
    
    // Initialize WebView2
    LOGINFO("Initializing WebView2...");
    WCHAR userDataFolder[MAX_PATH];
    ExpandEnvironmentStringsW(szUserDataFolder, userDataFolder, MAX_PATH);
    
    // Create WebView environment
    LOGINFO("Creating WebView2 environment with user data folder: " + WideToUtf8(userDataFolder));
    auto options = Microsoft::WRL::Make<CoreWebView2EnvironmentOptions>();
    options->put_AdditionalBrowserArguments(L"--disable-web-security --high-dpi-support=1");
    
    CreateCoreWebView2EnvironmentWithOptions(
        nullptr, userDataFolder, options.Get(),
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
                if (FAILED(result)) {
                    LOGERROR("Failed to create WebView2 environment: " + std::to_string(result));
                    return result;
                }
                
                LOGINFO("WebView2 environment created successfully");
                
                // Call CreateCoreWebView2Controller
                env->CreateCoreWebView2Controller(hWnd, 
                    Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                        [](HRESULT result, ICoreWebView2Controller* controller) -> HRESULT {
                            UNREFERENCED_PARAMETER(result);
                            
                            // Store the controller
                            webviewController = controller;
                            webviewController->get_CoreWebView2(&webviewWindow);
                            
                            // Configure settings
                            wil::com_ptr<ICoreWebView2Settings> settings;
                            webviewWindow->get_Settings(&settings);
                            settings->put_IsScriptEnabled(TRUE);
                            settings->put_AreDefaultScriptDialogsEnabled(TRUE);
                            settings->put_IsWebMessageEnabled(TRUE);

                            // Set bounds to fill the client area
                            RECT bounds;
                            GetClientRect(hWnd, &bounds);
                            webviewController->put_Bounds(bounds);
                            
                            // Initialize message handler
                            messageHandler = new WebViewMessageHandler(webviewWindow.get());
                            messageHandler->SetLlmService(llmService);
                            messageHandler->Initialize();
                            
                            // Navigate to HTML content
                            std::wstring contentUrl = GetLocalUri(L".\\htmlui\\html\\mainframe.html");
                            webviewWindow->Navigate(contentUrl.c_str());
                            
                            return S_OK;
                        }).Get());
                
                // Return success
                return S_OK;
            }).Get());
    
    // Message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Cleanup
    delete messageHandler;
    delete llmService;

    LOGINFO("Application exiting normally");

    return (int)msg.wParam;
}

/*++

Routine Description:

    Window procedure for the main application window. Handles window messages
    such as resizing, destruction, and keyboard input.

Arguments:

    hWnd - Handle to the window.
    message - The message.
    wParam - Additional message information.
    lParam - Additional message information.

Return Value:

    LRESULT indicating the result of the message processing.

--*/
LRESULT CALLBACK
MinimalWndProc(
    _In_ HWND hWnd,
    _In_ UINT message,
    _In_ WPARAM wParam,
    _In_ LPARAM lParam
    )
{
    switch (message) {
    case WM_SIZE:
        {
            // Resize WebView to fill client area
            if (webviewController != nullptr) {
                RECT bounds;
                GetClientRect(hWnd, &bounds);
                webviewController->put_Bounds(bounds);
            }
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            // Instead of exiting, minimize the window with Escape
            ShowWindow(hWnd, SW_MINIMIZE);
        }
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
