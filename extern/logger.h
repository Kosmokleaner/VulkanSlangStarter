/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Logger class for handling logging with different log levels.
 * 
 * Usage:
 * 
 * // Get the logger instance
 * Logger& logger = Logger::getInstance();
 * 
 * // Set the log level
 * logger.setLogLevel(Logger::LogLevel::eINFO);
 * 
 * // Set the information to show in the log
 * logger.setShowFlags(Logger::eSHOW_TIME | Logger::eSHOW_LEVEL);
 * 
 * // Set the output file : default is the name of the executable with .txt extension
 * logger.setOutputFile("logfile.txt");
 * 
 * // Enable or disable file output
 * logger.enableFileOutput(true);
 * 
 * // Set a custom log callback
 * logger.setLogCallback([](Logger::LogLevel level, const std::string& message) {
 *     std::cout << "Custom Log: " << message << std::endl;
 * });
 * 
 * // Log messages
 * LOGD("This is a debug message.");
 * LOGI("This is an info message.");
 * LOGW("This is a warning message.");
 * LOGE("This is an error message with id: %d.", integerValue);
 */


#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <cstdarg>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifdef APIENTRY
#undef APIENTRY
#endif
#define NOMINMAX  // Prevent windows.h from defining min and max macros
#include <windows.h>
#include <debugapi.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace utils {

class Logger
{
public:
  enum class LogLevel
  {
    eDEBUG = 0,
    eINFO,
    eWARNING,
    eERROR
  };

  enum ShowFlags
  {
    eSHOW_NONE  = 0,
    eSHOW_TIME  = 1 << 0,
    eSHOW_LEVEL = 1 << 1
  };

  using LogCallback = std::function<void(LogLevel, const std::string&)>;

  // Get the logger instance
  static Logger& getInstance()
  {
    static Logger instance;
    return instance;
  }

  // Set the minimum log level
  void setLogLevel(LogLevel level) { m_minLogLevel = level; }

  // Set the information to show in the log
  void setShowFlags(int flags) { m_show = flags; }

  // Set the output file
  void setOutputFile(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock(m_logMutex);
    if(m_logFile.is_open())
    {
      m_logFile.close();
    }
    m_logFile.open(filename, std::ios::out);
    m_logToFile = m_logFile.is_open();
  }

  // Enable or disable file output
  void enableFileOutput(bool enable) { m_logToFile = enable; }

  // Set a custom log callback
  void setLogCallback(LogCallback callback)
  {
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logCallback = callback;
  }

  // Log a message
  void log(LogLevel level, const char* format, ...)
  {
    if(level < m_minLogLevel)
      return;

    // Open the log file if it is enabled and not already open
    static bool firstLog = true;
    if(firstLog && m_logToFile && !m_logFile.is_open())
    {
      firstLog                   = false;
      std::string defaultLogFile = getExecutableName() + ".txt";
      setOutputFile(defaultLogFile);
    }

    // Format the message
    va_list args;
    va_start(args, format);
    std::string message = formatString(format, args);
    va_end(args);

    // Add time and level to the message
    std::ostringstream logStream;
    if(m_show & eSHOW_TIME)
      logStream << "[" << currentDateTime() << "] ";
    if(m_show & eSHOW_LEVEL)
      logStream << logLevelToString(level) << ": ";
    logStream << message;

    // Log to console (writes to std::cout/std::cerr -- a no-op in practice unless a real console
    // is actually attached, see AttachParentConsole()/Platform.cpp, so safe to always call)
    outputToConsole(level, logStream.str());

    // Log to file
    if(m_logToFile && m_logFile.is_open())
    {
      std::lock_guard<std::mutex> lock(m_logMutex);
      m_logFile << logStream.str() << std::endl;
    }

    // Log to callback
    if(m_logCallback)
    {
      m_logCallback(level, logStream.str());
    }

    // Log to OutputDebugString on Windows
#ifdef _WIN32
    OutputDebugStringA((logStream.str() + "\n").c_str());
#endif
  }

private:
  LogLevel      m_minLogLevel = LogLevel::eWARNING;  // Start at warning level
  std::ofstream m_logFile;                           // Output file stream
  bool          m_logToFile = true;                  // Enable file output
  std::mutex    m_logMutex;                          // Mutex to protect the log file and callback
  LogCallback   m_logCallback = nullptr;             // Custom log callback
  int           m_show        = eSHOW_NONE;          // Default shows no extra information

  Logger() {}

  ~Logger()
  {
    if(m_logFile.is_open())
    {
      m_logFile.close();
    }
  }

  Logger(const Logger&)            = delete;
  Logger& operator=(const Logger&) = delete;

  static std::string formatString(const char* format, va_list args)
  {
    // Initial buffer size
    int               bufferSize = 1024;
    std::vector<char> buffer(bufferSize);

    // Try to format the string into the buffer
    va_list argsCopy;
    va_copy(argsCopy, args);  // Copy args to reuse them for vsnprintf
    int requiredSize = vsnprintf(buffer.data(), bufferSize, format, argsCopy);
    va_end(argsCopy);

    // Check if the buffer was large enough
    if(requiredSize >= bufferSize)
    {
      bufferSize = requiredSize + 1;  // Increase buffer size as needed
      buffer.resize(bufferSize);
      vsnprintf(buffer.data(), bufferSize, format, args);  // Format again with correct size
    }

    return std::string(buffer.data());
  }

  static std::string logLevelToString(LogLevel level)
  {
    switch(level)
    {
      case LogLevel::eDEBUG:
        return "DEBUG";
      case LogLevel::eINFO:
        return "INFO";
      case LogLevel::eWARNING:
        return "WARNING";
      case LogLevel::eERROR:
        return "ERROR";
      default:
        return "";
    }
  }

  static void outputToConsole(LogLevel level, const std::string& message)
  {
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    if(level == LogLevel::eERROR)
    {
      SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
      std::cerr << message << std::endl;
    }
    else if(level == LogLevel::eWARNING)
    {
      SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
      std::cout << message << std::endl;
    }
    else
    {
      SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
      std::cout << message << std::endl;
    }
    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
#else
    if(level == LogLevel::eERROR)
    {
      std::cerr << "\033[1;31m" << message << "\033[0m" << std::endl;
    }
    else if(level == LogLevel::eWARNING)
    {
      std::cout << "\033[1;33m" << message << "\033[0m" << std::endl;
    }
    else
    {
      std::cout << message << std::endl;
    }
#endif
  }

  static std::string getExecutableName()
  {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string fullPath(buffer);
#else
    char        buffer[PATH_MAX];
    ssize_t     count = readlink("/proc/self/exe", buffer, PATH_MAX);
    std::string fullPath(buffer, (count > 0) ? count : 0);
#endif
    return std::filesystem::path(fullPath).stem().string();
  }

  static std::string currentDateTime(bool includeDate = false)
  {
    auto now       = std::chrono::system_clock::now();
    auto ms        = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm bt{};
#ifdef _WIN32
    localtime_s(&bt, &in_time_t);
#else
    localtime_r(&in_time_t, &bt);
#endif

    char buf[64];
    if(includeDate)
    {
      std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &bt);
    }
    else
    {
      std::strftime(buf, sizeof(buf), "%H:%M:%S", &bt);
    }

    // Append milliseconds
    static thread_local std::ostringstream oss;
    oss.str("");
    oss.clear();
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
  }
};

}  // namespace utils

// Logging macros
#define LOGD(format, ...) utils::Logger::getInstance().log(utils::Logger::LogLevel::eDEBUG, format, ##__VA_ARGS__)
#define LOGI(format, ...) utils::Logger::getInstance().log(utils::Logger::LogLevel::eINFO, format, ##__VA_ARGS__)
#define LOGW(format, ...) utils::Logger::getInstance().log(utils::Logger::LogLevel::eWARNING, format, ##__VA_ARGS__)
#define LOGE(format, ...) utils::Logger::getInstance().log(utils::Logger::LogLevel::eERROR, format, ##__VA_ARGS__)


#endif  // LOGGER_HPP
