#ifndef LOG_H
#define LOG_H

#include <stdarg.h>
#include <stdio.h>

#define LOG_TRACE  0
#define LOG_DEBUG  1
#define LOG_INFO   2
#define LOG_WARN   3
#define LOG_ERROR  4
#define LOG_FATAL  5
#define LOG_SILENT 6

#define LOG_USE_EM 1
#define LOG_LEVEL  LOG_WARN

extern void log_lock(void);
extern void log_unlock(void);

extern const char *level_strings[];
extern const char *level_colors[];

#define _log_log(level, ...)                                               \
    log_lock();                                                            \
    printf("%s%-5s\x1b[0m \x1b[90m%s:%.4d\x1b[0m > ", level_colors[level], \
           level_strings[level], __FILE__, __LINE__);                      \
    printf(__VA_ARGS__);                                                   \
    printf("\n");                                                          \
    log_unlock();

#if LOG_LEVEL <= LOG_TRACE
#define log_trace(...) _log_log(LOG_TRACE, __VA_ARGS__)
#else
#define log_trace(...) /* nop */
#endif

#if LOG_LEVEL <= LOG_DEBUG
#define log_debug(...) _log_log(LOG_DEBUG, __VA_ARGS__)
#else
#define log_debug(...) /* nop */
#endif

#if LOG_LEVEL <= LOG_INFO
#define log_info(...) _log_log(LOG_INFO, __VA_ARGS__)
#else
#define log_info(...) /* nop */
#endif

#if LOG_LEVEL <= LOG_WARN
#define log_warn(...) _log_log(LOG_WARN, __VA_ARGS__)
#else
#define log_warn(...) /* nop */
#endif

#if LOG_LEVEL <= LOG_ERROR
#define log_error(...) _log_log(LOG_ERROR, __VA_ARGS__)
#else
#define log_error(...) /* nop */
#endif

#if LOG_LEVEL <= LOG_FATAL
#define log_fatal(...) _log_log(LOG_FATAL, __VA_ARGS__)
#else
#define log_fatal(...) /* nop */
#endif

#endif /* _log_H_ */
