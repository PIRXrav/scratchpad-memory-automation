/* log.c */

extern void log_lock(void) { /* mutex lock */ }
extern void log_unlock(void) { /* mutex unlock */ }

const char *level_strings[] = {"TRACE", "DEBUG", "INFO",
                               "WARN",  "ERROR", "FATAL"};

const char *level_colors[] = {"\x1b[94m", "\x1b[36m", "\x1b[32m",
                              "\x1b[33m", "\x1b[31m", "\x1b[35m"};