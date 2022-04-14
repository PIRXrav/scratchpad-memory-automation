//***SMA DEF

#define N 16

//***SMA ARG

// test
int32_t tab32b[N];
int16_t tab16b[N];
int8_t  tab08b[N];

//***SMA FUN


void fun(){
    for (int n = 0; n < N; n++) {
        tab32b[n] = 0xcafecafe;
        tab16b[n] = 0xc0de;
        tab08b[n] = 0x42;
    }
}