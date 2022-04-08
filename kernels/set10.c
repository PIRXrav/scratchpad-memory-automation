//***SMA DEF

#define N 16

//***SMA ARG
 
char output[N];


//***SMA FUN


void set10(){
    for (int n = 0; n < N; n++) {
        output[n] = 10;
        for(int k = 0; k < 10; k++) {
            // output[n] += 1;
        }
    }
}