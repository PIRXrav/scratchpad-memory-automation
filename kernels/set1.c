//***SMA DEF

#define N 16
#define M 2
//***SMA ARG
 
char output[N][M];


//***SMA FUN


void set1(){
    for (int n = 0; n < N; n++) {
        for(int m = 0; m < M; m++) {
            output[n][m] = 1;
        }
    }
}