//***SMA DEF

#define N 16

//***SMA ARG
 
char input[N];
char output[N];


//***SMA FUN


void copy(){
    for (int n = 0; n < N; n++) {
        output[n +1 -1] = input[n + 1 -1]; // /!\ Overflow
    }
}