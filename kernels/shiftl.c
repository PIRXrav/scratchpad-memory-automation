//***SMA DEF

#define N 16

//***SMA ARG
 
char input[N];
char output[N];

    
//***SMA FUN


void test(){
    for (int n = 0; n < N; n++) {
        for(int k = 0; k < N; k++){
            output[n/2 + k/2] += input[n];
        }   
    }
}