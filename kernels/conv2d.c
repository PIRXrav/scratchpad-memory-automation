//***SMA DEF

#define Y 16
#define X 16
#define DKX 3
#define DKY 3

//***SMA ARG


char input[Y][X];
char output[Y-DKY+1][X-DKX+1];
char weights[DKY][DKX];


//***SMA FUN

void conv2d(){
    for (int y = 0; y < (Y-DKY+1); y++) {
        // output[n] = 0; TODO
        for (int x = 0; x < (X-DKX+1); x++) {
            for (int dky = 0; dky < DKY; dky++){
                for (int dkx = 0; dkx < DKX; dkx++){
                    output[y][x] += input[y+dky][x+dkx] * weights[dky][dkx];
                }
            }
        }
    }
}