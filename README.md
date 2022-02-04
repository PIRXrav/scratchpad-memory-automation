# scratchpad-memory-automation

WIP

## Motivation

The idea is to automate DMA transfers for any program. Memory transfers are now resolved at compile time.  

Architecture with $:

```
                                  $rules  
                          |----------|
[CPU] <===> [$L1] <==> [$L2] <===> [MEMORY]
               |----------|
                    $rules        
```

Architecture with scratchpad:

```
[CPU] <===> [SCRATCHPAD/DMA] <===> [MEMORY]
  |                             ^
  |-----------------------------|
        DMA READ / DMA WRITE
```
  
## How to

We perform a source-to-source conversion.

- Performing this optimization at the binary level is unthinkable to have optimized programs.
- Performing this optimization at the level of the LLVM IR is complicated.
- Doing it on the AST seems like the best option. We have to modify the structure of the program.