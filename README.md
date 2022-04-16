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


## TODO

- DMA min size
- DMA alignment
- DMA modelisation w/ timings :D
- Static performances analysis (python)
- Dynamic performances analysis (c: dma.h)
- multithread support
- Auto coalescing integration

## On hold

- Auto parallelisation (Typically after CLooG). Custom ? tiramisu ?
- SystemC hardware modelisation/design ?
- Insert dmamapping after CLooG: CLoogG -> DMA_MAPPING -> GCC
- BLAS demo ?
- Large scale integration ? Do not use kernels, but user code.

## References

* CLooG http://www.cloog.org/
* Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code https://arxiv.org/abs/1804.10694
* A Practical Automatic Polyhedral Parallelizer and Locality Optimizer https://asset-pdf.scinapse.io/prod/2034761517/2034761517.pdf
* Bernstein’s Conditions https://hal.inria.fr/hal-01930890/document
* A Tutorial on Abstract Interpretation https://homepage.cs.uiowa.edu/~tinelli/classes/seminar/Cousot--A%20Tutorial%20on%20AI.pdf
* https://en.wikipedia.org/wiki/Interval_arithmetic
* Sympy https://docs.sympy.org/latest/tutorial/intro.html
* Parma Polyhedra Library https://pythonhosted.org/pplpy/index.html#document-index
* Working with functions and control flow graphs https://gcc-python-plugin.readthedocs.io/en/latest/cfg.html
* AST from C code https://stackoverflow.com/questions/239722/ast-from-c-code
* Performing Source-to-Source Transformations with Clang https://llvm.org/devmtg/2013-04/krzikalla-slides.pdf
* Polly https://polly.llvm.org/docs/Architecture.html
* Basic linear algebra subprograms for fortran usage https://ntrs.nasa.gov/api/citations/19780018835/downloads/19780018835.pdf
* Automated Empirical Optimisation of Software and the ALTAS Project http://www.netlib.org/lapack/lawnspdf/lawn147.pdf
