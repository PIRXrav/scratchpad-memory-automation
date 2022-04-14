"""Let's parallelize C codes

Some ressources:

Loop parallelization algorithms: from parallelism extraction to code generation
Pierre Boulet, Alain Darte, Georges-André Silber, Frédéric Vivien
June 1997

A Practical Automatic Polyhedral Parallelizer and Locality Optimizer
Uday Bondhugula, Albert Hartono, J. Ramanujam, P. Sadayappan
2008
https://asset-pdf.scinapse.io/prod/2034761517/2034761517.pdf

Bernstein’s Conditions
Paul Feautrier
November 21, 2018
https://hal.inria.fr/hal-01930890/document

Bersntein’s conditions are a simple test for deciding if statements or
operations can be interchanged without modifying the program results. The test
applies to operations which read and write memory at well defined addresses.
If u is an operation, let M(u) be the set of (adresses of) the memory cells
it modifies, and R(u) the set of cells it reads. Operations u and v can be
reordered if:

            M(u) ∩ M(v) = M(u) ∩ R(v) = R(u) ∩ M(v) = ∅

If these conditions are met, one says that u and v commute or are independent.

Our idea is to find a subdivision of the iteration space in which all write
access addresses are unique to the sub-iteration space
"""

NB_CORE = 16

def parallelize(loops_access_names, loops_access_l, ref_access_names):
    """
    loops_access_names=['m', 'n', '__SENTINEL__']
    loops_access_l=[32, 32, 1]
    ref_access_names=['m', 'n']
    """
    index = 0
    block_size_f = loops_access_l[index] / NB_CORE
    block_size = int(block_size_f)
    # Strong requirement
    assert block_size == block_size_f
    raise NotImplementedError("TODO")
