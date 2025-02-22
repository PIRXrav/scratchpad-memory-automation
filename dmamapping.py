"""
SCRATCHPAD MEMORY AUTOMATION


C src
[pycparser] ---> pycparse AST
[AstToolkit] ---> Simplified AST (ast with only for/refs)
[__SMA__] ----> pycparse AST with memory transaction
[EXPORT] ----> C src

"""

import sys
from pycparser import c_ast
from collections import defaultdict
from itertools import chain
import numpy as np
import asttools as at
from asttools import stmt_c_to_ast, expr_c_to_ast
from math import floor
import logging
import sympy
from copy import copy
import polyhedron as poly
from gencode_dma import Gencode
# from math import gcd


log = logging.getLogger(__name__)

TYPES_SIZE = {'char': 1,
              'int': 4,
              'float': 4,
              'double': 8}

WORD_SIZE = 'DeleteMeFlake8ok'


def do_memory_mapping(ast, ref_decl_namespace, dma_size):
    log.debug(f"{ref_decl_namespace=}")
    for topfor in at.c_ast_get_all_topfor(ast):
        do_memory_mapping_on_topfor(ast, topfor, ref_decl_namespace, dma_size)


def do_memory_mapping_on_topfor(ast, topfor, ref_decl_namespace, dma_size):
    log.debug("TOP FORS:")

    # Comute refs
    refs = at.c_ast_get_all_top_ref(topfor)

    # Find duplicate refs
    merged_refs = defaultdict(list)
    for i, ref in enumerate((refs)):
        reg_name, ast_l = at.c_ast_ref_get_l(ref)
        merged_refs[reg_name].append(ref)

    # TODO: polyhedron
    # For now, all refl must be equal
    for ref_name, refs in merged_refs.items():
        log.debug(f"{reg_name}")
        for ref in refs:
            log.debug(f"{at.ast_to_c(ref):20} RW={at.c_ast_ref_is_rw(ast, ref)}")
            if at.ast_to_c(ref) != at.ast_to_c(refs[0]):
                raise Exception(
                    f"Unsupported reference: {at.ast_to_c(ref)} != {at.ast_to_c(refs[0])}"
                )

    nb_refs = len(merged_refs)
    if nb_refs > 3:
        raise Exception(f"Unsupported number of refs: {nb_refs}")

    log.debug(f"TOP REFS ({nb_refs}):")
    for i, (ref_name, refs) in enumerate(merged_refs.items()):
        log.debug(f"{at.ast_to_c(ref):20} RW={at.c_ast_ref_is_rw(ast, ref)}")
        dma_mapping_algo3(ast, refs, i, ref_decl_namespace, dma_size)


def c_ast_to_interval(ref, namespace):
    """
    Analyse the reference
    """
    class RefVisitor(c_ast.NodeVisitor):
        """
        ArrayRef(name=ArrayRef(name=ID(name='tab0'
                                    ),
                            subscript=ID(name='j'
                                            )
                            ),
                subscript=ID(name='i'
                            )
        )
        return ['tab0', 'j', 'i']
        """

        def __init__(self):
            self.stack = []
            self.depsid = set()

        def generic_visit(self, node):
            raise Exception("Unsupported OP: ", node)

        def visit_ID(self, node):
            if node.name in namespace:
                interval = copy(namespace[node.name])
                self.depsid.add(node.name)
            else:
                raise Exception(f"Unknown ID: {node} in {namespace}")
            # print(f'Visit ID {node.name}: push({interval})')
            self.stack.append(interval)

        def visit_BinaryOp(self, node):
            for n in node:
                self.visit(n)
            b = self.stack.pop()
            a = self.stack.pop()
            if node.op == "+":
                interval = a + b
            elif node.op == "-":
                interval = a - b
            elif node.op == "/":
                interval = a / b
            elif node.op == "*":
                interval = a * b
            else:
                raise Exception(f"Invalid OP {node.op} in {node}")
            # print(f'visit_BinaryOp : push({interval})')
            self.stack.append(interval)

        def visit_Constant(self, node):
            if node.type == "int":
                v = int(node.value)
                interval = poly.Interval(v, v)
            else:
                raise Exception(f"Invalid type {node.type} in {node}")
            # print(f'visit_Constant : push({interval})')
            self.stack.append(interval)

    rv = RefVisitor()
    rv.visit(ref)
    interval = rv.stack.pop()
    if rv.stack != []:
        raise Exception(f"Error during visit: {rv.stack=}")
    return interval, rv.depsid


def c_ast_ref_to_interval(ref, namespace):
    name, asts = at.c_ast_ref_get_l(ref)
    poly_ref = []
    for ast in asts:
        interval, deps = c_ast_to_interval(ast, namespace)
        # print(f"!!! INT({at.ast_to_c(ast)}) -> {interval} @ {namespace}")
        poly_ref.append((interval, deps))

    # print("!!! ", name, poly_ref)
    return name, asts, poly_ref


def get_size_of_type(ctype):
    """
    int32_t -> 4
    float32 -> 4
    int -> 4
    """
    if ctype in TYPES_SIZE:
        return TYPES_SIZE[ctype]

    for nb_byes in range(1, 64):  # int8_t to int512_t
        size = 8 * nb_byes
        if str(size) in ctype:
            return nb_byes
    raise Exception(f"Unknown type size: {ctype}")


def c_ast_expr_to_l(ast):
    return sympy.parsing.sympy_parser.parse_expr(at.ast_to_c(ast), evaluate=True)


def c_ast_arraydecl_to_l(decl):
    name, ast_type, asts = at.c_ast_arraydecl_get_all(decl)
    poly_l = [c_ast_expr_to_l(ast) for ast in reversed(asts)]
    assert len(ast_type.type.names) == 1
    poly_l.insert(0, get_size_of_type(ast_type.type.names[-1]))
    return name, ast_type, asts, poly_l


def c_ast_loop_to_interval_name(for_nodes):
    # Comute L ast for all for nodes
    poly_loop = []
    for name, ast_a, ast_b in map(at.c_ast_for_get_l, for_nodes):
        a = c_ast_to_interval(ast_a, {})[0].b
        b = c_ast_to_interval(ast_b, {})[0].b
        poly_loop.append((poly.Interval(a, b - 1), name))

    return poly_loop


def contain_ordered(listin, data):
    if listin == []:
        return True
    if data == []:
        return False
    if listin[0] == data[0]:
        return contain_ordered(listin[1:], data[1:])
    else:
        return contain_ordered(listin, data[1:])


def dma_mapping_algo3(ast, refs, iref, ref_decl_namespace, dma_size):
    """ """

    ref = refs[0]

    log.debug(f"========== DMA MAPPING {at.ast_to_c(ref)}")

    # Analyse Loops
    for_nodes = at.c_ast_get_for_fathers(ast, ref)
    poly_loop = c_ast_loop_to_interval_name(for_nodes)
    poly_loop_namespace = {name: interval for interval, name in poly_loop}
    loops_access_l = [*[interval.area() for interval, _ in poly_loop], 1]
    loops_access_names = [*[name for _, name in poly_loop], "__SENTINEL__"]

    # Analyse reference
    ref_is_read, ref_is_write = at.c_ast_ref_is_rw(ast, ref)
    ref_name, ref_l_ast, poly_ref = c_ast_ref_to_interval(
        ref, namespace=poly_loop_namespace
    )
    poly_ref_deps = [names for _, names in poly_ref]
    poly_ref_all_deps = set(chain(*poly_ref_deps))
    ref_access_names = [at.ast_to_c(ast) for ast in ref_l_ast]

    # Analyse reference declaration
    ref_decl_type_ast, ref_decl_l = ref_decl_namespace[ref_name]

    def decl_l_fit_word_size(decl_l):
        decl_l_cum = list(np.cumprod(ref_decl_l))
        print(f'{decl_l_cum=}')

        if 0:
            # Add memory paddind
            # Find where insert memory padding
            ID = 1
            if dma_size >= decl_l_cum[-1]:
                ID = -1
            else:
                while ID + 1 < len(decl_l_cum) and dma_size > decl_l_cum[ID + 1]:
                    ID += 1

            print(f'{ID=} -> {decl_l_cum[ID]=}')

            if decl_l_cum[ID] % WORD_SIZE == 0:
                return decl_l

            # if ID == -1:  # Last col
            #     log.warning("READ outside of allocated memory")
            #     return decl_l

            if ID == 1:  # First col
                decl_l[ID] += (WORD_SIZE - (decl_l[ID] % WORD_SIZE))
                return decl_l

            cur = decl_l_cum[ID] + (WORD_SIZE - (decl_l_cum[ID] % WORD_SIZE))
            print(f"Want {decl_l_cum[ID]=} -> {cur=}")

            print(f"range = [{decl_l_cum[ID - 1]}  :  {cur // decl_l[ID] + 1}]")
            for new_l in range(decl_l[ID - 1], cur // decl_l[ID] + 1 + 1):
                new_lp = cur / new_l
                if new_lp < decl_l[ID]:
                    continue
                if new_lp == int(new_lp):
                    new_lp = int(new_lp)
                    print((f"!HIT {new_lp} x {new_l} = {cur}"))
                    decl_l[ID] = new_lp
                    decl_l[ID - 1] = new_l
                    return decl_l
        elif 0:
            ID = 1
            pading = (((WORD_SIZE - ((decl_l[ID] * decl_l[ID - 1]) % WORD_SIZE))) % WORD_SIZE)
            pading = int(pading)
            log.debug(f'{decl_l[ID]=} {pading=} => {(decl_l[ID]+pading)/decl_l[ID]=}')
            decl_l[ID] += pading
            return decl_l
        else:
            return decl_l

        log.error(f"Indisisible memory declaration {decl_l}; We must add a lot of padding !")
        raise Exception(f"Invalid memory declaration: {decl_l}")

    ref_decl_l = decl_l_fit_word_size(ref_decl_l)
    ref_decl_l_cum = list(np.cumprod(ref_decl_l))

    # Update namespace
    ref_decl_namespace[ref_name] = (ref_decl_type_ast, ref_decl_l)

    # if 'input' in ref_name:
    #     print("-- no input")
    #     return
    # if 'weights' in ref_name:
    #     print("-- no weights")
    #     return
    # if 'out' in ref_name:
    #     print("-- no out")
    #     return

    def compute_area(ref, namespace):

        # Compute memory footprint for this namespace
        _, _, pr = c_ast_ref_to_interval(ref, namespace=namespace)
        # Compute poly area
        area = ref_decl_l[0]  # Type size
        for ir, (interval, deps) in enumerate(reversed(pr)):
            v = interval.area()
            if v == 0:  # Not a dim in the array
                continue
            if v == 1:  # Cst dim
                continue
            area *= v
            ir = len(ref_decl_l_cum) - ir - 1 - 1
            if ir != 0:
                area *= ref_decl_l_cum[ir]
            break

        # eq = f"{v}" + (f"*{ref_decl_l_cum[ir]}" if ir != 0 else "")
        # log.debug(f"AREA = {area} = {eq}  @ {pr} @ ns={namespace}")

        # Fix area to word size
        # area += (WORD_SIZE - area % WORD_SIZE) % WORD_SIZE
        return area

    # Compute poly ref area
    from copy import deepcopy

    poly_loop_namespace_sparse = deepcopy(poly_loop_namespace)
    loops_ref_access_l_cum = []
    for name in reversed(loops_access_names):
        # Update poly namespace
        poly_loop_namespace_sparse[name] = poly.Interval(0, 0)
        # Compute poly area
        area = compute_area(ref, poly_loop_namespace_sparse)
        # Append area
        loops_ref_access_l_cum.insert(0, area)

    loops_ref_access_l_ref_expr_dma = [
        [
            sympy.parsing.sympy_parser.parse_expr(s, evaluate=False)
            for s in (ref_access_names)
        ]
    ]
    for name in reversed(loops_access_names):
        # TODO use sympy
        cur_exprs = deepcopy(loops_ref_access_l_ref_expr_dma[-1])
        for i, _ in enumerate(cur_exprs):
            cur_exprs[i] = cur_exprs[i].subs(name, 0)
        loops_ref_access_l_ref_expr_dma.append(cur_exprs)
    loops_ref_access_l_ref_expr_dma = list(reversed(loops_ref_access_l_ref_expr_dma))

    loops_ref_access_l_ref_expr_ram = [
        [
            sympy.parsing.sympy_parser.parse_expr(s, evaluate=False)
            for s in (ref_access_names)
        ]
    ]
    for name in loops_access_names:
        cur_exprs = deepcopy(loops_ref_access_l_ref_expr_ram[-1])
        for i, _ in enumerate(cur_exprs):
            cur_exprs[i] = cur_exprs[i].subs(name, 0)
        loops_ref_access_l_ref_expr_ram.append(cur_exprs)

    log.debug(f"{ref_name=}")
    log.debug(f"{ref_is_read=}")
    log.debug(f"{ref_is_write=}")
    log.debug("GLOBAL NS")
    log.debug(f"  |->{ref_decl_l=}")
    log.debug(f"  '->{ref_decl_l_cum=}")
    log.debug(f"{poly_loop=}")
    log.debug(f"  |->{loops_access_names=}")
    log.debug(f"  |->{loops_access_l=}")
    log.debug(f"  `->{poly_loop_namespace=}")
    log.debug(f"{poly_ref=}")
    log.debug(f"  |->{ref_access_names=}")
    log.debug(f"  |->{poly_ref_deps=}")
    log.debug(f"  |->{poly_ref_all_deps=}")
    log.debug("LOOP X REF")
    log.debug(f"  `->{loops_ref_access_l_cum=}")
    log.debug(f"  `->{loops_ref_access_l_ref_expr_dma=}")
    log.debug(f"  `->{loops_ref_access_l_ref_expr_ram=}")

    # Check if area computation is ok
    assert loops_ref_access_l_cum[-1] == ref_decl_l_cum[-1]

    # Find where insert DMA LD/ST
    IL = 0
    if dma_size >= loops_ref_access_l_cum[-1]:  # We only need to do 1 transfert
        IL = -1
    else:
        while dma_size > loops_ref_access_l_cum[IL]:
            IL += 1

    # Find how to insert DMA LD/ST
    IR = 0
    if dma_size >= ref_decl_l_cum[-1]:  # We only need to do 1 transfert
        IR = -1
    else:
        while dma_size > ref_decl_l_cum[IR]:
            IR += 1

    log.debug(f"{IL=}")
    log.debug(f"{IR=}")

    log.debug(f"  `->{loops_ref_access_l_cum[IL]=}")
    log.debug(f"  `->{ref_access_names=}")

    # TODO: Lecture ne correspondant pas aux index direct exemple tab[i+1] (wip)
    # TODO: Partitionnement mémoire. Exemple Toeplitz matrix.
    # https://www.rle.mit.edu/eems/wp-content/uploads/2019/06/Tutorial-on-DNN-04-Kernel-Computations.pdf
    # Slide 25

    def namespace_reduction(poly_loop_namespace, names, div_name, div_blocks):
        # Compute namespace
        namespace = deepcopy(poly_loop_namespace)
        for name in reversed(names):
            if name == div_name:
                namespace[name] = poly.Interval(0, div_blocks - 1)
                return namespace
            else:
                namespace[name] = poly.Interval(0, 0)
        raise Exception(f"{div_name} not in {names}")

    # IL configuration
    il_repeat = loops_access_l[IL - 1]
    il_name = loops_access_names[IL - 1]
    il_subsize = loops_ref_access_l_cum[IL - 1]

    log.debug(f"Repeat {il_repeat} times the loop '{il_name}' #{il_subsize}")

    # Find the best division
    # TODO: Algo: binary search for perfs
    # Compute new loopp poly
    for block_size in reversed(range(1, il_repeat + 1)):
        poly_loop_namespace_partionned = namespace_reduction(
            poly_loop_namespace, loops_access_names, il_name, block_size
        )
        area = compute_area(ref, poly_loop_namespace_partionned)
        # log.debug(f"{block_size=} @ ns = {poly_loop_namespace_partionned} -> {area=}")
        # Is area valid AND is multiplicity ok ? and nb_repeat_residual == 0
        if area <= dma_size and il_repeat % block_size == 0:
            break

    log.debug(f"HIT : {block_size=} @ ns = {poly_loop_namespace_partionned} -> {area=}")
    assert area <= dma_size

    # Number of repeat for this configuration
    nb_repeat_int = block_size
    nb_repeat_residual = il_repeat % block_size
    dma_transfer_size_eff = nb_repeat_int * il_subsize
    dma_transfer_size = area

    nb_repeat_block = floor(il_repeat / block_size)

    if nb_repeat_residual:
        namespace_residual = namespace_reduction(
            poly_loop_namespace, loops_access_names, il_name, nb_repeat_residual
        )
        dma_transfer_size_residual = compute_area(ref, namespace_residual)
    else:
        dma_transfer_size_residual = 1010101010

    log.debug(
        f" ===> DMA OPS: {nb_repeat_int}->{dma_transfer_size}"
        f" & {nb_repeat_residual}->{dma_transfer_size_residual}"
    )
    log.debug(
        f" ===> DMA OPS: {nb_repeat_block} x {dma_transfer_size}"
        f" + {nb_repeat_residual>0} x {dma_transfer_size_residual}"
    )

    stats_dma_ops = nb_repeat_block * dma_transfer_size
    stats_dma_ops += (nb_repeat_residual > 0) * dma_transfer_size_residual
    log.debug(f"Nuber of bytes loaded/stored = {stats_dma_ops}")

    # DMA configuration
    adr_name = f"__SMA__{il_name}_adr{iref}"
    size_name = f"__SMA__{il_name}_size{iref}"
    cgen_dma_args = (str(iref), adr_name, size_name)

    # The higher Compound node
    super_top_comp = at.c_ast_get_upper_node(ast, for_nodes[-1])
    # Get Compound @ IL
    if IL == -1:
        topcomp = super_top_comp
    else:
        topcomp = for_nodes[IL - 1].stmt

    if not topcomp:
        raise Exception(f"No {topcomp=} @ {IL=}")

    # Compute
    if IR == -1:  # Array < DMA
        log.debug("--- WRAP MODE")
        # Compute memory mapping
        dma_transfer_size = loops_ref_access_l_cum[-1]
        dma_transfer_size_eff = dma_transfer_size
        mapping_ram_name = ref_name
        # Init DMA
        topcomp.block_items.insert(
            0, at.stmt_c_to_ast(Gencode.cgen_dma_init(iref, mapping_ram_name, dma_transfer_size))
        )
        if ref_is_read:  # insert LD
            topcomp.block_items.insert(
                1, expr_c_to_ast(Gencode.cgen_dma_ld(*cgen_dma_args))
            )
        if ref_is_write:  # Insert ST
            topcomp.block_items.append(
                expr_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args))
            )
        # Update ref
        buff_adr = Gencode.cgen_static_mac(ref_access_names, ref_decl_l_cum[0:-1])
        adr_ast = expr_c_to_ast(f"DMA_RW({iref}, {buff_adr})")
        to_type = c_ast.Typename(None, [], None, c_ast.PtrDecl([], ref_decl_type_ast))
        new_node = c_ast.Cast(to_type, adr_ast)  # Cast
        new_node = c_ast.UnaryOp('*', new_node)  # dereference
        at.c_ast_replace_nodes(ast, refs, [deepcopy(new_node) for _ in refs])

    else:
        # Loop names
        il_high_name = il_name + '_high'
        il_low_name = il_name + '_low'
        il_high_l = nb_repeat_block + (1 if nb_repeat_residual else 0)
        il_fornode = for_nodes[IL - 1]

        # replace ref@ -> dma@
        new_ref_expr_dma = [
            e.subs(il_name, il_low_name) for e in loops_ref_access_l_ref_expr_dma[IL]
        ]
        buff_adr = Gencode.cgen_static_mac(new_ref_expr_dma, ref_decl_l_cum)
        adr_ast = expr_c_to_ast(f"{buff_adr}")
        adr_ast = expr_c_to_ast(f"DMA_RW({iref}, {buff_adr})")
        to_type = c_ast.Typename(None, [], None, c_ast.PtrDecl([], ref_decl_type_ast))
        new_node = c_ast.Cast(to_type, adr_ast)  # Cast
        new_node = c_ast.UnaryOp('*', new_node)  # dereference
        at.c_ast_replace_nodes(ast, refs, [deepcopy(new_node) for _ in refs])

        # Compute base ref@
        rn = loops_ref_access_l_ref_expr_ram[IL - 1]
        new_ref_expr_ram = [
            e.subs(il_name, f"{il_high_name} * {block_size}") for e in rn
        ]
        mapping_ram_name = ref_name + "".join(reversed(list((f"[{i}]" for i in new_ref_expr_ram))))
        log.debug(f"  --> {mapping_ram_name=}")

        # Code generation
        #
        # for (int i_high = 0; i_high < 4; i_high++){
        #     DMA_LOAD();
        #     for (i_low = 0; i_low < C/4; i_low++){
        #         i = i_high * C/4 + i_low;
        #         if(i < 64){
        #             for int(j = 0; j < C; j++){
        #                 tab [i][j] += i + j;
        #             }
        #         }
        #     }
        #     DMA_STORE();
        # }

        # Update high loop variable & l
        at.c_ast_for_update_name(il_fornode, il_high_name)
        at.c_ast_for_update_l(il_fornode, at.expr_c_to_ast(str(il_high_l)))

        stmts = []

        # Set variables
        if nb_repeat_residual:
            size = f"""{il_high_name} != {il_high_l-1}
                    ? {dma_transfer_size}
                    : {dma_transfer_size_residual}"""
        else:
            size = str(dma_transfer_size)

        # for stmt in at.stmt_c_to_ast(f"""{{{size_name} = {size};
        #                                 {adr_name} = {'&' + mapping_ram_name};
        #                                 }}""").block_items:
        # Initialise DMA
        stmts.append(at.stmt_c_to_ast(Gencode.cgen_dma_init(iref, '&' + mapping_ram_name, size)))

        # Load
        if ref_is_read:
            stmts.append(
                stmt_c_to_ast(f"{Gencode.cgen_dma_ld(*cgen_dma_args)};")
            )

        if nb_repeat_residual:
            inner_for = stmt_c_to_ast(f"if({il_name} < {il_repeat}){{}}")
            inner_for.iftrue.block_items = topcomp.block_items  # Old statements
            inner_for = [inner_for]
        else:
            inner_for = topcomp.block_items

        # Subfor
        for_low = stmt_c_to_ast(f"""for(int {il_low_name} = 0;
                                        {il_low_name} < {block_size};
                                        {il_low_name}++){{
                                    }}""")
        for_low.stmt = c_ast.Compound(inner_for)

        # Update il_name
        il_name_subst_ast = expr_c_to_ast(f"{il_high_name} * {block_size} + {il_low_name}")
        at.c_ast_replace_id(for_low, il_name, il_name_subst_ast)

        stmts.append(for_low)

        # Store
        if ref_is_write:
            stmts.append(stmt_c_to_ast(Gencode.cgen_dma_st(*cgen_dma_args) + ';'))

        topcomp.block_items = stmts

        # print(at.ast_to_c_highlight(topcomp))

    dma_efficiency = dma_transfer_size / dma_size
    log.debug(f"             {dma_size=}")
    log.debug(f"    {dma_transfer_size=}")
    log.debug(f"{dma_transfer_size_eff=}")
    efficiency = dma_transfer_size_eff / dma_transfer_size
    log.debug(f" ------>      DMA USAGE = {dma_transfer_size/dma_size*100}%")
    log.debug(f" ------> DMA EFFICIENCY = {efficiency*100}%")
    EFF_THRESHOLD = 0.1
    if efficiency < EFF_THRESHOLD:  # less than 1 ochet used for 10 loaded
        log.warning(f"!!! {efficiency=} < {EFF_THRESHOLD} !!! You must do coalescing")
    log.debug(at.ast_to_c_highlight(ast))
    return dma_efficiency


# On veux 3 indicateurs:
#    - CPU   : l'algo ajoute peu de cycles cpu.
#    - DEBIT : les transferts mémoires sont de la taille du DMA au mieux.
#              On veux amortir le coup fixe
#    - PROG  : l'algo ne doit pas trop augmenter la taille du programme (bin)
#
# 3 optim possible:
# 1) Insertion entre les for.
#    - +CPU+PROG / -DEBIT (Sauf si on a de la chance)
#    - On minimise la taille du code
#    - Bien si taille proche de DMAMAX
# 2) Insertion dans le for avec un if.
#    - +DEBIT+PROG / -CPU
#    - On maximise le débit RAM
#    - Gros surcout cpu.
# 3) Division des boucles <=> 1) ?
#    - Surcoup taille programme
#    - +DEBIT+CPU / -PROG
# 4) Deroulage/Renroulage
#    - On deroule tout, on ajoute les transferts mem, on réenroule
#    - +DEBIT+CPU / -PROG

# EXEMPLE ALGO 3
#
# FOR int j = 0; j < 64; j++      # access L1 = 64*64
#  | FOR int i = 0; i < 64; i++   # access L0 = 64
#  |  | REF tab0[j][i]
#
# ===> EXEMPLES #DMA = 32 (#DMA < 1L0)
#
# FOR int j = 0; j < 64; j++
#  | FOR int mm = 0; mm < 2; mm++
#  |  | DMA tab[l][mm*32] -> BUFF # 32
#  |  | FOR int i2 = 0; i2 < 32; i2++
#  |  |  | REF BUFF[i2]
#
# ===> EXEMPLES #DMA = 128 (#DMA > 2L0 && #DMA < 1L1)
#
# FOR int j = 0; j < 64; j++
#  | DMA tab[j][0] -> BUFF # 128         # INSERT HERE
#  | FOR int mm = 0; mm < 2; mm++, j++   #
#  |  | FOR int i = 0; i < 64; i++
#  |  |  | REF BUFF[i + mm*64]
#
# ===> EXEMPLES #DMA = 64*8 (#DMA > 8L0 && #DMA < 1L1)
#
# FOR int j = 0; j < 64; j++
#  | DMA tab[j][0] -> BUFF #DMA          # INSERT HERE
#  | FOR int mm = 0; mm < 8; mm++, j++   #
#  |  | FOR int i = 0; i < 64; i++
#  |  |  | REF BUFF[i + mm*64]
#


# EXEMPLE ALGO 3 UNALIGNED
# FOR int j = 0; j < 6; j++      # access L1 = 6*6
#  | FOR int i = 0; i < 6; i++   # access L0 = 6
#  |  | REF tab0[j][i]
#
# ===> EXEMPLES #DMA = 25 (#DMA > 4 L0 && #DMA < 1L1)
# ===> VERSION AVEC IF
# FOR int j = 0; j < 6; j++
#  | DMA tab[j][0] -> BUFF #MIN(4*6, (6-j)*4)) # INSERT HERE
#  | FOR int mm = 0; mm < 4 && j < 6; mm++, j++   #
#  |  | FOR int i = 0; i < 6; i++
#  |  |  | REF BUFF[i + mm*6]
#
# ===> VERSION AVEC DUPLICATION BOUCLE
# int j = 0
# FOR ; j < NEW; j++      # NEW = LAST - LAST%4
#  | DMA tab[j][0] -> BUFF #(4*6)) # INSERT HERE
#  | FOR int mm = 0; mm < 4; mm++, j++
#  |  | FOR int i = 0; i < 6; i++
#  |  |  | REF BUFF[i + mm*6]
# DMA tab[j][0] -> BUFF #(2*4)
# FOR int mm = 0; mm < 2; mm ++
#  | FOR int i = 0; i < 6; i++
#  |  | REF BUFF[i + mm*6]
#
# # + Aucun curcout cpu !!
# # - Programme plus gros / plus compliqué à unroll ?


def main(filename):
    ast = at.file_to_ast(filename)
    do_memory_mapping(ast)
    print("CGEN")
    print(at.ast_to_c_highlight(ast))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a filename as argument")
