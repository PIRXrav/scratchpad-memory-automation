#!/usr/bin/env python

#===- cindex-includes.py - cindex/Python Inclusion Graph -----*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


from pprint import pprint

def get_diag_info(diag):
    print(dir(diag))
    return { 'severity' : diag.severity,
             'location' : diag.location,
             'spelling' : diag.spelling,
             'ranges' : list(diag.ranges),
             'fixits' : list(diag.fixits),
             'option' : diag.option,
             'category_name' : diag.category_name, 
             'category_number' : diag.category_number,
             'children' : diag.children,
             'disable_option' : diag.disable_option, 
             'format' : diag.format, 
             'from_param' : diag.from_param, 
             'ptr' : diag.ptr}


def get_cursor_id(cursor, cursor_list = []):
    if cursor is None:
        return None

    # FIXME: This is really slow. It would be nice if the index API exposed
    # something that let us hash cursors.
    for i,c in enumerate(cursor_list):
        if cursor == c:
            return i
    cursor_list.append(cursor)
    return len(cursor_list) - 1

def get_info(node, depth=0):
    children = [get_info(c, depth+1)
                for c in node.get_children()]
    return {'kind' : node.kind,
            'children' : children }
    return { 'id' : get_cursor_id(node),
             'kind' : node.kind,
             'usr' : node.get_usr(),
             'spelling' : node.spelling,
             'location' : node.location,
             'extent.start' : node.extent.start,
             'extent.end' : node.extent.end,
             'is_definition' : node.is_definition(),
             'definition id' : get_cursor_id(node.get_definition()),
             'children' : children }


def exportc(node):
    return node.get_tokens + "\n".join((exportc(c) for c in node.get_children()))

import clang

def main():
    import sys
    from clang.cindex import Index

    from optparse import OptionParser, OptionGroup

    parser = OptionParser("usage: %prog [options] {filename} [clang-args*]")
    parser.disable_interspersed_args()
    (opts, args) = parser.parse_args()
    if len(args) == 0:
        parser.error('invalid number arguments')

    # FIXME: Add an output file option
    out = sys.stdout

    index = Index.create(excludeDecls=False)
    tu = index.parse(None, args)
    if not tu:
        parser.error("unable to load input")

    # print(dir(tu))
    print(dir(tu.cursor))

    def traverse(node):
        # print(dir(node))
        print(node.displayname)
        
        for child in node.get_children():
            traverse(child)
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
            pass
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            pass

    # traverse(tu.cursor)

    pprint(('diags', [get_diag_info(d) for d in  tu.diagnostics]))
    pprint(('nodes', get_info(tu.cursor)))

    print("".join(map(lambda x:x.spelling, tu.cursor.get_tokens())))
    # A helper function for generating the node name.
    def name(f):
        if f:
            return "\"" + f.name + "\""

    # Generate the include graph
    out.write("digraph G {\n")
    for i in tu.get_includes():
        line = "  ";
        if i.is_input_file:
            # Always write the input file as a node just in case it doesn't
            # actually include anything. This would generate a 1 node graph.
            line += name(i.include)
        else:
            line += '%s->%s' % (name(i.source), name(i.include))
        line += "\n";
        out.write(line)
    out.write("}\n")

if __name__ == '__main__':
    main()
