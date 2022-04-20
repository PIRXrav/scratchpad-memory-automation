"""Some c gencode funcs
"""

from pygments import highlight
from pygments.lexers import CLexer
from pygments.formatters import TerminalFormatter

def ppdict(dict):
    """Pretty print dict
    """
    def ident(s):
        return s.replace("\n", "\n" + (" " * 10 + " | "))

    return "\n".join((f"{k:>10} : {ident(str(v))}" for k, v in dict.items()))


def comment_header(title, **kwargs):
    """Generate c header
    """
    comment_raw = " * " + ppdict(kwargs).replace("\n", "\n * ")
    return f"/**{title}\n *\n" + comment_raw + "\n *\n */\n\n"


def c_highlight(code):
    """Print highlighted code in terminal
    """
    formatter = TerminalFormatter(bg="dark", linenos=True)
    return highlight(code, CLexer(), formatter)
