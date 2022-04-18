import subprocess

def write_file(filename, code):
    """write file filename with code as data"""
    with open(filename, "w") as f:
        f.write(code)


def shell(cmd, verbose=False):
    """Shell cmd wrapper"""
    if verbose:
        print(cmd)
    try:
        res = subprocess.check_output(cmd, shell=True).decode('UTF-8')
        if verbose:
            print(res)
    except subprocess.CalledProcessError as grepexc:
        print("error code", grepexc.returncode, grepexc.output.decode('UTF-8'))
        return grepexc.returncode
    return 0

def cpp(code, config):
    """Python CPP wrapper"""
    tmpf = "/tmp/sma_cpp"
    with open(tmpf, "w") as f:
        f.write(code)
    cpp_defines = " ".join(map(lambda c: f"-D{c[0]}={c[1]}", config.items()))
    return subprocess.check_output(
        f"cpp -P {cpp_defines} {tmpf}", shell=True, universal_newlines=True
    )
