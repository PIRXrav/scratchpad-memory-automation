import numpy as np


def concaty(*args):
    return '\n'.join(map(lambda t: ''.join(t), zip(*map(lambda s: s.split('\n'), args))))


def termhist(a, title, xsize=10, ysize=10):
    values, _ = np.histogram(a, bins=xsize)
    normv = 1 / np.max(values) * ysize
    mat = np.zeros((xsize, ysize))
    for i_v, v in enumerate(values):
        for k in range(int((v * normv))):
            mat[i_v][k] = v
    d = 6
    hist = '-' * d + '+' + '-' * xsize + '+' + '\n'
    hist += ' ' * d + "|" + title.center(xsize) + "|\n"
    hist += '-' * d + '+' + '-' * xsize + '+' + '\n'
    for y in range(ysize):
        hist += f'{int(((ysize-y-1)/normv)):>5} |'
        for x in range(xsize):
            hist += '\u2588' if mat[x][ysize - y - 1] else ' '
        hist += '|\n'
    hist += '-' * d + '+' + '-' * xsize + '+' + '\n'
    return hist


def termhists(aa, titles, **kwargs):
    return concaty(*tuple(map(lambda o: termhist(*o, **kwargs), zip(aa, titles))))


if __name__ == '__main__':
    x = np.random.normal(size=2000)
    print(x)
    print(termhist(x, 'x'))
    print(termhists([np.random.normal(size=200) for _ in range(3)], ['x', 'y', 'z'], ysize=20, xsize=25))
