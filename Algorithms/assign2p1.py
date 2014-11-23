import sys

pow2s = lambda x: [ 2**i for i in range(x) ]

def scramble(x):
    if len(x)<=2: return x
    else:
        xlen = len(x)
        hxlen = int(xlen/2)
        h1 = x[:hxlen]
        h2 = x[hxlen:xlen]
        r1 = scramble(h1)
        r2 = scramble(h2)
        return ''.join([ r1[i]+r2[i] for i in range(len(r1)) ])

def main(argv):
    i = 0
    while len(argv) not in pow2s(15):
        argv+=' '
        i += 1
        if i > 9000: sys.exit(1)
    print(scramble(argv))

if __name__=='__main__':
    from sys import argv
    [ main(argv[i]) for i in range(len(argv)) if i>0 ]
