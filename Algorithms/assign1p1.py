#!/usr/bin/python
from sys import argv
test = lambda x,y: y if x%y==0 else 0
def run_try(x):
	y = 0
	z = x
	primes = set()
	while True:
		y = max(test(z,i) for i in range(z/2+2) if i>0)
		primes.add(z/y)
		if z/y == z: break
		else: z = y 
	return primes
print str(run_try(int(argv[1])))

