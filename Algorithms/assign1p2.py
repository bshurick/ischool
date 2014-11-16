#!/usr/bin/python

from re import sub
x = sub(r'[-/]','',raw_input('Input phone #:'))
x_int = int(x)
nums = [ int(i) for i in x ]
sum_nums = sum(nums)
y = x_int - sum_nums
while True:
	y_nums = [ int(i) for i in str(y) ]
	if len(y_nums)==1: break
	y = sum(y_nums)
print y
