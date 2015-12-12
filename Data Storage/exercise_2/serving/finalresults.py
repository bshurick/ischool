#!/usr/bin/env python
from __future__ import print_function
import psycopg2, sys
import datetime as DT

DB = 'tcount'
TBL = 'Tweetwordcount'

def get_one_result(word):
	SQL = '''
		SELECT cnt FROM {tbl}
		WHERE day='{day}'
		AND word='{word}'
		;
	'''.format(tbl=TBL,word=word,day=DT.datetime.now().strftime('%Y-%m-%d'))
	conn = psycopg2.connect("user=postgres dbname='{}'".format(DB))
	cur = conn.cursor()
	cur.execute(SQL)
	result = cur.fetchone()
	cur.close()
	conn.close()
	if result: return result[0]
	else: return 0

def get_more_results():
	SQL = '''
		SELECT word, cnt
		FROM {tbl}
		WHERE day='{day}'
		;
	'''.format(db=DB,tbl=TBL,day=DT.datetime.now().strftime('%Y-%m-%d'))
	conn = psycopg2.connect("user=postgres dbname='{}'".format(DB))
	cur = conn.cursor()
	cur.execute(SQL)
	result = sorted(cur.fetchall(),key=lambda x: x[0])
	cur.close()
	conn.close()
	if result: return result
	else: return []

if __name__=='__main__':
	args_len = len(sys.argv)
	if args_len==2:
		word = sys.argv[1].lower()
		print('Total number of occurences of "{}": {}'.format(word,get_one_result(word)))
	if args_len==1:
		print(', '.join([ '({},{})'.format(f[0],f[1]) for f in get_more_results() ]))

