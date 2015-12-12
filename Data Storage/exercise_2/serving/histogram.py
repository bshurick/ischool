#!/usr/bin/env python
from __future__ import print_function
import psycopg2, sys
import datetime as DT

DB = 'tcount'
TBL = 'Tweetwordcount'

def get_hist(st,ed):
	SQL = '''
		SELECT word, cnt
		FROM {tbl}
		WHERE cnt between {st} and {ed}
		;
	'''.format(	
		tbl=TBL
		,st=st
		,ed=ed
	)
	conn = psycopg2.connect("user=postgres dbname='{}'".format(DB))
        cur = conn.cursor()
        cur.execute(SQL)
        result = sorted(cur.fetchall(),key=lambda x: x[1],reverse=True)
        cur.close()
        conn.close()
	if result: return result
	else: return []

if __name__=='__main__':
	if len(sys.argv)==3:
		res = get_hist(sys.argv[1],sys.argv[2])
		for r in res:
			print('{} {}'.format(r[0],r[1]))
	else:
		raise Exception('Need to have [start] and [end] as argv') 
