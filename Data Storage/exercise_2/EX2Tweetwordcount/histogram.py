#!/usr/bin/env python
from __future__ import print_function
import psycopg2, sys

DB = 'Tcount'
TBL = 'Tweetwordcount'

def get_hist(st,ed):
	SQL = '''
		SELECT word, cnt
		FROM {db}.{tbl}
		WHERE cnt between {st} and {ed}
		AND day={day}
		;
	'''.format(	
		db=DB
		,tbl=TBL
		,day=DT.datetime.now().strftime('%Y-%m-%d')
		,st=st
		,ed=ed
	)
	conn = psycopg2.connect("user=postgres")
        cur = conn.cursor()
        cur.execute(SQL)
        result = sorted(cur.fetchall(),lambda x: x[1],reverse=True)
        cur.close()
        conn.close()

if __name__=='__main__':
	if len(sys.argv)==3:
		res = get_hist(sys.argv[1],sys.argv[2])
		for r in res:
			print('{} {}'.format(r[0],r[1])
	else:
		raise Exception('Need to have [start] and [end] as argv') 
