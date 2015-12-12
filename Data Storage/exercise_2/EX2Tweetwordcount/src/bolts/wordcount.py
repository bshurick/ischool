from __future__ import absolute_import, print_function, unicode_literals
from collections import Counter
from streamparse.bolt import Bolt
import re, psycopg2
import datetime as DT
TBL='Tweetwordcount'
DB='tcount'

class WordCounter(Bolt):
	def initialize(self, conf, ctx):
		self.counts = Counter()
	
	def incrementPostgres(self, word):
		conn = psycopg2.connect("user=postgres dbname='{}'".format(DB))
		cur = conn.cursor()
		cur.execute('''SELECT * from {}
			WHERE word='{}' 
			;'''.format(
				TBL, word ))
		if cur.fetchone():
			SQL = '''UPDATE {}
				SET cnt=cnt+1
				WHERE word='{}'
				;
			'''.format(TBL,word)
		else:
			SQL = '''INSERT INTO {}
				(word, cnt)
				VALUES ('{}','{}');
				'''.format(TBL, word, 1)
		cur.execute(SQL)
		conn.commit()
		cur.close()
		conn.close()
	
	def process(self, tup):
		word = tup.values[0]
		word = re.sub(ur'[^\w+]','',word.lower(),flags=re.UNICODE)
		
		self.incrementPostgres(word)
		
		# Increment the local count
		self.counts[word] += 1
		self.emit([word, self.counts[word]])
		self.log('{}: {}'.format(word, self.counts[word]))

