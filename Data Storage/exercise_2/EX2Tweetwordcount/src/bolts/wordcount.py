from __future__ import absolute_import, print_function, unicode_literals
from collections import Counter
from streamparse.bolt import Bolt
import psycopg2
import datetime as DT
TBL='Tweetwordcount'
DB='tcount'

class WordCounter(Bolt):
	def initialize(self, conf, ctx):
		self.counts = Counter()
	
	def incrementPostgres(word):
		conn = psycopg2.connect("user=postgres dbname='{}'".format(DB))
		cur = conn.cursor()
		cur.execute('''SELECT * from {}
			WHERE word='{}' and day='{}';'''.format(
				TBL, word, DT.datetime.now().strftime('%Y-%m-%d')))
		if cur.fetchone():
			SQL = '''UPDATE {}
				SET cnt=cnt+1
				WHERE word='{}'
				AND day='{}';
			'''.format(TBL,word,
					DT.datetime.now().strftime('%Y-%m-%d'))
		else:
			SQL = '''INSERT INTO {}
				(day, word, cnt)
				VALUES ('{}','{}','{}');
				'''.format(TBL,DT.datetime.now().strftime('%Y-%m-%d'),
						word)
		cur.execute(SQL)
		conn.commit()
		cur.close()
		conn.close()
	
	def process(self, tup):
		word = tup.values[0]
		word = word.lower()
		
		incrementPostgres(word)
		
		# Increment the local count
		self.counts[word] += 1
		self.emit([word, self.counts[word]])
		self.log('{}: {}'.format(word, self.counts[word]))

