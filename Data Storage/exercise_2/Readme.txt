Steps to Deploy Application
1) Install pip and then install each of the dependencies listed above
2) Install postgres and ensure server is running
3) Execute ./ddls/run_ddls.sh to create Postgres database and word count table 
4) Within directory EX2Tweetwordcount execute command "sparse run" (to run in background, execute "nohup sparse run")  
5) Execute ./ddls/finalresults.py to see all word count results in real-time
6) Execute ./ddls/finalresults.py [word] to see word count results for [word] in real-time
7) Execute ./ddls/histogram.py [min] [max] to display a sorted list of all words with count at least [min] and less than [max]

