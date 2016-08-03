#!/usr/bin/env bash

#### spark script 
# download pip
wget https://bootstrap.pypa.io/get-pip.py

# install pip
sudo python get-pip.py

# update debian packages
sudo apt-get update

# install git
sudo apt-get install -y git

# install g++
sudo apt-get install -y g++
sudo apt-get install -y make

# install java
sudo apt-get install -y default-jre
sudo apt-get install -y default-jdk

# set java home
export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64/jre

# install python dev
sudo apt-get install -y python-dev

# install numpy 
sudo pip install numpy scipy ipython pandas sklearn keras xgboost

# mount file system
sudo mkfs -t ext4 /dev/xvdb
sudo mkdir /Data
sudo mkdir /Data/tmp
sudo mkdir /Data/tmp/logs
sudo mkdir /Data/tmp/jobs
sudo mount /dev/xvdb /Data
sudo chown -R ubuntu:ubuntu /Data
cd /Data
wget https://www.dropbox.com/s/h4ceb0togkejs99/WISDM_ar_latest.tar
tar xf WISDM_ar_latest.tar
cd WISDM_ar_v1.1/
cat WISDM_ar_v1.1_raw.txt | awk -F',' 'BEGIN { OFS=","; RS=";"; ORS=";"; } \
{ if (NF==6) { print $0,"" } else if (NF==7) { print $0 }}' > raw_data_fixed.txt
cd ~
# move data into /Data folder


# download maven 3.3+
wget http://apache.osuosl.org/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar xvzf apache-maven-3.3.9-bin.tar.gz
cd apache-maven-3.3.9

# install maven
MAVEN_HOME=$( pwd )
sudo ln -s $MAVEN_HOME/bin/mvn /usr/bin/mvn

# download spark
cd ..
wget http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0.tgz
tar xvzf spark-2.0.0.tgz

# build spark 
cd spark-2.0.0
build/mvn -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -DskipTests clean package

# deploy keys
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# create conf/spark-defaults.conf
echo 'spark.driver.memory              20g'> conf/spark-defaults.conf
echo 'spark.executor.memory            20g'>> conf/spark-defaults.conf
echo 'spark.executor.cores             3'>> conf/spark-defaults.conf

# create conf/spark-env.sh 
echo '#!/usr/bin/env bash' > conf/spark-env.sh
echo 'SPARK_LOCAL_DIRS=/Data/tmp' >> conf/spark-env.sh
echo 'SPARK_WORKER_DIR=/Data/tmp/jobs' >> conf/spark-env.sh
echo 'SPARK_LOG_DIR=/Data/tmp/logs' >> conf/spark-env.sh
echo 'SPARK_WORKER_INSTANCES=2' >> conf/spark-env.sh

# startup spark
export SPARK_HOME=$(pwd)
./sbin/spark-config.sh
./sbin/stop-all.sh
./sbin/start-all.sh
cd ~

