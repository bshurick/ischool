#!/usr/bin/env bash

#### spark script 
# download pip
wget https://bootstrap.pypa.io/get-pip.py

# install pip
sudo python get-pip.py

# install numpy 
sudo pip install numpy scipy

# install java
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install default-jdk

# set java home
export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64/jre

# download maven 3.3+
wget http://apache.osuosl.org/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar xvzf apache-maven-3.3.9-bin.tar.gz
cd apache-maven-3.3.9

# install maven
MAVEN_HOME=$( pwd )
sudo ln -s $MAVEN_HOME/bin/mvn /usr/bin/mvn

# download spark
cd ..
wget http://mirror.cc.columbia.edu/pub/software/apache/spark/spark-1.6.2/spark-1.6.2.tgz
tar xvzf spark-1.6.2.tgz

# build spark 
cd spark-1.6.2
build/mvn -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -DskipTests clean package

# deploy keys
ssh-keygen -t rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# startup spark
export SPARK_HOME=$(pwd)
./sbin/spark-config.sh
./sbin/stop-all.sh
./sbin/start-all.sh
cd ~

