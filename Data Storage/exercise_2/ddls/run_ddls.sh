#!/usr/bin/env bash
psql -Upostgres -c' create database tcount;'
psql -Upostgres -dtcount -f tweetwordcount.sql
