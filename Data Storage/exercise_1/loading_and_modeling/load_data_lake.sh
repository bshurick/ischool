#!/usr/bin/env bash

# Remove old files
hadoop fs -rm -r -f /user/w205/hospital_compare
hadoop fs -mkdir /user/w205/hospital_compare
hadoop fs -mkdir /user/w205/hospital_compare/hospitals
hadoop fs -mkdir /user/w205/hospital_compare/effective_care
hadoop fs -mkdir /user/w205/hospital_compare/readmissions
hadoop fs -mkdir /user/w205/hospital_compare/measure_dates
hadoop fs -mkdir /user/w205/hospital_compare/surveys_responses

# Rename files
cat 'Hospital General Information.csv' | awk 'NR>1' > hospitals.csv
cat 'Timely and Effective Care - Hospital.csv' | awk 'NR>1' > effective_care.csv
cat 'Readmissions and Deaths - Hospital.csv' | awk 'NR>1' > readmissions.csv
cat 'Measure Dates.csv' | awk 'NR>1' > measure_dates.csv
cat hvbp_hcahps_05_28_2015.csv | awk 'NR>1' > surveys_responses.csv

# Send files to HDFS
hadoop fs -copyFromLocal hospitals.csv /user/w205/hospital_compare/hospitals/hospitals.csv
hadoop fs -copyFromLocal effective_care.csv /user/w205/hospital_compare/effective_care/effective_care.csv
hadoop fs -copyFromLocal readmissions.csv /user/w205/hospital_compare/readmissions/readmissions.csv
hadoop fs -copyFromLocal measure_dates.csv /user/w205/hospital_compare/measure_dates/measure_dates.csv
hadoop fs -copyFromLocal surveys_responses.csv /user/w205/hospital_compare/surveys_responses/surveys_responses.csv
