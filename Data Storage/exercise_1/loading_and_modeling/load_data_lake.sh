
# Rename files
mv 'Hospital General Information.csv' hospitals.csv
mv 'Timely and Effective Care - Hospital.csv' effective_care.csv
mv 'Readmissions and Deaths - Hospital.csv' readmissions.csv
mv 'Measure Dates.csv' measuredates.csv
mv hvbp_hcahps_05_28_2015.csv surveys_responses.csv

# Send files to HDFS
hadoop fs -copyFromLocal hospitals.csv /user/w205/hospital_compare/hospitals.csv
hadoop fs -copyFromLocal effective_care.csv /user/w205/hospital_compare/effective_care.csv
hadoop fs -copyFromLocal readmissions.csv /user/w205/hospital_compare/readmissions.csv
hadoop fs -copyFromLocal measuredates.csv /user/w205/hospital_compare/measuredates.csv
hadoop fs -copyFromLocal surveys_responses.csv /user/w205/hospital_compare/surveys_responses.csv
