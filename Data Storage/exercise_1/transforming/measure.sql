DROP TABLE IF EXISTS production.Measure;
CREATE TABLE production.Measure
STORED AS ORC 
AS
SELECT 
measure_id ,
measure_name ,
measure_start_date ,
measure_end_date ,
measure_start_quarter ,
measure_end_quarter 
FROM staging.measure_dates
SORT BY measure_id
;
