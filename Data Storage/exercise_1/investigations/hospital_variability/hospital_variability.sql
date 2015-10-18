USE production;
SET hive.cli.print.header=true;
SELECT

h.measure_name 
, stddev(measure_rank) stddev_score 

from (
select 
measure_id 
, percent_rank() OVER (partition by provider_id order by score) measure_rank 
from Procedure
) m
join Measure h
on m.measure_id = h.measure_id 

group by h.measure_name
order by stddev_score desc 
limit 10
;
