USE production;
SET hive.cli.print.header=true;
SELECT

h.state
,avg(measure_rank) avg_measure_rank
,stddev(measure_rank) stddev_measure_rank 
,sum(measure_rank) sum_measure_rank
,count(measure_rank) cnt_measure_rank

from (
select 
provider_id
, percent_rank() OVER (partition by measure_id order by score) measure_rank 
from Procedure
) m
join Hospital h
on m.provider_id = h.provider_id 

group by h.state
order by avg_measure_rank desc 
limit 10
;
