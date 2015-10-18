USE production;
SET hive.cli.print.header=true;
SELECT

corr(avg_measure_rank,avg_hcahps_rank) corr_rank 

from (
select
provider_id 
,avg(measure_rank) avg_measure_rank
from (
select 
provider_id 
, percent_rank() OVER (partition by measure_id order by score) measure_rank 
from Procedure
) p 
group by provider_id 
sort by provider_id 
) p2 
left join (
select
provider_id
, avg(hcahps_rank) avg_hcahps_rank
from (
select
provider_id 
, percent_rank() OVER (order by hcahps_base_score) hcahps_rank
from Survey
) s 
group by provider_id 
sort by provider_id 
) s2
on p2.provider_id = s2.provider_id 
;
