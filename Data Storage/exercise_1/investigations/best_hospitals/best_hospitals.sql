SELECT
h.hospital_name
, sum(p.score) total_score
, avg(p.score) avg_score 
, stddev_pop(p.score) stdev_score 
from production.Hospital h 
	left join production.Procedure p 
	on h.provider_id = p.provider_id 
group by h.hospital_name
order by avg_score desc 
limit 10
;
