DROP TABLE IF EXISTS production.Hospital;
CREATE TABLE production.Hospital 
STORED AS ORC 
AS
SELECT 
provider_id ,
hospital_name ,
address ,
city ,
state ,
zip_code ,
county_name ,
phone_number ,
hospital_type ,
hospital_ownership ,
emergency_services
FROM staging.hospitals  
SORT BY provider_id 
;
