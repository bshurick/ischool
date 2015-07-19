from bs4 import BeautifulSoup
import re
f = open('Data/SanFranciscoElectricityUse/SanFranciscoElectricityUse.shp.xml')
xml_doc = f.readlines()
soup = BeautifulSoup(xml_doc[0],'xml')
print re.sub(r'[^\/\.a-zA-Z0-9<>=\"\';,\n\r\s]','',soup.prettify())
