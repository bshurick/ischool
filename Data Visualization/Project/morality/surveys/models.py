from __future__ import unicode_literals

from django.db import models

class Survey(models.Model):
	STATES = [('AL', 'AL'), ('AK', 'AK'), ('AZ', 'AZ'), ('AR', 'AR'), ('CA', 'CA'), ('CO', 'CO'), ('CT', 'CT'), ('DE', 'DE'), ('DC', 'DC'), ('FL', 'FL'), ('GA', 'GA'), ('HI', 'HI'), ('ID', 'ID'), ('IL', 'IL'), ('IN', 'IN'), ('IA', 'IA'), ('KS', 'KS'), ('KY', 'KY'), ('LA', 'LA'), ('ME', 'ME'), ('MD', 'MD'), ('MA', 'MA'), ('MI', 'MI'), ('MN', 'MN'), ('MS', 'MS'), ('MO', 'MO'), ('MT', 'MT'), ('NE', 'NE'), ('NV', 'NV'), ('NH', 'NH'), ('NJ', 'NJ'), ('NM', 'NM'), ('NY', 'NY'), ('NC', 'NC'), ('ND', 'ND'), ('OH', 'OH'), ('OK', 'OK'), ('OR', 'OR'), ('PA', 'PA'), ('RI', 'RI'), ('SC', 'SC'), ('SD', 'SD'), ('TN', 'TN'), ('TX', 'TX'), ('UT', 'UT'), ('VT', 'VT'), ('VA', 'VA'), ('WA', 'WA'), ('WV', 'WV'), ('WI', 'WI'), ('WY', 'WY')]
	Q1  = ((-1,'Refused')
                ,(1, 'Strongly disagree')
                ,(2, 'Disagree')
                ,(3, 'Somewhat disagree') 
		,(4, 'Neither agree nor disagree')
                ,(5, 'Somewhat agree')
                ,(6, 'Agree')
                ,(7, 'Strongly agree'))
	Q3 = ((-1, 'Refused')
                ,(1, 'Strongly agree')
                ,(2, 'Agree')
                ,(3, 'Somewhat agree')
                ,(4, 'Neither agree nor disagree')
                ,(5, 'Somewhat disagree')
                ,(6, 'Disagree')
                ,(7, 'Strongly disagree'))
	state = models.CharField(max_length=2, choices=STATES)
	gender = models.CharField(max_length=1, choices=(('Male','Male'),('Female','Female')))
	i_develop_strong_emotions_toward_people_i_can_rely_on = models.CharField(max_length=1, choices=Q1)
	parents_should_empower_children_as_much_as_possible_so_that_they_may_follow_their_dreams = models.CharField(max_length=1, choices=Q1)
	moral_standards_should_be_seen_as_individualistic_what_one_person_considers_to_be_moral_may_be_judged_as_immoral_by_another_person = models.CharField(max_length=1, choices=Q3)
	video_choice = models.CharField(max_length=10, blank=True, null=True)
