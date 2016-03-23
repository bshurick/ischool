
from django.forms import ModelForm
from morality.models import Survey

class SurveyForm(ModelForm):
	class Meta:
		model = Survey
		fields = [ 
			'state','gender',
			'i_develop_strong_emotions_toward_people_i_can_rely_on',
			'parents_should_empower_children_as_much_as_possible_so_that_they_may_follow_their_dreams',
			'moral_standards_should_be_seen_as_individualistic_what_one_person_considers_to_be_moral_may_be_judged_as_immoral_by_another_person',
		]

