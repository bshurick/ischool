
from django.forms import ModelForm
from surveys.models import Survey

class SurveyForm(ModelForm):
	class Meta:
		model = Survey
		fields = '__all__'
