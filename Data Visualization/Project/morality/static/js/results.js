
var resultmap_q1 = new Datamap({
  scope: 'usa',
  element: document.getElementById('resultmap_q1'),
  geographyConfig: {
    highlightBorderColor: '#bada55',
    popupTemplate: function(geography, data) {
      return '<div class="hoverinfo"><strong>' + geography.properties.name + '</strong>' +
          '<br />Strongly Agree: ' +  data.stronglyAgree + 
          '<br />Agree: ' +  data.agree + 
          '<br />Somewhat agree: ' + data.somewhatAgree +
          '<br />Neither agree nor disagree: ' + data.neitherAgreeNorDisagree +
          '<br />Somewhat disagree: ' + data.somewhatDisagree +
          '<br />Disagree: ' + data.disagree +
          '<br />Strongly disagree: ' + data.stronglyDisagree +
          '<br />Refused: ' + data.refused + ' </div>'
    },
    highlightBorderWidth: 3
  },

  fills: {
  'Strongly agree': '#5BC3F7',
  'Agree': '#51B9ED',
  'Somewhat agree': '#5185ED',
  'Neither agree nor disagree': '#6B51ED',
  'Somewhat disagree': '#FFD5A1',
  'Disagree ': '#FFC18D', 
  'Strongly disagree': '#E37B47',
  'Refused ': '#B6B6B4',     
  defaultFill: '#B6B6B4'
},data:{{ map1data }} 
});
resultmap_q1.labels();


var resultmap_q2 = new Datamap({
  scope: 'usa',
  element: document.getElementById('resultmap_q2'),
  geographyConfig: {
    highlightBorderColor: '#bada55',
    popupTemplate: function(geography, data) {
      return '<div class="hoverinfo"><strong>' + geography.properties.name + '</strong>' +
          '<br />Strongly Agree: ' +  data.stronglyAgree + 
          '<br />Agree: ' +  data.agree + 
          '<br />Somewhat agree: ' + data.somewhatAgree +
          '<br />Neither agree nor disagree: ' + data.neitherAgreeNorDisagree +
          '<br />Somewhat disagree: ' + data.somewhatDisagree +
          '<br />Disagree: ' + data.disagree +
          '<br />Strongly disagree: ' + data.stronglyDisagree +
          '<br />Refused: ' + data.refused + ' </div>'
    },
    highlightBorderWidth: 3
  },

  fills: {
  'Strongly agree': '#5BC3F7',
  'Agree': '#51B9ED',
  'Somewhat agree': '#5185ED',
  'Neither agree nor disagree': '#6B51ED',
  'Somewhat disagree': '#FFD5A1',
  'Disagree ': '#FFC18D', 
  'Strongly disagree': '#E37B47',
  'Refused ': '#B6B6B4',     
  defaultFill: '#B6B6B4'
},data: {{ map2data }}
});
resultmap_q2.labels();


var resultmap_q3 = new Datamap({
  scope: 'usa',
  element: document.getElementById('resultmap_q3'),
  geographyConfig: {
    highlightBorderColor: '#bada55',
    popupTemplate: function(geography, data) {
      return '<div class="hoverinfo"><strong>' + geography.properties.name + '</strong>' +
          '<br />Strongly Agree: ' +  data.stronglyAgree + 
          '<br />Agree: ' +  data.agree + 
          '<br />Somewhat agree: ' + data.somewhatAgree +
          '<br />Neither agree nor disagree: ' + data.neitherAgreeNorDisagree +
          '<br />Somewhat disagree: ' + data.somewhatDisagree +
          '<br />Disagree: ' + data.disagree +
          '<br />Strongly disagree: ' + data.stronglyDisagree +
          '<br />Refused: ' + data.refused + ' </div>'
    },
    highlightBorderWidth: 3
  },

  fills: {
  'Strongly agree': '#5BC3F7',
  'Agree': '#51B9ED',
  'Somewhat agree': '#5185ED',
  'Neither agree nor disagree': '#6B51ED',
  'Somewhat disagree': '#FFD5A1',
  'Disagree ': '#FFC18D', 
  'Strongly disagree': '#E37B47',
  'Refused ': '#B6B6B4',     
  defaultFill: '#B6B6B4'
}, data:{{ map3data }}
});
resultmap_q3.labels();

