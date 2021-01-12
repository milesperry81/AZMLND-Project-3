import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://fff526da-be67-4d69-85f7-1a4825c61903.eastus.azurecontainer.io/score'
#scoring_uri = 'http://a4fcf479-b8a6-4646-b5f2-f5c45351c624.eastus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
#key = 'kib3kJNDahThtfar7P7CaGJeNXKduIXM'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 75,
            "anaemia": 0,
            "creatinine_phosphokinase": 582,
            "diabetes": 0,
            "ejection_fraction": 20,
            "high_blood_pressure": 1,
            "platelets": 265000,
            "serum_creatinine": 1.9,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 0,
            "time": 4
          },
          {
            "age": 55,
            "anaemia": 0,
            "creatinine_phosphokinase": 7861,
            "diabetes": 0,
            "ejection_fraction": 38,
            "high_blood_pressure": 0,
            "platelets": 263358.03,
            "serum_creatinine": 1.1,
            "serum_sodium": 136,
            "sex": 1,
            "smoking": 0,
            "time": 6
          }
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


