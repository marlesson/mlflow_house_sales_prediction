# Locust An open source load testing tool.
#
# https://locust.io/

from locust import HttpLocust, TaskSet, task
import json

def predict(l):
    headers = {'content-type': 'application/json'}
    data    = [
                  {
                    "bedrooms": 6,
                    "bathrooms": 3.0,
                    "sqft_living": 3420,
                    "sqft_lot": 22421,
                    "floors": 1.0,
                    "waterfront": 0,
                    "view": 0,
                    "condition": 5,
                    "grade": 9,
                    "sqft_above": 2270,
                    "sqft_basement": 1150,
                    "yr_built": 1948,
                    "yr_renovated": 0,
                    "zipcode": 98006,
                    "lat": 47.5508,
                    "long": -122.189,
                    "sqft_living15": 2430,
                    "sqft_lot15": 15560
                  }
                ]
              

    l.client.post("/invocations", data=json.dumps(data), headers=headers)

class WebsiteTasks(TaskSet):
    tasks = {predict: 1}

class WebsiteUser(HttpLocust):
    task_set = WebsiteTasks
    min_wait = 5000
    max_wait = 9000
