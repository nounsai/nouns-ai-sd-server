import sys
import requests

from old.db import fetch_requests

if __name__ == '__main__':
    requests_df = fetch_requests()
    requests_df = requests_df[requests_df['state'] != 'DONE']
    if len(requests_df[requests_df['state'] == 'PROCESSING']) > 0:
        sys.exit(0)
    else:
        requests_df = requests_df[requests_df['state'] == 'QUEUED']
        request_id = requests_df.iloc[-1]['id']
        url = "http://localhost:5000/requests/{}/process".format(request_id)
        headers = { 'challenge-token': "OdKwF2webnHi9sp6ZgW5qunaW@s4eOm8#xqXDT06AXtwqsUw^%A2" }
        response = requests.request("GET", url, headers=headers)
