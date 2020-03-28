import threading

import requests
from taiga import TaigaAPI

# python3 manage.py flush --noinput && python3 manage.py loaddata initial_project_templates && python3 manage.py runserver
# make -j8 && cp python.exe ../taiga-venv/bin/python && cp python.exe ../taiga-venv/bin/python3 && cp python.exe ../taiga-venv/bin/python3.7 && cp python.exe ../python3/bin/python3.7
HOST = 'http://localhost:8000'


def register_user(username, email, password):
    r = requests.post(HOST + '/api/v1/auth/register', json={
        'email': email,
        'full_name': 'User {}'.format(username),
        'password': password,
        'type': 'public',
        'username': username,
        'accepted_terms': True
    })
    return r.status_code == 201


def action(api):
    # Get the project.
    project = api.projects.get(1)

    # List issues.
    project.list_issues()


# Function to repeat action.
def run():
    while True:
        action(api)


# Create a user.
register_user('user_1', 'user_1@example.com', '123456')

# Auth.
api = TaigaAPI(host=HOST)
api.auth('user_1', '123456')

# # Create a project.
# new_project = api.projects.create('project', 'project')
#
# # Create many new issues.
# for _ in range(1000):
#     new_project.add_issue(
#         'New Issue',
#         new_project.priorities.get(name='High').id,
#         new_project.issue_statuses.get(name='New').id,
#         new_project.issue_types.get(name='Bug').id,
#         new_project.severities.get(name='Minor').id,
#         description='Bug #5'
#     )

# Create one thread per user.
for i in range(32):
    t = threading.Thread(target=run)
    t.start()
