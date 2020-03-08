import requests
import threading
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


def action(username, password):
    api = TaigaAPI(host=HOST)
    api.auth(username, password)

    # # Create a project.
    # project = api.projects.create('Project', 'Description')
    #
    # # Update the description.
    # project.name = 'Updated Description'
    # project.update()
    #
    # # Delete the project.
    # project.delete()


# Function to repeat action.
def run(username):
    while True:
        action(username, '123456')


# Create a user.
register_user('user_1', 'user_1@example.com', '123456')


# Create one thread per user.
for i in range(32):
    t = threading.Thread(target=run, args=(f'user_1',))
    t.start()
