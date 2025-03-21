import smtplib
from email.message import EmailMessage

from celery import Celery

from config import SMTP_PASSWORD, SMTP_USER

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


celery = Celery('tasks', broker='redis://localhost:6379')


def get_template_email(username: str):
    email = EmailMessage()
    email['Subject'] = 'Привет'
    email['From'] = SMTP_USER
    email['To'] = SMTP_USER
    email.set_content(
        '<div>'
        f'<h1 style="color: red;">Здравствуйте, {username}</h1>'
        '</div>',
        subtype='html'
    )
    return email


@celery.task(default_retry_delay=5, max_retries=3)
def send_email(username: str):
    email = get_template_email(username)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        print(server.login(SMTP_USER, SMTP_PASSWORD))
        try:
            server.send_message(email)
        except:
            send_email.retry()
