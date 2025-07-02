import os
import smtplib
import imghdr
from email.message import EmailMessage
import os
import smtplib
import imghdr
from email.message import EmailMessage
EMAIL_ADDRESS = os.environ.get('testing.pratham@gmail.com')
EMAIL_PASSWORD = os.environ.get("testing.pratham@123")

# contacts = ['YourAddress@gmail.com', 'test@example.com']

msg = EmailMessage()
msg['Subject'] = 'Check out Bronx as a puppy!'
msg['From'] = EMAIL_ADDRESS
msg['To'] = 'pratham.solanki12@gmail.com'

msg.set_content('Butterworth filter using IIR graphs')

files = ['graph1.pdf', 'graph2.pdf']

for file in files:
    with open(file, 'rb') as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login("testing.pratham@gmail.com", "testing.pratham@123")
    smtp.send_message(msg)
print("email sent successfully ")
