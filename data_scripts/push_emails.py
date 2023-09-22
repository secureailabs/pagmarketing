import email
import imaplib
from email.header import decode_header
import time
from email.mime.text import MIMEText
import smtplib
import json
import streamlit as st
import time

json_file = st.file_uploader("Choose Email JSON file")
username = st.text_input("Gmail user")
password = st.text_input("Pass(word", type="password")
receiver_email = st.text_input("Email Address")
send = st.button("Send Email")
if json_file:
    #f = open(json_file)
    data = json.load(json_file)
    #f.close()
    st.info(len(data))


if send:
    SMTP_HOST = 'smtp.google.com'
    SMTP_USER = username + "@gmail.com"
    SMTP_PASS = password
    from_email = SMTP_USER
    to_emails = [receiver_email]
    
    
    # Opening JSON file

    
    # Iterating through the json
    # list
    i = 0
    
    for email_info in data:
        i = i + 1
        # if i <= 281: #recently_diagnosed
        #     continue
        if (i <= 50):
            continue
        try:
            body = email_info["body"]
            subject = email_info["subject"]

            headers = f"From: {from_email}\r\n"
            headers += f"To: {', '.join(to_emails)}\r\n" 
            headers += f"Subject: " + subject + "\r\n"
            email_message = headers + "\r\n" + body  # Blank line needed between headers and body

            #st.info(email_message)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            # server.connect('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, password)

            server.sendmail(from_email, to_emails, email_message)
            server.quit()
        except Exception as e:
            #st.info("Error Happened \t" + str(e))
            st.info(i)
            break
        
        if i % 20 == 0:
            st.info(str(i) + " Sent")
            st.info("Sleeping")
            time.sleep(10)
            st.info("Restart")
    st.info("Complete")

