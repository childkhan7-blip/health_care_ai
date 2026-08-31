import os
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from google import genai
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
import json


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Healthcare AI Patient Pre-Diagnosis Agent"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GEMINI API
# =========================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# =========================================================
# TEMPORARY MEMORY
# =========================================================

conversations = {}

# Number of AI follow-up questions asked
question_counts = {}

# Store appointments temporarily
appointments = []


# =========================================================
# DATA MODELS
# =========================================================

class PatientMessage(BaseModel):
    patient_id: str
    message: str


class Appointment(BaseModel):
    patient_id: str
    name: str
    email: EmailStr
    date: str
    time: str


# =========================================================
# EMAIL CONFIGURATION
# =========================================================

# conf = ConnectionConfig(
#     MAIL_USERNAME="rehmankhan8321539@gmail.com",
#     MAIL_PASSWORD="lhrh jkew jqqf avbl",
#     MAIL_FROM="rehmankhan8321539@gmail.com",
#     MAIL_PORT=587,
#     MAIL_SERVER="smtp.gmail.com",
#     MAIL_STARTTLS=True,
#     MAIL_SSL_TLS=False,
#     USE_CREDENTIALS=True
# )

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Healthcare AI Agent is running!"
    }


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat(patient: PatientMessage):

    # -----------------------------------------------------
    # Create new patient conversation
    # -----------------------------------------------------

    if patient.patient_id not in conversations:

        conversations[patient.patient_id] = []

        question_counts[patient.patient_id] = 0


    # -----------------------------------------------------
    # Save patient message
    # -----------------------------------------------------

    conversations[patient.patient_id].append(
        f"Patient: {patient.message}"
    )


    # -----------------------------------------------------
    # Get conversation
    # -----------------------------------------------------

    conversation = "\n".join(
        conversations[patient.patient_id]
    )


    # -----------------------------------------------------
    # Current question number
    # -----------------------------------------------------

    current_question = question_counts[
        patient.patient_id
    ]


    # -----------------------------------------------------
    # Gemini prompt
    # -----------------------------------------------------

    prompt = f"""

You are a Healthcare AI Patient Pre-Diagnosis and Triage Agent.

Your job is to:

1. Understand the patient's symptoms.
2. Ask useful follow-up questions.
3. Perform preliminary triage.
4. Classify the case as:
   - mild
   - urgent
   - critical

IMPORTANT:

The patient should receive a maximum of THREE
follow-up questions.

The current number of follow-up questions already
asked is:

{current_question}

If fewer than 3 follow-up questions have been asked,
ask ONE useful follow-up question.

If 3 follow-up questions have already been asked,
complete the assessment.

Do NOT keep asking questions forever.

---------------------------------------------------------

TRIAGE LEVELS

MILD:
Symptoms appear non-emergency and routine medical
care may be appropriate.

URGENT:
The patient should receive medical attention soon.

CRITICAL:
The patient may have a medical emergency.

For critical cases, clearly tell the patient to seek
emergency medical care immediately.

Never provide a definitive medical diagnosis.

---------------------------------------------------------

IMPORTANT EMERGENCY RULE

If the patient's symptoms indicate a possible emergency,
you MUST clearly tell them to seek emergency medical care
immediately.

The system will still allow appointment booking for
demonstration purposes, but the patient must NOT be told
to wait for the appointment instead of seeking emergency
care.

---------------------------------------------------------

BOOKING

After THREE follow-up questions, assessment must be
completed.

For mild and urgent cases, tell the patient they can
book an appointment.

For critical cases, first emphasize emergency care.
The system will ALSO provide a booking option, but
booking must never delay emergency treatment.

---------------------------------------------------------

CONVERSATION

{conversation}

---------------------------------------------------------

Return ONLY valid JSON.

Use exactly:

{{
    "response": "response to patient",
    "triage": "mild",
    "status": "continue"
}}

Allowed triage:

"mild"
"urgent"
"critical"

Allowed status:

"continue"
"completed"

"""


    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    # -----------------------------------------------------
    # Gemini response
    # -----------------------------------------------------

    ai_text = response.text.strip()


    # Remove markdown JSON fences
    if ai_text.startswith("```"):

        ai_text = ai_text.replace(
            "```json",
            ""
        )

        ai_text = ai_text.replace(
            "```",
            ""
        )

        ai_text = ai_text.strip()


    # -----------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------

    try:

        result = json.loads(ai_text)

    except json.JSONDecodeError:

        result = {
            "response": ai_text,
            "triage": "mild",
            "status": "continue"
        }


    # -----------------------------------------------------
    # Increase question count
    #
    # If Gemini asks a question, this counts as one.
    # -----------------------------------------------------

    if result["status"] == "continue":

        question_counts[
            patient.patient_id
        ] += 1


    # -----------------------------------------------------
    # FORCE COMPLETION AFTER 3 QUESTIONS
    # -----------------------------------------------------

    if question_counts[
        patient.patient_id
    ] >= 3:

        result["status"] = "completed"


    # -----------------------------------------------------
    # Emergency handling
    # -----------------------------------------------------

    if result["triage"] == "critical":

        result["status"] = "completed"

        result["response"] = (
            result["response"]
            + "\n\n🚨 IMPORTANT: Your symptoms may "
              "require immediate medical attention. "
              "Please seek emergency medical care "
              "immediately or contact your local "
              "emergency service. Do not wait for "
              "an appointment."
        )


    # -----------------------------------------------------
    # Save AI response
    # -----------------------------------------------------

    conversations[
        patient.patient_id
    ].append(
        f"AI: {result['response']}"
    )


    # -----------------------------------------------------
    # Determine next action
    # -----------------------------------------------------

    if result["triage"] == "critical":

        next_action = "emergency_and_booking"


    elif result["status"] == "completed":

        next_action = "book_appointment"


    else:

        next_action = "continue_assessment"


    # -----------------------------------------------------
    # SHOW BOOKING
    #
    # Critical cases ALSO get booking option,
    # as requested.
    # -----------------------------------------------------

    if result["status"] == "completed":

        show_booking = True

    else:

        show_booking = False


    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {

        "patient_id": patient.patient_id,

        "response": result["response"],

        "triage": result["triage"],

        "status": result["status"],

        "next_action": next_action,

        "show_booking": show_booking,

        "questions_asked": question_counts[
            patient.patient_id
        ]

    }


# =========================================================
# APPOINTMENT BOOKING
# =========================================================

@app.post("/book-appointment")
async def book_appointment(
    appointment: Appointment
):

    # -----------------------------------------------------
    # Create appointment
    # -----------------------------------------------------

    new_appointment = {

        "patient_id": appointment.patient_id,

        "name": appointment.name,

        "email": str(appointment.email),

        "date": appointment.date,

        "time": appointment.time

    }


    # -----------------------------------------------------
    # Save appointment temporarily
    # -----------------------------------------------------

    appointments.append(
        new_appointment
    )


    # -----------------------------------------------------
    # Confirmation email
    # -----------------------------------------------------

    message = MessageSchema(

        subject="Healthcare Appointment Confirmation",

        recipients=[
            appointment.email
        ],

        body=f"""

Hello {appointment.name},

Your healthcare appointment has been successfully booked.

Appointment Details:

Patient ID: {appointment.patient_id}

Date: {appointment.date}

Time: {appointment.time}

Thank you.

Healthcare AI Patient Pre-Diagnosis Agent

""",

        subtype="plain"
    )


    # -----------------------------------------------------
    # Send email
    # -----------------------------------------------------

    fm = FastMail(conf)

    await fm.send_message(
        message
    )


    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    return {

        "message":
        "Appointment booked and confirmation email sent!",

        "appointment":
        new_appointment

    }