from pydantic import BaseModel
import joblib
from fastapi import APIRouter

predict_router = APIRouter(prefix='/predict', tags=['Predictions'])

scaler = joblib.load('scaler.pkl2')
model = joblib.load('model.pkl2')




class PersonSchema(BaseModel):
    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: float
    person_home_ownership: str
    loan_amnt: float
    loan_intent: float
    loan_percent_income: float
    credit_score: float
    previous_loan_defaults_on_file: str



@predict_router.post('/')
async def predict(person: PersonSchema) :
    person_dict = person.dict()

    new_person_gender = person_dict.pop( 'person_gender')
    person_gender1_0 = [1 if new_person_gender == 'male' else 0]


    new_person_education = person_dict.pop('person_education')
    person_education1_0 = [1 if new_person_education == 'Bachelor' else 0,
                        1 if new_person_education == 'Doctorate' else 0,
                        1 if new_person_education == 'High School' else 0,
                        1 if new_person_education == 'Master' else 0,
                        ]

    new_home_ownership = person_dict.pop('person_home_ownership')
    home_ownership1_0 =  [1 if new_home_ownership == 'OTHER' else 0,
                         1 if new_home_ownership == 'OWN' else 0,
                         1 if new_home_ownership == 'RENT' else 0,
                         ]


    new_loan_intent = person_dict.pop('loan_intent')
    loan_intent1_0 = [1  if new_loan_intent == 'EDUCATION' else 0,

                      1 if new_loan_intent == 'HOMEIMPROVEMENT' else 0,

                      1 if new_loan_intent == 'MEDICAL' else 0,

                      1 if new_loan_intent == 'PERSONAL' else 0,
                      1 if new_loan_intent == 'VENTURE' else 0
                      ]

    new_defaults = person_dict.pop('previous_loan_defaults_on_file')
    new_defaults_0 = [1 if new_defaults == 'Yes' else 0
    ]

    features = (list(person_dict.values()) + person_gender1_0 + person_education1_0 +
                home_ownership1_0 + loan_intent1_0 + new_defaults_0)

    scaled_data = scaler.transform([features])
    pred = model.predict(scaled_data)[0][1]
    prob = model.predict_proba(scaled_data)[0][1]

    return {"approved": bool(pred), "probability": round(prob, 2)}