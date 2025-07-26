from fastapi import FastAPI
import uvicorn
from predict import predict_router

bank_app = FastAPI()
bank_app.include_router(predict_router)




if __name__ == '__main__':
    uvicorn.run(bank_app, host='127.0.0.1', port=8010)




