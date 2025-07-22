from sqlalchemy.orm import sessionmaker
from frdr.database.connections import engine
from frdr.database.schema.core import FedSpeech
from datetime import datetime

class FedSpeechDao():

    def __init__(self):

        Session = sessionmaker(bind=engine)
        self.session = Session()


    def add_fed_speech(self,new_speech):

        self.session.add(new_speech)
        self.session.commit()


if __name__ == '__main__':
    new_speech = FedSpeech(
        date=datetime(2023, 12, 15),
        speaker="Jerome Powell",
        title="Inflation and the Economy",
        location="Washington, D.C.",
        url="https://www.federalreserve.gov/example",
        text="Full speech text here..."
    )

    dao = FedSpeechDao()
    dao.add_fed_speech(new_speech)


