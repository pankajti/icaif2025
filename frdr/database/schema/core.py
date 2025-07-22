from sqlalchemy import  Column, Integer, String, Text, Date, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from frdr.database.connections import engine

from sqlalchemy import Column, Integer, Date, String, JSON, ForeignKey
from sqlalchemy.orm import relationship


Base = declarative_base()

class FedSpeech(Base):
    __tablename__ = "fed_speeches"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    speaker = Column(String, nullable=True)
    title = Column(String, nullable=False)
    location = Column(String, nullable=True)
    url = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    analysis = relationship("FedSpeechAnalysis", back_populates="speech", uselist=False)




class FedSpeechAnalysis(Base):
    __tablename__ = "fed_speech_analysis"

    id = Column(Integer, primary_key=True)
    speech_id = Column(Integer, ForeignKey("fed_speeches.id"), unique=True)

    chair = Column(String)
    emphasis = Column(JSON)  # {"inflation": 0.65, "employment": 0.25, "other": 0.1}
    tags = Column(JSON)  # list of {"paragraph": ..., "theme": ...}

    speech = relationship("FedSpeech", back_populates="analysis")


def main () :

    Base.metadata.create_all(engine)

if __name__ == '__main__':
    main()