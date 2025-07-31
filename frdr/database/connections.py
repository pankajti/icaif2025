from frdr.config.config import DATABASE_URL
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()