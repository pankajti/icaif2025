from frdr.database.schema.core import FedSpeech
from frdr.database.dao.fed_speech_dao import FedSpeechDao
from graph.create_pipeline import create_fed_speech_pipeline
from frdr.database.schema.core import FedSpeechAnalysis
from datetime import date
from tqdm import tqdm
import pandas as pd
import logging
from sqlalchemy import not_

LOG_FILE = "fed_speech_analysis_single.log"

# --- Logger Setup ---
def setup_logger():
    # Configure logging for each process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'), # 'a' for append to log file
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger() # Initialize logger in the main process as well


class FedSpeechPipelineRunner:
    def __init__(self, dao: FedSpeechDao):
        self.dao = dao
        self.graph = create_fed_speech_pipeline()

    def get_speeches_to_analyze(self, limit=None):
        """
        Retrieves speeches that have not yet been analyzed.
        """
        # 1. Get IDs of already analyzed speeches
        logger.info("Querying for already analyzed speech IDs...")
        analyzed_speech_ids = self.dao.session.query(FedSpeechAnalysis.speech_id).distinct().all()
        # Convert list of tuples to a set for efficient lookup
        analyzed_speech_ids = {s_id for (s_id,) in analyzed_speech_ids}
        logger.info(f"Found {len(analyzed_speech_ids)} speeches already analyzed.")

        # 2. Query for all speeches, excluding those already analyzed
        query = self.dao.session.query(FedSpeech).order_by(FedSpeech.date.desc())

        if analyzed_speech_ids:
            # Use SQLAlchemy's not_() and .in_() to exclude IDs
            query = query.filter(not_(FedSpeech.id.in_(analyzed_speech_ids)))
            logger.info("Filtering out already analyzed speeches...")

        if limit is not None:
            query = query.limit(limit)
            logger.info(f"Applying limit of {limit} to remaining speeches.")

        speeches_to_process = query.all()
        logger.info(f"Identified {len(speeches_to_process)} speeches for analysis.")
        return speeches_to_process

    def get_speeches(self, limit=10):

        all_speeches = self.dao.session.query(FedSpeech).order_by(FedSpeech.date.desc())
        return list(all_speeches)
        #return self.dao.session.query(FedSpeech).order_by(FedSpeech.date.desc()).limit(limit).all()

    def analyze_speech(self, speech: FedSpeech):
        paragraphs = speech.text.split("\n\n")
        result = self.graph.invoke({"paragraphs": paragraphs})
        analysis = {
            "id": speech.id,
            "title": speech.title,
            "date": speech.date,
            "speaker": speech.speaker,
            "emphasis": result['emphasis'],
            "tags": result['tags']
        }
        return analysis

    def classify_chair(self,speech_date: date) -> str:
        """
        Returns the name of the Fed Chair based on the speech date.
        """
        if speech_date < date(1987, 8, 11):
            return "Pre-Greenspan"
        elif date(1987, 8, 11) <= speech_date < date(2006, 2, 1):
            return "Alan Greenspan"
        elif date(2006, 2, 1) <= speech_date < date(2014, 2, 3):
            return "Ben Bernanke"
        elif date(2014, 2, 3) <= speech_date < date(2018, 2, 5):
            return "Janet Yellen"
        elif date(2018, 2, 5) <= speech_date:
            return "Jerome Powell"
        else:
            return "Unknown"

    def run(self, limit=10):
        speeches = self.get_speeches_to_analyze(limit=None)
        results = []
        for idx , speech in tqdm(enumerate(speeches), desc="Analysing result", total=len(speeches)):
            try:

                analysis = self.analyze_speech(speech)
                results.append(analysis)
                logger.info(f"Completed analysis of  {speech.id}")

                analysis_db = FedSpeechAnalysis(speech_id=speech.id)
                analysis_db.emphasis=analysis['emphasis']
                #analysis_db.tags=analysis['tags']
                analysis_db.chair = self.classify_chair(analysis['date'])

                self.dao.session.add(analysis_db)
                if idx%10==0:
                    self.dao.session.commit()
                    logger.info(f"loaded {idx} data ")

            except Exception as e:
                logger.error(f"❌ Failed to analyze speech ID {speech.id}")
                self.dao.session.rollback()
                #raise e

        self.dao.session.commit()
        results_df = pd.DataFrame(results)

        results_df.to_csv('results_df.csv', index=False)

        return results_df



if __name__ == '__main__':
    dao = FedSpeechDao()
    runner = FedSpeechPipelineRunner(dao)
    results = runner.run()
    print(results)