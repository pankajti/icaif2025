from frdr.database.schema.core import FedSpeech
from frdr.database.dao.fed_speech_dao import FedSpeechDao
from graph.create_pipeline import create_fed_speech_pipeline
from frdr.database.schema.core import FedSpeechAnalysis
from datetime import date

import pandas as pd

class FedSpeechPipelineRunner:
    def __init__(self, dao: FedSpeechDao):
        self.dao = dao
        self.graph = create_fed_speech_pipeline()

    def get_speeches(self, limit=10):
        return self.dao.session.query(FedSpeech).order_by(FedSpeech.date.desc())
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
        speeches = self.get_speeches(limit)
        results = []
        for speech in speeches:
            try:

                analysis = self.analyze_speech(speech)
                results.append(analysis)
                print(f"Completed analysis of  {speech.id}")

                analysis_db = FedSpeechAnalysis(speech_id=speech.id)
                analysis_db.emphasis=analysis['emphasis']
                #analysis_db.tags=analysis['tags']
                analysis_db.chair = self.classify_chair(analysis['date'])

                self.dao.session.add(analysis_db)
                self.dao.session.commit()

            except Exception as e:
                print(f"❌ Failed to analyze speech ID {speech.id}")
                #raise e
        results_df = pd.DataFrame(results)

        results_df.to_csv('results_df.csv', index=False)

        return results_df



if __name__ == '__main__':
    dao = FedSpeechDao()
    runner = FedSpeechPipelineRunner(dao)
    results = runner.run()
    print(results)