from frdr.database.schema.core import FedSpeech
from frdr.database.dao.fed_speech_dao import FedSpeechDao
from graph.create_pipeline import create_fed_speech_pipeline
from frdr.database.schema.core import FedSpeechAnalysis
from datetime import date

import pandas as pd

class FedSpeechPipelineRunner:
    def __init__(self, dao: FedSpeechDao, history_path="results_df.csv"):
        self.dao = dao
        # Load rolling baseline data for theme averaging
        self.history_df = pd.read_csv(history_path, parse_dates=["date"])
        self.graph = create_fed_speech_pipeline(self.history_df)

    def get_speeches(self, limit=10):
        speeches = self.dao.session.query(FedSpeech).order_by(FedSpeech.date.desc())
        return speeches

    def analyze_speech(self, speech: FedSpeech):
        paragraphs = speech.text.split("\n\n")
        # For each paragraph, pass its date if needed by ThemeAverager
        # Here we assume all paragraphs belong to the same speech/date
        result = self.graph.invoke({"paragraphs": paragraphs, "date": speech.date.strftime("%Y-%m-%d")})
        analysis = {
            "id": speech.id,
            "title": speech.title,
            "date": speech.date,
            "speaker": speech.speaker,
            "emphasis": result['emphasis'],
            "tags": result['tags'],
            "baseline": result.get('baseline', []),
            "surprise": result.get('surprise', [])
        }
        return analysis

    def classify_chair(self, speech_date: date) -> str:
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
                analysis_db.emphasis = analysis['emphasis']
                #analysis_db.tags = analysis['tags']
                analysis_db.chair = self.classify_chair(analysis['date'])
                # Optionally: add surprise, baseline etc. if your DB schema supports it

                self.dao.session.add(analysis_db)
                self.dao.session.commit()

            except Exception as e:
                print(f"❌ Failed to analyze speech ID {speech.id}: {e}")
                #raise e
        results_df = pd.DataFrame(results)
        results_df.to_csv('results_df.csv', index=False)
        return results_df

if __name__ == '__main__':
    dao = FedSpeechDao()
    runner = FedSpeechPipelineRunner(dao)
    results = runner.run()
    print(results)
