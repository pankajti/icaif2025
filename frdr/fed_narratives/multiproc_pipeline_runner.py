import logging
import os
from datetime import date
from multiprocessing import Pool, Manager, cpu_count
import pandas as pd
from tqdm import tqdm
from sqlalchemy import not_

# IMPORTANT: These imports need to be here for the worker processes to find them
from frdr.database.dao.fed_speech_dao import FedSpeechDao
from frdr.database.schema.core import FedSpeech, FedSpeechAnalysis
from graph.create_pipeline import create_fed_speech_pipeline

# --- Configuration ---
LOG_FILE = "fed_speech_analysis.log"
DEFAULT_NUM_PROCESSES = max(1, cpu_count() - 1)
BATCH_SIZE = 10 # Define the batch size for database commits

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

# Global variables to hold DAO and Graph for each worker process
# These will be initialized once per process by the initializer function
_process_dao = None
_process_graph = None

def _worker_initializer():
    """
    Initializes a database DAO and the analysis graph for each worker process.
    This function is called once per child process when the pool starts.
    """
    global _process_dao
    global _process_graph
    _process_dao = FedSpeechDao()
    _process_graph = create_fed_speech_pipeline()
    logger.info(f"Worker process {os.getpid()} initialized DAO and Graph.")


def _analyze_speech_worker(speech: FedSpeech):
    """
    Worker function to analyze a single speech.
    This function will be run by each process in the pool.
    It accesses the DAO and graph through the global variables initialized by _worker_initializer.
    """
    global _process_dao
    global _process_graph

    if _process_dao is None or _process_graph is None:
        _worker_initializer() # Fallback, should be handled by Pool's initializer

    try:
        paragraphs = speech.text.split("\n\n")
        result = _process_graph.invoke({"paragraphs": paragraphs}) # Use the process-specific graph instance
        analysis = {
            "id": speech.id,
            "title": speech.title,
            "date": speech.date,
            "speaker": speech.speaker,
            "emphasis": result['emphasis'],
            "tags": result['tags']
        }
        logger.info(f"Successfully analyzed speech ID {speech.id} in process {os.getpid()}")
        return analysis
    except Exception as e:
        logger.error(f"❌ Failed to analyze speech ID {speech.id} in process {os.getpid()}: {e}")
        # Rollback the session in case of an error within the worker, as a good practice
        if _process_dao and _process_dao.session:
            _process_dao.session.rollback()
        return None


class FedSpeechPipelineRunner:
    def __init__(self, main_dao: FedSpeechDao):
        self.main_dao = main_dao

    def get_speeches_to_analyze(self, limit=None):
        """
        Retrieves speeches that have not yet been analyzed.
        """
        logger.info("Querying for already analyzed speech IDs...")
        analyzed_speech_ids = self.main_dao.session.query(FedSpeechAnalysis.speech_id).distinct().all()
        analyzed_speech_ids = {s_id for (s_id,) in analyzed_speech_ids}
        logger.info(f"Found {len(analyzed_speech_ids)} speeches already analyzed.")

        query = self.main_dao.session.query(FedSpeech).order_by(FedSpeech.date.desc())

        if analyzed_speech_ids:
            query = query.filter(not_(FedSpeech.id.in_(analyzed_speech_ids)))
            logger.info("Filtering out already analyzed speeches...")

        if limit is not None:
            query = query.limit(limit)
            logger.info(f"Applying limit of {limit} to remaining speeches.")

        speeches_to_process = query.all()
        logger.info(f"Identified {len(speeches_to_process)} speeches for analysis.")
        return speeches_to_process

    def classify_chair(self, speech_date: date) -> str:
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

    def run(self, limit=None, num_processes=DEFAULT_NUM_PROCESSES, batch_size=BATCH_SIZE):
        # Fetch only the speeches that need to be analyzed
        speeches = self.get_speeches_to_analyze(limit)

        if not speeches:
            logger.info("No new speeches found to analyze. Exiting.")
            return pd.DataFrame() # Return an empty DataFrame

        logger.info(f"Starting analysis of {len(speeches)} speeches using {num_processes} processes with batch size {batch_size}.")

        all_results_for_df = [] # To collect all successful analyses for the final DataFrame
        processed_count = 0

        with Manager() as manager:
            # We don't need a shared list here directly for collecting results,
            # as imap_unordered provides them one by one.
            # The manager is still available if you need other shared objects.

            with Pool(processes=num_processes, initializer=_worker_initializer) as pool:
                results_iterator = pool.imap_unordered(_analyze_speech_worker, speeches)

                # Iterate through results as they come in and commit in batches
                for analysis in tqdm(results_iterator, total=len(speeches), desc="Analyzing Speeches"):
                    if analysis is not None:
                        all_results_for_df.append(analysis) # Collect for final DataFrame

                        try:
                            analysis_db = FedSpeechAnalysis(speech_id=analysis['id'])
                            analysis_db.emphasis = analysis['emphasis']
                            # analysis_db.tags = analysis['tags']
                            analysis_db.chair = self.classify_chair(analysis['date'])
                            self.main_dao.session.add(analysis_db) # Add to the main process's session
                            processed_count += 1

                            # Commit in batches
                            if processed_count % batch_size == 0:
                                self.main_dao.session.commit()
                                logger.info(f"Committed {processed_count} analyses to the database so far.")

                        except Exception as e:
                            self.main_dao.session.rollback() # Rollback on error in batch
                            logger.error(f"❌ Failed to prepare/commit database entry for speech ID {analysis['id']}: {e}. Rolling back batch.")
                            # Decide whether to re-raise or continue. For robustness, continuing might be better.

        # After the loop, commit any remaining items in the last partial batch
        if processed_count % batch_size != 0:
            try:
                self.main_dao.session.commit()
                logger.info(f"Committed final {processed_count % batch_size} analyses to the database.")
            except Exception as e:
                self.main_dao.session.rollback()
                logger.critical(f"FATAL ERROR: Failed to commit remaining analyses to the database: {e}")
                #raise e

        if not all_results_for_df:
            logger.warning("No speeches were successfully analyzed. No data to commit or save.")
            return pd.DataFrame()

        logger.info(f"Total {len(all_results_for_df)} analyses processed and saved.")

        results_df = pd.DataFrame(all_results_for_df)
        output_csv_path = 'results_df.csv'
        results_df.to_csv(output_csv_path, index=False)
        logger.info(f"Results saved to {output_csv_path}")

        return results_df


if __name__ == '__main__':
    if __name__ == '__main__':
        main_dao_instance = FedSpeechDao()

        num_desired_processes = 4
        speeches_limit = None # Set to None to process all unanalyzed speeches
        custom_batch_size = 10 # You can override the default BATCH_SIZE here

        runner = FedSpeechPipelineRunner(main_dao_instance)
        results = runner.run(
            limit=speeches_limit,
            num_processes=num_desired_processes,
            batch_size=custom_batch_size # Pass the custom batch size
        )

        if not results.empty:
            logger.info("\nFinal Results DataFrame:")
            logger.info(results.head())
        else:
            logger.info("No results generated.")