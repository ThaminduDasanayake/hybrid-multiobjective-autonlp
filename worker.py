import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from automl import HybridAutoML
from utils import DataLoader, to_python_type
from utils.job_manager import JobManager
from utils.logger import get_logger

# Configure logger
logger = get_logger("worker")

def main():
    parser = argparse.ArgumentParser(description="AutoML Worker Process")
    parser.add_argument("--job-id", required=True, help="Job ID")
    parser.add_argument("--config", required=True, help="Path to configuration JSON")
    parser.add_argument("--jobs-dir", default="jobs", help="Directory for job data")
    args = parser.parse_args()
    
    job_id = args.job_id
    config_path = args.config
    jobs_dir = args.jobs_dir
    
    logger.info(f"Worker started for job {job_id}")
    
    job_manager = JobManager(jobs_dir=jobs_dir)
    
    try:
        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Update status
        status = job_manager.get_status(job_id)
        status["status"] = "running"
        status["message"] = "Loading dataset..."
        job_manager.update_status(job_id, status)
        
        # Load data
        logger.info(f"Loading dataset: {config['dataset_name']}")
        data_loader = DataLoader(cache_dir="./data")
        X_train, y_train = data_loader.load_dataset(
            config['dataset_name'],
            subset='train',
            max_samples=config.get('max_samples', 2000)
        )
        
        # Define progress callback
        def progress_callback(progress_info):
            current_status = job_manager.get_status(job_id)
            if current_status:
                current_status.update({
                    "current_generation": progress_info.get("current_generation"),
                    "total_generations": progress_info.get("total_generations"),
                    "progress": progress_info.get("progress"),
                    "message": progress_info.get("message")
                })
                job_manager.update_status(job_id, current_status)
        
        # Initialize AutoML
        automl = HybridAutoML(
            X_train=X_train,
            y_train=y_train,
            population_size=config.get('population_size', 20),
            n_generations=config.get('n_generations', 10),
            bo_calls=config.get('bo_calls', 15),
            random_state=42,
            early_stopping=True,
            checkpoint_dir=os.path.join(jobs_dir, job_id, "checkpoints")
        )
        
        # Run AutoML
        status["message"] = "Running optimization..."
        job_manager.update_status(job_id, status)
        
        results = automl.run(callback=progress_callback)
        
        # Save results
        logger.info("Saving results...")
        result_path = os.path.join(jobs_dir, job_id, "result.json")
        with open(result_path, "w") as f:
            json.dump(to_python_type(results), f, indent=2)
            
        # Update status to completed
        status = job_manager.get_status(job_id)
        status["status"] = "completed"
        status["progress"] = 100
        status["message"] = "Optimization completed successfully"
        status["result_path"] = result_path
        job_manager.update_status(job_id, status)
        
        logger.info("Job completed successfully")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        logger.error(traceback.format_exc())
        
        # Update status to failed
        status = job_manager.get_status(job_id) or {}
        status["status"] = "failed"
        status["error"] = str(e)
        status["message"] = f"Failed: {str(e)}"
        job_manager.update_status(job_id, status)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
