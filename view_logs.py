import time
import sys
from huggingface_hub import HfApi

def tail_logs(job_id, namespace):
    api = HfApi()
    print(f"Streaming logs for Job {job_id}...")
    print("-" * 50)
    
    # We maintain the raw generator block to stream chunks
    try:
        # fetch_job_logs stream=True is the default behavior.
        for chunk in api.fetch_job_logs(job_id=job_id, namespace=namespace):
            # chunk might be bytes
            if isinstance(chunk, bytes):
                print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
            else:
                print(chunk, end="", flush=True)
                
    except KeyboardInterrupt:
        print("\nStopped streaming.")
    except Exception as e:
        print(f"\nError streaming logs: {e}")

if __name__ == "__main__":
    # Your current active job id:
    JOB_ID = "69eceaafd70108f37acde9f6"
    NAMESPACE = "s1nn3rx69"
    tail_logs(JOB_ID, NAMESPACE)
