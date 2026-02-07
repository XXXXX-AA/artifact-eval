# run_client.py
import os
from decentralized_worker_manager import DecentralizedWorkerManager

if __name__ == "__main__":
    
    rank        = int(os.getenv("RANK", os.getenv("FEDML_CLIENT_ID", "0")))
    world_size  = int(os.getenv("WORLD_SIZE", "4"))
    bootstrap   = os.getenv("BOOTSTRAP_ADDR", "tcp://<host>:<port>")
    compress    = float(os.getenv("COMPRESS_RATIO", "0.0"))

    mgr = DecentralizedWorkerManager(
        rank=rank,
        world_size=world_size,
        bootstrap=bootstrap,
        compress_ratio=compress,
        
    )
    mgr.run_sync()
