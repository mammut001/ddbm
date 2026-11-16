#!/bin/bash
# Stand-in for run_unet2.sh to test the driver on a CPU session: ~5 minutes of fake training progress.
F=/content/flags; mkdir -p $F /content/runs/sid_s1 /content/results /content/logs
touch $F/RUNNING_train
for i in 1 2 3 4 5 6 7; do
  echo "{\"step\": $((i * 100)), \"loss\": 0.1, \"lr\": 0.0002, \"it_s\": 3.6, \"time\": 0}" >> /content/runs/sid_s1/log.jsonl
  echo "checkpoint $i" > /content/runs/sid_s1/last.pt
  sleep 40
done
rm -f $F/RUNNING_train; touch $F/DONE_train $F/ALL_DONE
