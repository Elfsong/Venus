sudo umount /mnt/disks/venus_data 
sudo mdadm --create /dev/md0 --level=0 --raid-devices=2  /dev/disk/by-id/google-local-nvme-ssd-0  /dev/disk/by-id/google-local-nvme-ssd-1
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 /mnt/disks/venus_data/
sudo chmod a+w /mnt/disks/venus_data/



sudo umount /mnt/disks/lucky_space
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8  /dev/disk/by-id/google-local-nvme-ssd-*
sudo mkfs.ext4 -F /dev/md0
sudo mount /dev/md0 /mnt/disks/lucky_space
sudo chmod a+w /mnt/disks/lucky_space


python3 -m vllm.entrypoints.openai.api_server --port 18001 --model meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 \
--tensor-parallel-size 8 --pipeline-parallel-size 1 --swap-space 16 --gpu-memory-utilization 0.99 --dtype auto \
--served-model-name llama31-405b-fp8 --max-num-seqs 32 --max-model-len 32768 --max-num-batched-tokens 32768 \
--max-seq-len-to-capture 32768
--download_dir=/mnt/disks/lucky_space

