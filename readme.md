# 균일하지 않은 비독립항등분포 데이터 스트림에 대한 실시간 테스트 단계 도메인 적응 기술
## TTA for Noisy and Non IID data stream (Year 2)
### Environment
You can get the required environment using conda and the provided environment.yml file

`conda env create -f environment.yml -n medai2`

### Datasets
The base dataset for this replication is CIFAR10(-C). Source training of the model is on CIFAR10 *only*, and corresponding noises and corruptions will be based on CIFAR10 settings\

To download and process CIFAR-10, run the following script:\
`bash download_cifar10c.sh`\

### Run
To run experiments on temporally-correlated data, use the script below: \
`python main.py --iabn --tgt_use_learned_stats --optimize --adapt --distribution dirichlet`\
Alternatively, change to `--distribution random` to test on uniform random data

After running at least one of the above experiments, you can use the same source-trained base model for other experiments by setting the `--use_checkpoint` flag

Additionally, you can set the seed (default: `0`) using `--seed` and the device (default: `cuda`) using `--device`

### Logs
Logs are automatically uploaded to wandb if wandb api key is set in environment.

Alternatively, results are stored in `./logs`
