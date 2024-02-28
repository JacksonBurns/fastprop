for benchmark in ara esol flash freesolv fubrain h2s hiv hopv15_subset pah pgp qm8 qm9 quantumscents sider ysi ocelot
do
    fastprop train $benchmark/$benchmark.yml >> $benchmark/run_log.txt 2>&1 || echo "failed $benchmark"
done
