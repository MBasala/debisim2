ARG CUDA_VERSION=12.8.1

FROM fly0ut/debisim2p5:${CUDA_VERSION} AS base

COPY --chown=debisim:debisim examples/ /app/examples

ENTRYPOINT ["python", "run_dataset_generator.py"]
CMD ["--config", "configs/config_runpod_threat_demo.py", \
     "--sim_dir", "results/threat_demo", \
     "--num_bags", "1"]