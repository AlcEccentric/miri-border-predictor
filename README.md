# miri-border-predictor

## Local testing guide

The `batch_knn_test.py` is for local testing purpose. Update the `CONFIG` in the main()
method to configure the testing.

Testing summary `batch_knn_summary.md` will be written to the root path.
Debugging plots will be generated under `debug` directory.

Although `knn_config.py` contains most of the configurations related to prediction,
the following methods in `main.py` contains configurations for data loading.
- get_min_event_id

When you change the loading related configurations in `batch_knn_test.py`,
remember to change the configuration values in `main.py` as well.