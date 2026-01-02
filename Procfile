# Ticker is set via TICKER environment variable
worker: python main.py live --ticker ${TICKER:-DE} --lookback 1 --multiplier 0.4 --enable-stop-loss --consecutive-limit 10 --enable-regime-filter --regime-threshold 1.5 --regime-lookback 5 --trade-size 10
