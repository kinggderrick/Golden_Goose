import MetaTrader5 as mt5
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_connection(config_path="config.json"):
    try:
        with open(config_path) as f:
            config = json.load(f)['mt5']
        
        required_keys = ['login', 'password', 'server']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required key in config: {key}")
        
        if not mt5.initialize(login=config['login'], password=config['password'], server=config['server']):
            raise ConnectionError(f"MT5 failed to initialize: {mt5.last_error()}")
        
        logging.info(f"Connected to {config['server']} (Account #{config['login']})")
        
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to get account info")
        
        logging.info(f"Account Info: {account_info}")
        
        mt5.shutdown()
        logging.info("MT5 connection closed")
        
    except Exception as e:
        logging.error(f"Connection test failed: {str(e)}", exc_info=True)
        mt5.shutdown()
        raise

if __name__ == "__main__":
    test_connection()
