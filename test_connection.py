import MetaTrader5 as mt5
import logging
import json
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("connection_tests.log"),
        logging.StreamHandler()
    ]
)

def test_connection(config_path="config.json", retry_count=3, retry_delay=5):
    """
    Test connection to MetaTrader 5 terminal with retry logic
    
    Args:
        config_path (str): Path to configuration file
        retry_count (int): Number of connection attempts
        retry_delay (int): Delay between retries in seconds
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    attempt = 0
    
    while attempt < retry_count:
        attempt += 1
        logging.info(f"Connection attempt {attempt} of {retry_count}")
        
        try:
            # Check if config file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            # Load configuration
            with open(config_path) as f:
                config = json.load(f)
            
            # Check for MT5 section in config
            if 'mt5' not in config:
                raise KeyError("MT5 configuration section not found in config file")
                
            mt5_config = config['mt5']
            
            # Validate required configuration
            required_keys = ['login', 'password', 'server']
            missing_keys = [key for key in required_keys if key not in mt5_config]
            if missing_keys:
                raise KeyError(f"Missing required keys in config: {', '.join(missing_keys)}")
            
            # Shutdown any existing MT5 connections
            if mt5.initialize():
                mt5.shutdown()
                logging.info("Closed existing MT5 connection")
                time.sleep(1)  # Brief pause before reconnecting
            
            # Initialize connection
            logging.info(f"Connecting to server: {mt5_config['server']} with account #{mt5_config['login']}")
            if not mt5.initialize(
                login=int(mt5_config['login']), 
                password=mt5_config['password'], 
                server=mt5_config['server']
            ):
                error_code = mt5.last_error()
                error_message = f"MT5 failed to initialize. Error code: {error_code}"
                if error_code == 10000:
                    error_message += " (No error)"
                elif error_code == 10013:
                    error_message += " (Invalid login credentials)"
                elif error_code == 10018:
                    error_message += " (Connection failed)"
                raise ConnectionError(error_message)
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                raise ValueError("Failed to get account info")
            
            # Print detailed account information
            logging.info("CONNECTION SUCCESSFUL")
            logging.info(f"Account Info:")
            logging.info(f"  Server: {mt5_config['server']}")
            logging.info(f"  Account #: {account_info.login}")
            logging.info(f"  Balance: {account_info.balance}")
            logging.info(f"  Equity: {account_info.equity}")
            logging.info(f"  Margin: {account_info.margin}")
            logging.info(f"  Free Margin: {account_info.margin_free}")
            logging.info(f"  Margin Level: {account_info.margin_level}%")
            
            # Terminal info
            terminal_info = mt5.terminal_info()
            if terminal_info is not None:
                logging.info(f"Terminal Info:")
                logging.info(f"  Build: {terminal_info.build}")
                logging.info(f"  Connected: {terminal_info.connected}")
                logging.info(f"  Path: {terminal_info.path}")
            
            # Gracefully shutdown
            mt5.shutdown()
            logging.info("MT5 connection closed properly")
            
            return True
            
        except Exception as e:
            logging.error(f"Connection attempt {attempt} failed: {str(e)}")
            
            if attempt < retry_count:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for subsequent retries
                retry_delay = int(retry_delay * 1.5)
            else:
                logging.error(f"All {retry_count} connection attempts failed")
                # Ensure MT5 is shut down
                if mt5.initialize():
                    mt5.shutdown()
                return False
                
    return False

def check_symbols(config_path="config.json", symbols=None):
    """
    Check if specified symbols are available and get their properties
    
    Args:
        config_path (str): Path to configuration file
        symbols (list): List of symbol names to check, or None to use from config
    """
    try:
        # Connect to MT5
        if not test_connection(config_path):
            return
            
        # Load symbols from config if not provided
        if symbols is None:
            with open(config_path) as f:
                config = json.load(f)
            symbols = config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
            
        # Initialize MT5
        with open(config_path) as f:
            mt5_config = json.load(f)['mt5']
        
        if not mt5.initialize(
            login=int(mt5_config['login']), 
            password=mt5_config['password'], 
            server=mt5_config['server']
        ):
            logging.error(f"Failed to initialize MT5: {mt5.last_error()}")
            return
            
        # Check each symbol
        logging.info(f"Checking {len(symbols)} symbols...")
        for symbol in symbols:
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                logging.warning(f"Symbol {symbol} not found")
                continue
                
            # Log symbol properties
            logging.info(f"Symbol: {symbol}")
            logging.info(f"  Trade allowed: {'Yes' if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else 'No'}")
            logging.info(f"  Spread: {symbol_info.spread} points")
            logging.info(f"  Tick value: {symbol_info.trade_tick_value}")
            logging.info(f"  Min lot: {symbol_info.volume_min}")
            logging.info(f"  Max lot: {symbol_info.volume_max}")
            logging.info(f"  Lot step: {symbol_info.volume_step}")
            
        mt5.shutdown()
        
    except Exception as e:
        logging.error(f"Error checking symbols: {str(e)}")
        if mt5.initialize():
            mt5.shutdown()

if __name__ == "__main__":
    test_connection()
    # Uncomment to check symbols as well
    # check_symbols()