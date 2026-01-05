"""
Telegram Notifier for Forex Signal Model.

Sends trading signals to your Telegram chat when BUY/SELL detected.

Setup:
1. Open Telegram and search for @BotFather
2. Send /newbot and follow instructions to create bot
3. Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
4. Start a chat with your bot and send any message
5. Run: python telegram_notifier.py --setup
6. Enter your bot token when prompted

Usage:
    from telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send_signal(signal_dict)
"""

import json
import logging
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Config file path
CONFIG_DIR = Path(__file__).parent / 'config'
TELEGRAM_CONFIG = CONFIG_DIR / 'telegram_config.json'


class TelegramNotifier:
    """
    Send trading signals via Telegram bot.
    """
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize notifier.
        
        Args:
            bot_token: Telegram bot token (optional, loads from config)
            chat_id: Telegram chat ID (optional, loads from config)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        # Load from config if not provided
        if not self.bot_token or not self.chat_id:
            self._load_config()
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def _load_config(self) -> None:
        """Load credentials from config file."""
        if TELEGRAM_CONFIG.exists():
            with open(TELEGRAM_CONFIG, 'r') as f:
                config = json.load(f)
                self.bot_token = self.bot_token or config.get('bot_token')
                self.chat_id = self.chat_id or config.get('chat_id')
    
    def _save_config(self) -> None:
        """Save credentials to config file."""
        CONFIG_DIR.mkdir(exist_ok=True)
        config = {
            'bot_token': self.bot_token,
            'chat_id': self.chat_id
        }
        with open(TELEGRAM_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to {TELEGRAM_CONFIG}")
    
    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a text message.
        
        Args:
            text: Message text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully
        """
        if not self.bot_token or not self.chat_id:
            logger.error("Bot token or chat ID not configured")
            return False
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Failed to send: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def send_signal(self, signal: Dict) -> bool:
        """
        Send a trading signal notification.
        
        Args:
            signal: Signal dictionary from SignalGenerator
            
        Returns:
            True if sent successfully
        """
        # Skip HOLD signals
        if signal.get('signal') == 'HOLD':
            return False
        
        # Format message
        emoji = '🟢' if signal['signal'] == 'BUY' else '🔴'
        
        message = f"""
{emoji} <b>FOREX SIGNAL ALERT</b>

<b>📊 {signal['ticker']}: {signal['signal']}</b>

💰 Entry: <code>{signal['entry_price']}</code>
🛑 Stop Loss: <code>{signal['stop_loss']}</code>
🎯 Take Profit: <code>{signal['take_profit']}</code>
📐 Position Size: <code>{signal['position_size']}</code> units
📈 Confidence: <code>{signal['confidence']}%</code>
🌐 Regime: <code>{signal.get('regime', 'N/A')}</code>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        return self.send_message(message.strip())
    
    def send_daily_summary(self, signals: List[Dict]) -> bool:
        """
        Send summary of all signals.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            True if sent successfully
        """
        actionable = [s for s in signals if s['signal'] != 'HOLD']
        
        if not actionable:
            # Send summary that no trades today
            message = f"""
📊 <b>DAILY SIGNAL SUMMARY</b>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}

No actionable signals today.
All pairs show low confidence - HOLD recommended.

Stay patient! 🎯
"""
        else:
            lines = []
            for s in actionable:
                emoji = '🟢' if s['signal'] == 'BUY' else '🔴'
                lines.append(f"{emoji} {s['ticker']}: {s['signal']} @ {s['entry_price']}")
            
            signals_text = '\n'.join(lines)
            message = f"""
📊 <b>DAILY SIGNAL SUMMARY</b>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}

{signals_text}

Check full details in signals folder!
"""
        
        return self.send_message(message.strip())
    
    def get_chat_id(self) -> Optional[str]:
        """
        Get chat ID from recent messages to the bot.
        
        Returns:
            Chat ID if found
        """
        url = f"{self.base_url}/getUpdates"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('result'):
                    # Get most recent chat
                    chat = data['result'][-1]['message']['chat']
                    return str(chat['id'])
        except Exception as e:
            logger.error(f"Error getting chat ID: {e}")
        
        return None
    
    def test_connection(self) -> bool:
        """Test the Telegram connection."""
        return self.send_message("🔔 Forex Signal Bot connected successfully!")


def setup_telegram():
    """Interactive setup for Telegram bot."""
    print("\n" + "=" * 50)
    print("🤖 TELEGRAM BOT SETUP")
    print("=" * 50)
    
    print("""
Steps to create your Telegram bot:
1. Open Telegram and search for @BotFather
2. Send /newbot
3. Follow instructions to name your bot
4. Copy the bot token you receive
5. Start a chat with your new bot (search for it)
6. Send any message to the bot (e.g., "hello")
""")
    
    # Get bot token
    bot_token = input("Paste your bot token here: ").strip()
    
    if not bot_token:
        print("❌ No token provided. Exiting.")
        return
    
    # Create notifier
    notifier = TelegramNotifier(bot_token=bot_token)
    
    # Get chat ID
    print("\nGetting your chat ID...")
    chat_id = notifier.get_chat_id()
    
    if not chat_id:
        print("❌ Could not find chat ID. Make sure you sent a message to your bot.")
        chat_id = input("Enter chat ID manually (or press Enter to retry): ").strip()
        
        if not chat_id:
            return
    
    print(f"✅ Found chat ID: {chat_id}")
    
    # Update notifier
    notifier.chat_id = chat_id
    
    # Test connection
    print("\nTesting connection...")
    if notifier.test_connection():
        print("✅ Test message sent! Check your Telegram.")
        
        # Save config
        notifier._save_config()
        print("\n✅ Setup complete! Your bot is ready to send signals.")
    else:
        print("❌ Failed to send test message. Check your token and try again.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Telegram notification for forex signals')
    parser.add_argument('--setup', action='store_true', help='Run interactive setup')
    parser.add_argument('--test', action='store_true', help='Send test message')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_telegram()
    elif args.test:
        notifier = TelegramNotifier()
        if notifier.test_connection():
            print("✅ Test message sent!")
        else:
            print("❌ Failed. Run --setup first.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
